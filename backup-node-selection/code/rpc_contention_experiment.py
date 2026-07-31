"""Standalone localhost RPC-contention benchmark for backup-node selection.

Runs real Backup XML-RPC servers (exp_package's 2PC backup_2pc.Backup, unmodified)
on localhost ports, one per unique backup host in a mapping, then fires concurrent
bursts of update_state() replicate calls per primary against its assigned backup.
This is the same call Primary.replicate() makes, without the Mininet network path.

Each primary uses a burst of BURST_SIZE simultaneous in-flight requests so a backup
shared by multiple primaries accumulates a real request queue at its single-threaded
SimpleXMLRPCServer.

Compares every registered backup-selection method (baseline / random / round_robin /
heuristic / optimal, plus rl if a trained policy is available) across multiple trials,
with Mann-Whitney U significance testing and bootstrap confidence intervals on the
per-call latency distributions.
"""

import argparse
import copy
import csv
import multiprocessing
import os
import random as random_module
import statistics
import time
import xmlrpc.client
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from exp_package.Two_phase_commit.backup_2pc import Backup

import backup_selector
import config
import topology_features

DUMMY_STATE = {"10.0.0.1,10.0.0.2,1": 42}
RESULTS_DIR = "results_rpc"
RESULTS_CSV = "{}/rpc_contention_raw.csv".format(RESULTS_DIR)
RESULTS_FIELDS = ["method", "context_seed", "trial", "primary", "backup", "backup_load", "latency_ms"]


def _make_update_state(extra_delay_s=0.0):
    """Backup.update_state without the print() calls (which would flood stdout over
    thousands of RPC calls), plus an optional server-side delay so degraded backups in
    the scaled scenario actually run slower rather than only scoring worse."""
    def _update_state(state):
        if extra_delay_s:
            time.sleep(extra_delay_s)
        return True
    return _update_state


def _serve_backup(ip, port, extra_delay_s=0.0):
    """Entry point for each backup's own OS process. Must be a top-level function (not a
    closure) so it can be pickled for multiprocessing.Process. Running each backup as a
    separate process avoids the GIL-sharing artifacts threads would introduce."""
    backup = Backup(ip, port)
    backup.server.register_multicall_functions()
    backup.server.register_function(_make_update_state(extra_delay_s), "update_state")
    backup.server.serve_forever()


def start_backup_servers(unique_backups, base_port, slow_hosts=None, slow_delay_s=0.05):
    """One Backup XML-RPC server per unique backup host, each in its own OS
    process on its own port. Hosts in `slow_hosts` get an artificial per-call delay
    (simulating a genuinely degraded/overloaded machine). Returns ({backup_name:
    port}, [processes])."""
    slow_hosts = slow_hosts or set()
    ports = {}
    processes = []
    for i, backup_name in enumerate(sorted(unique_backups)):
        port = base_port + i
        delay = slow_delay_s if backup_name in slow_hosts else 0.0
        proc = multiprocessing.Process(target=_serve_backup, args=("127.0.0.1", port, delay), daemon=True)
        proc.start()
        processes.append(proc)
        ports[backup_name] = port
    return ports, processes


MAX_RETRY_ATTEMPTS = 8
FAILED_CALL_PENALTY_MS = 10000.0


def replicate_burst(port, n_calls, burst_size):
    """Fires n_calls requests using burst_size concurrent in-flight connections, so a
    shared backend accumulates a request queue instead of seeing calls one at a time.
    Each worker lane keeps one persistent ServerProxy for all its calls; a fresh proxy
    per call would add a TCP handshake every time and swamp the measurement with
    connection-thrash noise.

    Retries with jittered exponential backoff up to MAX_RETRY_ATTEMPTS. A call that never
    succeeds is recorded as a failure with a large penalty latency instead of crashing the
    run, so heavy contention shows up as elevated latency and a nonzero failure count."""

    def worker(n_calls_for_this_worker):
        proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:{}".format(port))
        latencies = []
        failures = 0
        for _ in range(n_calls_for_this_worker):
            start = time.perf_counter()
            succeeded = False
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    proxy.update_state(DUMMY_STATE)
                    succeeded = True
                    break
                except (ConnectionResetError, ConnectionRefusedError, OSError):
                    if attempt == MAX_RETRY_ATTEMPTS - 1:
                        break
                    backoff = min(0.05 * (2 ** attempt), 1.0) + random_module.uniform(0, 0.02)
                    time.sleep(backoff)
                    proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:{}".format(port))
            if succeeded:
                latencies.append((time.perf_counter() - start) * 1000)
            else:
                failures += 1
                latencies.append(FAILED_CALL_PENALTY_MS)
        return latencies, failures

    calls_per_worker = [n_calls // burst_size] * burst_size
    for i in range(n_calls % burst_size):
        calls_per_worker[i] += 1

    all_latencies = []
    total_failures = 0
    with ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(worker, c) for c in calls_per_worker if c > 0]
        for future in futures:
            latencies, failures = future.result()
            all_latencies.extend(latencies)
            total_failures += failures

    if total_failures:
        print("  [warning] {} call(s) failed after {} retries against port {}".format(
            total_failures, MAX_RETRY_ATTEMPTS, port))

    return all_latencies


def backup_load(mapping):
    load = defaultdict(list)
    for primary, backup in mapping.items():
        load[backup].append(primary)
    return load


def run_trial(mapping, base_port, n_calls, burst_size, slow_hosts=None):
    unique_backups = set(mapping.values())
    ports, processes = start_backup_servers(unique_backups, base_port, slow_hosts=slow_hosts)
    time.sleep(0.3)  # let server processes finish starting up and binding

    try:
        latencies_by_primary = {}
        with ThreadPoolExecutor(max_workers=len(mapping)) as primary_pool:
            futures = {
                primary: primary_pool.submit(replicate_burst, ports[backup], n_calls, burst_size)
                for primary, backup in mapping.items()
            }
            for primary, future in futures.items():
                latencies_by_primary[primary] = future.result()
    finally:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        time.sleep(0.2)  # let the OS fully release ports/sockets before the next method

    return latencies_by_primary


def record_raw(method, context_seed, trial, mapping, latencies_by_primary):
    load = backup_load(mapping)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        for primary, latencies in latencies_by_primary.items():
            backup = mapping[primary]
            n_sharing = len(load[backup])
            for latency in latencies:
                writer.writerow({
                    "method": method,
                    "context_seed": context_seed,
                    "trial": trial,
                    "primary": primary,
                    "backup": backup,
                    "backup_load": n_sharing,
                    "latency_ms": latency,
                })


_rl_model_cache = {}


def _rl_mapping(ctx, hosts, primaries, model_path="rl_model.zip", algorithm="ppo"):
    cache_key = (model_path, algorithm)
    if cache_key not in _rl_model_cache:
        import rl_backup_selector
        _rl_model_cache[cache_key] = (
            rl_backup_selector.load_rl_model(model_path, algorithm), rl_backup_selector)
    model, rl_backup_selector = _rl_model_cache[cache_key]
    return rl_backup_selector.predict_mapping(model, hosts, primaries, ctx)


def _rl_dqn_mapping(ctx, hosts, primaries, model_path="rl_model_dqn.zip"):
    """DQN counterpart of _rl_mapping, reported under the rl_dqn method name. "rl" is always
    PPO and "rl_dqn" is always DQN, each loaded from its own checkpoint path, so the two are
    never silently pointed at the same model."""
    return _rl_mapping(ctx, hosts, primaries, model_path=model_path, algorithm="dqn")


# Every method function takes (ctx, hosts, primaries) so the same registry works for
# both the fixed 9-host DEFT topology and an arbitrary scaled/synthetic one.
METHODS = {
    "baseline": lambda ctx, hosts, primaries: backup_selector.get_baseline_mapping(),
    "concentrated": lambda ctx, hosts, primaries: backup_selector.concentrated_mapping(primaries, hosts),
    "random": lambda ctx, hosts, primaries: backup_selector.random_mapping(primaries, hosts, seed=123),
    "round_robin": lambda ctx, hosts, primaries: backup_selector.round_robin_mapping(primaries, hosts),
    "heuristic": lambda ctx, hosts, primaries: backup_selector.select_backups(primaries, hosts, ctx=ctx),
    "optimal": lambda ctx, hosts, primaries: backup_selector.optimal_mapping(primaries, hosts, ctx=ctx),
    "rl": _rl_mapping,
    "rl_dqn": _rl_dqn_mapping,
}


def bootstrap_ci(values, n_resamples=5000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def rank_biserial_effect_size(a, b, u_stat=None):
    """Effect size companion to Mann-Whitney U: rank-biserial correlation r in
    [-1, 1]. |r| ~0.1/0.3/0.5 are conventionally small/medium/large. With large n,
    p-values are almost always significant regardless of practical magnitude, so r is
    the number to read for whether a difference matters."""
    n1, n2 = len(a), len(b)
    if u_stat is None:
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 1 - (2 * u_stat) / (n1 * n2)


def run_experiment(context_seeds, trials, n_calls, burst_size, methods, order_seed=0,
                    hosts=None, primaries=None, graph=None, overrides=None, slow_hosts=None,
                    rl_model_path="rl_model.zip", rl_dqn_model_path="rl_model_dqn.zip"):
    """Runs the full method comparison independently across each seed in context_seeds
    (a fresh topology/resource realization and a fresh mapping per method for each seed),
    then aggregates across all (context_seed x trial) combinations. Running many trials of
    one fixed instance only captures RPC-timing noise; varying the seed is what shows the
    result holds across different problem instances."""
    hosts = hosts if hosts is not None else config.HOSTS
    primaries = primaries if primaries is not None else config.PRIMARIES

    method_fns = dict(METHODS)
    method_fns["rl"] = lambda ctx, hosts, primaries: _rl_mapping(ctx, hosts, primaries, model_path=rl_model_path, algorithm="ppo")
    method_fns["rl_dqn"] = lambda ctx, hosts, primaries: _rl_dqn_mapping(ctx, hosts, primaries, model_path=rl_dqn_model_path)

    all_latencies = {name: [] for name in methods}
    base_port = 9001
    order_rng = random_module.Random(order_seed)

    for context_seed in context_seeds:
        ctx = topology_features.build_feature_context(hosts=hosts, seed=context_seed, graph=graph, overrides=overrides)
        method_fns["random"] = lambda ctx, hosts, primaries, cs=context_seed: backup_selector.random_mapping(primaries, hosts, seed=cs)
        # deepcopy per method: heuristic and rl mutate ctx in place via _apply_load_penalty
        # (capacity -= 20; health *= 0.95), so a shared ctx would make each method's scores
        # depend on which methods ran before it.
        mappings = {name: method_fns[name](copy.deepcopy(ctx), hosts, primaries) for name in methods}

        print("\n--- context_seed={} ---".format(context_seed))
        if slow_hosts:
            for name in methods:
                n_slow_uses = sum(1 for b in mappings[name].values() if b in slow_hosts)
                print("  {:<12} uses degraded host(s) {}/{} times, mapping: {}".format(
                    name, n_slow_uses, len(mappings[name]), mappings[name]))
        else:
            for name, mapping in mappings.items():
                print("  {:<12} mapping: {}".format(name, mapping))

        for trial in range(trials):
            # Randomize per-trial method order so no method is favored or penalized by
            # when it runs, since background OS load varies over a long run.
            trial_order = list(methods)
            order_rng.shuffle(trial_order)

            for name in trial_order:
                mapping = mappings[name]
                latencies_by_primary = run_trial(mapping, base_port, n_calls, burst_size, slow_hosts=slow_hosts)
                record_raw(name, context_seed, trial, mapping, latencies_by_primary)
                for vals in latencies_by_primary.values():
                    all_latencies[name].extend(vals)
                base_port += len(set(mapping.values())) + 1
            print("  trial {}/{} done (order: {})".format(trial + 1, trials, trial_order))

    n_seeds = len(context_seeds)
    print("\n\n===== SUMMARY across {} context seeds x {} trials ({} calls/primary, burst={}) =====".format(
        n_seeds, trials, n_calls, burst_size))

    print("\n{:<12} {:>8} {:>10} {:>10} {:>10} {:>10} {:>24}".format(
        "method", "n", "mean_ms", "median_ms", "p95_ms", "p99_ms", "95% CI of mean"))
    for name in methods:
        vals = all_latencies[name]
        arr = np.asarray(vals)
        lo, hi = bootstrap_ci(vals)
        print("{:<12} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>24}".format(
            name, len(vals), arr.mean(), np.median(arr),
            np.percentile(arr, 95), np.percentile(arr, 99),
            "[{:.4f}, {:.4f}]".format(lo, hi),
        ))

    print("\n===== Pairwise Mann-Whitney U tests + rank-biserial effect size =====")
    print("(|r| ~0.1/0.3/0.5 ~ small/medium/large; p-values alone are near-meaningless at this n)")
    method_list = list(methods)
    for i in range(len(method_list)):
        for j in range(i + 1, len(method_list)):
            a, b = method_list[i], method_list[j]
            u_stat, p_value = stats.mannwhitneyu(all_latencies[a], all_latencies[b], alternative="two-sided")
            r = rank_biserial_effect_size(all_latencies[a], all_latencies[b], u_stat=u_stat)
            sig = "significant (p<0.05)" if p_value < 0.05 else "not significant"
            print("{:<12} vs {:<12} p={:.6f}  r={:+.4f}  {}".format(a, b, p_value, r, sig))

    print("\n===== Latency broken down by backup_load (pooled across all methods/seeds) =====")
    print("This isolates the real mechanism: does sharing a backup among more primaries")
    print("increase latency, independent of which method produced that sharing pattern?")
    by_load = defaultdict(list)
    raw_rows = list(csv.DictReader(open(RESULTS_CSV)))
    for row in raw_rows:
        by_load[int(row["backup_load"])].append(float(row["latency_ms"]))
    print("{:>6} {:>8} {:>10} {:>10} {:>10} {:>10}".format("load", "n", "mean_ms", "median_ms", "p95_ms", "p99_ms"))
    for load in sorted(by_load):
        arr = np.asarray(by_load[load])
        print("{:>6} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            load, len(arr), arr.mean(), np.median(arr), np.percentile(arr, 95), np.percentile(arr, 99)
        ))

    return all_latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-seeds", type=int, nargs="+", default=[42],
                         help="one topology/resource realization per seed; results aggregate across all of them")
    parser.add_argument("--trials", type=int, default=5, help="RPC-timing trials per context seed")
    parser.add_argument("--calls", type=int, default=200, help="replicate() calls per primary per trial")
    parser.add_argument("--burst", type=int, default=10, help="concurrent in-flight requests per primary")
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--n-hosts", type=int, default=None, help="use a scaled synthetic topology instead of config.py's 9 hosts")
    parser.add_argument("--n-primaries", type=int, default=None)
    parser.add_argument("--n-degraded", type=int, default=0, help="mark this many backup-eligible hosts as artificially degraded (low health/capacity)")
    parser.add_argument("--scale-seed", type=int, default=0)
    parser.add_argument("--rl-model", default="rl_model.zip", help="PPO checkpoint for the 'rl' method")
    parser.add_argument("--rl-dqn-model", default="rl_model_dqn.zip", help="DQN checkpoint for the 'rl_dqn' method")
    args = parser.parse_args()

    # Save this run's output to findings/ as well as the terminal (see run_logging.py).
    import run_logging
    run_logging.capture_to_findings("latency_{}hosts".format(args.n_hosts or 9))

    hosts, primaries, graph, overrides, slow_hosts = None, None, None, None, None
    if args.n_hosts is not None:
        hosts, primaries, graph = topology_features.generate_scaled_topology(
            n_hosts=args.n_hosts, n_primaries=args.n_primaries, seed=args.scale_seed
        )
        if args.n_degraded > 0:
            backup_pool = [h for h in hosts if h not in primaries]
            degraded = backup_pool[:args.n_degraded]
            # keep max_capacity in the normal range so the capacity ratio
            # (capacity/max_capacity), not just the raw number, is also low for degraded hosts.
            overrides = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded}
            slow_hosts = set(degraded)

    run_experiment(
        args.context_seeds, args.trials, args.calls, args.burst, args.methods,
        hosts=hosts, primaries=primaries, graph=graph, overrides=overrides, slow_hosts=slow_hosts,
        rl_model_path=args.rl_model, rl_dqn_model_path=args.rl_dqn_model,
    )
