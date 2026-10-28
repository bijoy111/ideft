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
import time
import xmlrpc.client
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from exp_package.Two_phase_commit.backup_2pc import Backup

import analysis
import backup_selector
import config
import topology_features

DUMMY_STATE = {"10.0.0.1,10.0.0.2,1": 42}
RESULTS_DIR = "results_rpc"
RESULTS_CSV = "{}/rpc_contention_raw.csv".format(RESULTS_DIR)
RESULTS_FIELDS = ["method", "context_seed", "trial", "primary", "backup", "backup_load",
                  "service_ms", "retry_wait_ms", "wall_ms", "attempts", "succeeded"]


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

# xmlrpc.client.ServerProxy applies no socket timeout, so a connection the server has
# accepted but is not reading stalls the client forever. DEFT's backup is a
# single-threaded SimpleXMLRPCServer, so one wedged connection blocks the accept loop
# and every other client queues behind it indefinitely. At 128 hosts with 80 primaries
# on one backup this deadlocked a run outright: over an 80 s observation the server's
# CPU time stayed frozen at 8 s with 87 sockets in SYN-SENT, 20 established, and zero
# forward progress. A production 2PC client would bound the wait and retry, so the
# benchmark client does too. The timeout sits well above the largest genuine service
# time observed anywhere in these experiments (7.2 s), so it bounds the deadlock
# without censoring real slow calls. This changes the benchmark client only; the
# backup server under test is still DEFT's own, unmodified.
CALL_TIMEOUT_S = 20.0


class _TimeoutTransport(xmlrpc.client.Transport):
    """xmlrpc.client.Transport that applies a socket timeout to its connections."""

    def __init__(self, timeout=None):
        super().__init__()
        # Resolved at call time, not bound as a default at definition time, so
        # CALL_TIMEOUT_S stays overridable by tests and callers.
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = CALL_TIMEOUT_S if self._timeout is None else self._timeout
        return conn


def make_proxy(port):
    return xmlrpc.client.ServerProxy(
        "http://127.0.0.1:{}".format(port), transport=_TimeoutTransport())

# A call that never succeeds is recorded as censored (succeeded=False) and contributes
# only to the failure rate. The previous harness imputed a 10,000 ms sentinel into the
# latency sample instead, which silently contaminated every percentile computed from it:
# a p99 above that sentinel could not be distinguished from a p99 made of sentinels.


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
        proxy = make_proxy(port)
        records = []
        failures = 0
        for _ in range(n_calls_for_this_worker):
            call_start = time.perf_counter()
            service_s = None
            attempts = 0
            for attempt in range(MAX_RETRY_ATTEMPTS):
                attempts += 1
                attempt_start = time.perf_counter()
                try:
                    proxy.update_state(DUMMY_STATE)
                    # Time the successful attempt alone. This is the backup's real
                    # service + queueing delay; it is the only quantity here that is a
                    # property of the server under test.
                    service_s = time.perf_counter() - attempt_start
                    break
                except (ConnectionResetError, ConnectionRefusedError, OSError):
                    if attempt == MAX_RETRY_ATTEMPTS - 1:
                        break
                    backoff = min(0.05 * (2 ** attempt), 1.0) + random_module.uniform(0, 0.02)
                    time.sleep(backoff)
                    proxy = make_proxy(port)
            wall_s = time.perf_counter() - call_start
            if service_s is None:
                failures += 1
                records.append({
                    "service_ms": "", "retry_wait_ms": wall_s * 1000,
                    "wall_ms": wall_s * 1000, "attempts": attempts, "succeeded": False,
                })
            else:
                # Everything outside the successful attempt is failed attempts plus this
                # client's own exponential-backoff sleep: a property of the retry policy,
                # not of the backup. Keeping it fused into one number (as the previous
                # timer, started outside the retry loop, did) attributes the client's
                # sleep constants to backup queueing delay.
                records.append({
                    "service_ms": service_s * 1000,
                    "retry_wait_ms": (wall_s - service_s) * 1000,
                    "wall_ms": wall_s * 1000, "attempts": attempts, "succeeded": True,
                })
        return records, failures

    calls_per_worker = [n_calls // burst_size] * burst_size
    for i in range(n_calls % burst_size):
        calls_per_worker[i] += 1

    all_records = []
    total_failures = 0
    with ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(worker, c) for c in calls_per_worker if c > 0]
        for future in futures:
            records, failures = future.result()
            all_records.extend(records)
            total_failures += failures

    if total_failures:
        print("  [warning] {} call(s) failed after {} retries against port {}".format(
            total_failures, MAX_RETRY_ATTEMPTS, port))

    return all_records


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
        records_by_primary = {}
        with ThreadPoolExecutor(max_workers=len(mapping)) as primary_pool:
            futures = {
                primary: primary_pool.submit(replicate_burst, ports[backup], n_calls, burst_size)
                for primary, backup in mapping.items()
            }
            for primary, future in futures.items():
                records_by_primary[primary] = future.result()
    finally:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        time.sleep(0.2)  # let the OS fully release ports/sockets before the next method

    return records_by_primary


def reset_raw_csv():
    """Start each experiment invocation from an empty raw file.

    The previous version appended forever and then read the whole file back for its
    load-breakdown table, so that table silently pooled calls from earlier runs at
    other scales and other harness versions.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULTS_FIELDS).writeheader()


def record_raw(method, context_seed, trial, mapping, records_by_primary):
    load = backup_load(mapping)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        for primary, records in records_by_primary.items():
            backup = mapping[primary]
            n_sharing = len(load[backup])
            for rec in records:
                row = {
                    "method": method,
                    "context_seed": context_seed,
                    "trial": trial,
                    "primary": primary,
                    "backup": backup,
                    "backup_load": n_sharing,
                }
                row.update(rec)
                writer.writerow(row)


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


# Statistics live in analysis.py. They operate on runs -- one (method, context_seed,
# trial) execution -- rather than on individual calls. Pooling calls treated thousands
# of correlated requests against one server process as independent samples, which drove
# p-values to zero without adding evidence; see analysis.py's module docstring.


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

    reset_raw_csv()
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
                records_by_primary = run_trial(mapping, base_port, n_calls, burst_size, slow_hosts=slow_hosts)
                record_raw(name, context_seed, trial, mapping, records_by_primary)
                base_port += len(set(mapping.values())) + 1
            print("  trial {}/{} done (order: {})".format(trial + 1, trials, trial_order))

    n_seeds = len(context_seeds)
    print("\n\n===== RUN SUMMARY: {} context seeds x {} trials ({} calls/primary, burst={}) =====".format(
        n_seeds, trials, n_calls, burst_size))
    print("Independent observations per method: {} runs ({} seeds x {} trials).".format(
        n_seeds * trials, n_seeds, trials))

    # Pass the path, not a materialised list: at 128 hosts this file holds on the
    # order of a million call records.
    analysis.print_decomposition(RESULTS_CSV)
    analysis.print_report(RESULTS_CSV, metric="service_ms", title="service latency")

    return RESULTS_CSV


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
