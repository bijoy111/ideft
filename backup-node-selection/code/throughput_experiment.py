"""Throughput and failure-rate benchmark for backup-node selection methods.

Complements rpc_contention_experiment.py (per-call latency) with the other two
metrics DEFT's base paper (CNSM'23) reports: throughput and packet drop. Runs each
primary's burst workers for a fixed wall-clock duration instead of a fixed call
count, so throughput (successful calls/sec) is directly meaningful; a fixed
call count conflates "slower" with "fewer calls attempted".

On "packet drop": this testbed is RPC-level (TCP), not DEFT's packet-level
input-buffer-overflow mechanism (which needs the Mininet/stamped-packet path the
RPC-isolation testbed bypasses, see EXPERIMENTS.md Sec 3.1/5). The closest metric
here is RPC call failure rate: calls that time out or fail after retries under
sustained load. It reflects the same phenomenon (an overloaded backup failing to
keep up) but is not a literal reproduction of DEFT's packet-drop metric.
"""

import argparse
import copy
import csv
import os
import random as random_module
import socket
import time
import xmlrpc.client
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import analysis
import backup_selector
import config
import topology_features
from rpc_contention_experiment import (
    DUMMY_STATE, METHODS, _rl_mapping, backup_load, start_backup_servers,
)

RESULTS_DIR = "results_rpc"
THROUGHPUT_CSV = "{}/throughput_raw.csv".format(RESULTS_DIR)
THROUGHPUT_FIELDS = [
    "method", "context_seed", "trial", "primary", "backup", "backup_load",
    "successful_calls", "failed_calls", "duration_s", "throughput_cps", "failure_rate_pct",
]

CALL_TIMEOUT_S = 2.0


def throughput_burst(port, duration_s, burst_size):
    """burst_size concurrent worker lanes hammer one backup for a fixed
    wall-clock duration, counting successes and failures (timeouts/connection
    errors, retried once via a fresh connection before counting as failed)."""

    def worker():
        socket.setdefaulttimeout(CALL_TIMEOUT_S)
        proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:{}".format(port))
        successes = 0
        failures = 0
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            try:
                proxy.update_state(DUMMY_STATE)
                successes += 1
            except (ConnectionResetError, ConnectionRefusedError, OSError, socket.timeout):
                failures += 1
                proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:{}".format(port))
        return successes, failures

    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(worker) for _ in range(burst_size)]
        for future in futures:
            s, f = future.result()
            successes += s
            failures += f

    return successes, failures


def run_trial(mapping, base_port, duration_s, burst_size, slow_hosts=None):
    unique_backups = set(mapping.values())
    ports, processes = start_backup_servers(unique_backups, base_port, slow_hosts=slow_hosts)
    time.sleep(0.3)

    try:
        results_by_primary = {}
        with ThreadPoolExecutor(max_workers=len(mapping)) as primary_pool:
            futures = {
                primary: primary_pool.submit(throughput_burst, ports[backup], duration_s, burst_size)
                for primary, backup in mapping.items()
            }
            for primary, future in futures.items():
                results_by_primary[primary] = future.result()
    finally:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        time.sleep(0.2)

    return results_by_primary


def record_raw(method, context_seed, trial, mapping, results_by_primary, duration_s):
    load = backup_load(mapping)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(THROUGHPUT_CSV)
    with open(THROUGHPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THROUGHPUT_FIELDS)
        if write_header:
            writer.writeheader()
        for primary, (successes, failures) in results_by_primary.items():
            backup = mapping[primary]
            n_sharing = len(load[backup])
            total = successes + failures
            writer.writerow({
                "method": method,
                "context_seed": context_seed,
                "trial": trial,
                "primary": primary,
                "backup": backup,
                "backup_load": n_sharing,
                "successful_calls": successes,
                "failed_calls": failures,
                "duration_s": duration_s,
                "throughput_cps": successes / duration_s,
                "failure_rate_pct": (100.0 * failures / total) if total else 0.0,
            })


def run_experiment(context_seeds, trials, duration_s, burst_size, methods, order_seed=0,
                    hosts=None, primaries=None, graph=None, overrides=None, slow_hosts=None,
                    rl_model_path="rl_model.zip"):
    hosts = hosts if hosts is not None else config.HOSTS
    primaries = primaries if primaries is not None else config.PRIMARIES

    method_fns = dict(METHODS)
    method_fns["rl"] = lambda ctx, hosts, primaries: _rl_mapping(ctx, hosts, primaries, model_path=rl_model_path)

    order_rng = random_module.Random(order_seed)
    # Keyed by (context_seed, trial) rather than appended to a flat list, so every
    # method's runs can be matched up and compared with a paired test: each method is
    # executed under the same seeds and trials, and pairing removes the between-run
    # variance (background OS load, seed difficulty) that dominates at this sample size.
    all_throughput = {name: {} for name in methods}
    all_failure_rate = {name: {} for name in methods}
    base_port = 9501

    for context_seed in context_seeds:
        ctx = topology_features.build_feature_context(hosts=hosts, seed=context_seed, graph=graph, overrides=overrides)
        method_fns["random"] = lambda ctx, hosts, primaries, cs=context_seed: backup_selector.random_mapping(primaries, hosts, seed=cs)
        # deepcopy per method: heuristic and rl mutate ctx in place (_apply_load_penalty),
        # so a shared ctx would make the scores depend on method order.
        mappings = {name: method_fns[name](copy.deepcopy(ctx), hosts, primaries) for name in methods}

        print("\n--- context_seed={} ---".format(context_seed))

        for trial in range(trials):
            trial_order = list(methods)
            order_rng.shuffle(trial_order)
            for name in trial_order:
                mapping = mappings[name]
                results_by_primary = run_trial(mapping, base_port, duration_s, burst_size, slow_hosts=slow_hosts)
                record_raw(name, context_seed, trial, mapping, results_by_primary, duration_s)

                total_successes = sum(s for s, f in results_by_primary.values())
                total_failures = sum(f for s, f in results_by_primary.values())
                total_calls = total_successes + total_failures
                all_throughput[name][(context_seed, trial)] = total_successes / duration_s
                all_failure_rate[name][(context_seed, trial)] = (
                    (100.0 * total_failures / total_calls) if total_calls else 0.0)

                base_port += len(set(mapping.values())) + 1
            print("  trial {}/{} done (order: {})".format(trial + 1, trials, trial_order))

    n_seeds = len(context_seeds)
    print("\n\n===== SUMMARY across {} context seeds x {} trials ({}s bursts, burst={}) =====".format(
        n_seeds, trials, duration_s, burst_size))
    print("(aggregate throughput/failure-rate per trial, summed/averaged across all primaries)")
    print("(aggregate throughput per run; one run = one context_seed x trial)")
    print("\n{:<12} {:>8} {:>14} {:>16} {:>22}".format(
        "method", "n_runs", "mean_cps", "mean_fail_pct", "95% CI (cps)"))
    for name in methods:
        vals = list(all_throughput[name].values())
        lo, hi = analysis.cluster_bootstrap_ci(vals)
        print("{:<12} {:>8} {:>14.2f} {:>16.4f} {:>22}".format(
            name, len(vals), float(np.mean(vals)),
            float(np.mean(list(all_failure_rate[name].values()))),
            "[{:.2f}, {:.2f}]".format(lo, hi)))

    # Reuse the run-level machinery from analysis.py: paired Wilcoxon signed-rank with
    # Holm-Bonferroni across the family, in place of an unpaired Mann-Whitney per pair
    # with no correction. Throughput was already a per-run quantity, so the sample size
    # does not change here; what changes is that the test now exploits the pairing and
    # that the family-wise error rate is controlled across all comparisons.
    runs = {name: {key: {"mean": value} for key, value in all_throughput[name].items()}
            for name in methods}
    comparisons = analysis.paired_comparisons(runs, metric="mean")
    if comparisons:
        n_pairs = comparisons[0]["n_pairs"]
        print("\n===== Paired Wilcoxon signed-rank on per-run aggregate throughput =====")
        print("Paired by (context_seed, trial); Holm-Bonferroni across the {} comparisons.".format(
            len(comparisons)))
        floor = analysis.min_attainable_p(n_pairs)
        if floor > 0.01:
            print("[power] with n={} pairs the smallest attainable two-sided p is {:.4f};".format(
                n_pairs, floor))
            print("        a non-significant result is weak evidence, not evidence of no effect.")
        fmt = "{:<12} vs {:<12} n={:<3} {:>11} {:>8} {:>11} {:>11}  {}"
        print(fmt.format("method A", "method B", "", "d_median", "r", "p_raw", "p_holm", ""))
        for c in comparisons:
            print(fmt.format(
                c["a"], c["b"], c["n_pairs"],
                "{:.3f}".format(c["median_diff"]), "{:.3f}".format(c["r"]),
                "{:.4f}".format(c["p"]), "{:.4f}".format(c["p_holm"]),
                "significant" if c["p_holm"] < 0.05 else "n.s."))

    return all_throughput, all_failure_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=3.0, help="seconds of sustained load per trial")
    parser.add_argument("--burst", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    parser.add_argument("--n-hosts", type=int, default=None)
    parser.add_argument("--n-primaries", type=int, default=None)
    parser.add_argument("--n-degraded", type=int, default=0)
    parser.add_argument("--scale-seed", type=int, default=0)
    parser.add_argument("--rl-model", default="rl_model.zip")
    args = parser.parse_args()

    # Save this run's output to findings/ as well as the terminal (see run_logging.py).
    import run_logging
    run_logging.capture_to_findings("throughput_{}hosts".format(args.n_hosts or 9))

    hosts, primaries, graph, overrides, slow_hosts = None, None, None, None, None
    if args.n_hosts is not None:
        hosts, primaries, graph = topology_features.generate_scaled_topology(
            n_hosts=args.n_hosts, n_primaries=args.n_primaries, seed=args.scale_seed
        )
        if args.n_degraded > 0:
            backup_pool = [h for h in hosts if h not in primaries]
            degraded = backup_pool[:args.n_degraded]
            overrides = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded}
            slow_hosts = set(degraded)

    run_experiment(
        args.context_seeds, args.trials, args.duration, args.burst, args.methods,
        hosts=hosts, primaries=primaries, graph=graph, overrides=overrides, slow_hosts=slow_hosts,
        rl_model_path=args.rl_model,
    )
