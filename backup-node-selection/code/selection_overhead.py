"""Measures wall-clock time to compute one full backup-selection mapping via
each method, across topology sizes.

This covers the selection-speed side that the latency-under-contention experiments
(rpc_contention_experiment.py) don't: a selection algorithm is only usable online if
it computes a new mapping fast enough relative to the failure/overload-detection
interval it serves. A method that matches the optimal bound on quality but takes 100x
longer to compute is a different deployment story from one that is comparably fast.

Usage:
    python3 selection_overhead.py
"""

import copy
import os
import time

import numpy as np

import backup_selector
import config
import rl_backup_selector
import topology_features

N_REPEATS = 30


def time_call(fn, prep=None, n_repeats=N_REPEATS):
    """Times fn() over n_repeats. `prep`, if given, runs before the clock starts on each
    iteration and its result is passed to fn. This hands heuristic/rl a fresh context copy
    per call (they mutate it in place) without counting the copy time as selection overhead."""
    prep = prep or (lambda: None)
    fn(prep())  # warmup: keep first-call/cache effects out of the timed samples
    times = []
    for _ in range(n_repeats):
        arg = prep()
        start = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(times)
    return arr.mean(), np.median(arr), arr.std()


def measure_scenario(label, hosts, primaries, graph, overrides, rl_model_path, seed=42,
                     rl_dqn_model_path=None):
    ctx = topology_features.build_feature_context(hosts=hosts, seed=seed, graph=graph, overrides=overrides)
    model = rl_backup_selector.load_rl_model(rl_model_path, "ppo")

    # heuristic and rl mutate the context in place (_apply_load_penalty), so each timed
    # call gets a fresh copy via `prep`, produced outside the clock. Methods that ignore
    # the context (concentrated/random/round_robin) get prep=None.
    fresh = lambda: copy.deepcopy(ctx)
    calls = {
        "concentrated": (None, lambda c: backup_selector.concentrated_mapping(primaries, hosts)),
        "random": (None, lambda c: backup_selector.random_mapping(primaries, hosts, seed=seed)),
        "round_robin": (None, lambda c: backup_selector.round_robin_mapping(primaries, hosts)),
        "heuristic": (fresh, lambda c: backup_selector.select_backups(primaries, hosts, ctx=c)),
        "optimal": (fresh, lambda c: backup_selector.optimal_mapping(primaries, hosts, ctx=c)),
        "rl": (fresh, lambda c: rl_backup_selector.predict_mapping(model, hosts, primaries, c)),
    }
    # Timing PPO and DQN back-to-back in the same process keeps their comparison internally
    # consistent, since absolute wall-clock timings are machine- and load-sensitive.
    if rl_dqn_model_path is not None and os.path.exists(rl_dqn_model_path):
        dqn_model = rl_backup_selector.load_rl_model(rl_dqn_model_path, "dqn")
        calls["rl_dqn"] = (fresh, lambda c: rl_backup_selector.predict_mapping(dqn_model, hosts, primaries, c))

    print("\n--- {} ({} hosts, {} primaries) ---".format(label, len(hosts), len(primaries)))
    print("{:<12} {:>10} {:>10} {:>10}".format("method", "mean_ms", "median_ms", "std_ms"))
    results = {}
    for name, (prep, fn) in calls.items():
        mean, median, std = time_call(fn, prep=prep)
        results[name] = (mean, median, std)
        print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f}".format(name, mean, median, std))
    return results


def main():
    # Save this run's output to findings/ as well as the terminal (see run_logging.py).
    import run_logging
    run_logging.capture_to_findings("selection_overhead")

    measure_scenario("homogeneous-9", config.HOSTS, config.PRIMARIES, None, None, "rl_model.zip",
                     rl_dqn_model_path="rl_model_dqn.zip")

    hosts16, primaries16, graph16 = topology_features.generate_scaled_topology(n_hosts=16, n_primaries=10, seed=0)
    backup_pool16 = [h for h in hosts16 if h not in primaries16]
    degraded16 = backup_pool16[:2]
    overrides16 = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded16}
    measure_scenario("scaled-16", hosts16, primaries16, graph16, overrides16, "rl_model_scaled.zip",
                     rl_dqn_model_path="rl_model_dqn_scaled.zip")

    hosts32, primaries32, graph32 = topology_features.generate_scaled_topology(n_hosts=32, n_primaries=20, seed=0)
    backup_pool32 = [h for h in hosts32 if h not in primaries32]
    degraded32 = backup_pool32[:4]
    overrides32 = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded32}
    measure_scenario("scaled-32", hosts32, primaries32, graph32, overrides32, "rl_model_scaled32.zip",
                     rl_dqn_model_path="rl_model_dqn_scaled32_mitigated.zip")

    hosts128, primaries128, graph128 = topology_features.generate_scaled_topology(n_hosts=128, n_primaries=80, seed=0)
    backup_pool128 = [h for h in hosts128 if h not in primaries128]
    degraded128 = backup_pool128[:16]
    overrides128 = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded128}
    measure_scenario("scaled-128", hosts128, primaries128, graph128, overrides128, "rl_model_scaled128_v3.zip",
                     rl_dqn_model_path="rl_model_dqn_scaled128.zip")


if __name__ == "__main__":
    main()
