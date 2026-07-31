"""Cross-algorithm structural comparison of the PPO and DQN backup-selection policies.

Addresses whether the load-balancing gap observed for PPO at large host counts is specific
to PPO or a property of the problem formulation. DQN is a deliberately different design
point (off-policy replay instead of on-policy rollouts, value-based instead of actor-critic,
epsilon-greedy decay instead of an entropy bonus) trained and evaluated through the same
BackupSelectionEnv, reward, and inference path as PPO, so only the learner differs.

Both policies are scored *structurally*: degraded-host assignments, distinct backups used,
and maximum per-backup load. These are properties of the produced mapping rather than of
RPC timing, so unlike per-call latency they are hardware-independent and reproduce exactly
across machines. They are also the metrics the load-balancing claim is actually about, and
rpc_contention_experiment.py's own backup_load breakdown establishes that backup load is
what drives latency under contention.

Usage:
    python rl_algorithm_comparison.py            # 5 context seeds
    python rl_algorithm_comparison.py --seeds 5
"""

import argparse
import os
from collections import Counter

import rl_backup_selector
import topology_features

# (label, n_hosts, n_primaries, n_degraded, ppo_checkpoint, dqn_checkpoint)
SCENARIOS = [
    ("9-host", 9, 5, 0, "rl_model.zip", "rl_model_dqn.zip"),
    ("16-host", 16, 10, 2, "rl_model_scaled.zip", "rl_model_dqn_scaled.zip"),
    ("32-host", 32, 20, 4, "rl_model_scaled32.zip", "rl_model_dqn_scaled32_mitigated.zip"),
    ("128-host", 128, 80, 16, "rl_model_scaled128_v3.zip", "rl_model_dqn_scaled128.zip"),
]


def evaluate(model, hosts, primaries, graph, overrides, degraded, n_seeds):
    """Returns (distinct_range, peak_load_range, degraded_hits, total_assignments)."""
    distinct, peak, deg_hits = [], [], 0
    for seed in range(n_seeds):
        ctx = topology_features.build_feature_context(
            hosts=hosts, seed=seed, graph=graph, overrides=overrides)
        mapping = rl_backup_selector.predict_mapping(model, hosts, primaries, ctx)
        loads = Counter(mapping.values())
        distinct.append(len(loads))
        peak.append(max(loads.values()))
        deg_hits += sum(1 for b in mapping.values() if b in degraded)
    return (min(distinct), max(distinct)), (min(peak), max(peak)), deg_hits, n_seeds * len(primaries)


def run(n_seeds=5):
    import run_logging
    run_logging.capture_to_findings("rl_algorithm_comparison")

    print("PPO vs DQN, structural comparison over {} context seeds per scenario.\n".format(n_seeds))
    print("Metrics are properties of the mapping (not RPC timing), so they are")
    print("hardware-independent and reproduce across machines.\n")
    print("{:<10} {:<5} {:>12} {:>12} {:>14}".format(
        "scale", "algo", "distinct", "peak load", "degraded"))

    for label, n_hosts, n_primaries, n_degraded, ppo_path, dqn_path in SCENARIOS:
        if n_hosts == 9:
            import config
            hosts, primaries, graph = config.HOSTS, config.PRIMARIES, None
            degraded, overrides = set(), None
        else:
            hosts, primaries, graph = topology_features.generate_scaled_topology(
                n_hosts=n_hosts, n_primaries=n_primaries, seed=0)
            pool = [h for h in hosts if h not in primaries]
            degraded = set(pool[:n_degraded])
            overrides = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded}

        for algo_label, path, algo in [("PPO", ppo_path, "ppo"), ("DQN", dqn_path, "dqn")]:
            if not os.path.exists(path):
                print("{:<10} {:<5} {:>12}".format(label, algo_label, "(no checkpoint)"))
                continue
            model = rl_backup_selector.load_rl_model(path, algo)
            (d_lo, d_hi), (p_lo, p_hi), hits, total = evaluate(
                model, hosts, primaries, graph, overrides, degraded, n_seeds)
            print("{:<10} {:<5} {:>12} {:>12} {:>14}".format(
                label, algo_label,
                "{}-{} of {}".format(d_lo, d_hi, n_primaries),
                "{}-{}".format(p_lo, p_hi),
                "{}/{}".format(hits, total)))
        print()

    print("[output captured to findings/rl_algorithm_comparison.txt]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    run(n_seeds=args.seeds)
