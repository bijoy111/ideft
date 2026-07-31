"""Structural comparison of decentralized / load-aware placement baselines against the
feature-aware methods, over both star and multi-hub topologies.

Baselines added here (the NFV-context analogs of decentralized, load-aware replica
placement -- cf. CRUSH/Dynamo, consistent-hashing variants):
  * hrw             : plain rendezvous hashing (decentralized, health/capacity-unaware)
  * weighted_hrw    : health/capacity-weighted rendezvous hashing (quality-aware)
  * weighted_chash  : weighted consistent hashing (Dynamo-style ring, quality-aware)
  * rr_random       : round-robin with a random starting offset (fairer than the fixed
                      alphabetical round-robin, whose degraded exposure is a naming artifact)
scored structurally -- degraded-host hit rate, max per-backup load, distinct backups used --
across the real 9-host DEFT topology, the synthetic 16/32/128-host single-hub stars, and
16/32-host multi-hub topologies (topology diversity), 5 context seeds each. No RPC calls.

Usage:
    python baseline_comparison.py            # 5 seeds
    python baseline_comparison.py --seeds 5
"""

import argparse
import copy

import backup_selector
import topology_features
from weight_sensitivity import build_topologies, evaluate

# Each method -> callable(primaries, hosts, ctx, seed) -> mapping.
METHODS = {
    "random":         lambda pr, ho, ctx, sd: backup_selector.random_mapping(pr, ho, seed=sd),
    "round_robin":    lambda pr, ho, ctx, sd: backup_selector.round_robin_mapping(pr, ho),
    "rr_random":      lambda pr, ho, ctx, sd: backup_selector.randomized_round_robin_mapping(pr, ho, seed=sd),
    "hrw":            lambda pr, ho, ctx, sd: backup_selector.hrw_mapping(pr, ho),
    "weighted_hrw":   lambda pr, ho, ctx, sd: backup_selector.weighted_hrw_mapping(pr, ho, ctx=copy.deepcopy(ctx)),
    "weighted_chash": lambda pr, ho, ctx, sd: backup_selector.weighted_consistent_hash_mapping(pr, ho, ctx=copy.deepcopy(ctx)),
    "heuristic":      lambda pr, ho, ctx, sd: backup_selector.select_backups(pr, ho, ctx=copy.deepcopy(ctx)),
    "optimal":        lambda pr, ho, ctx, sd: backup_selector.optimal_mapping(pr, ho, ctx=copy.deepcopy(ctx)),
}
ORDER = ["random", "round_robin", "rr_random", "hrw", "weighted_hrw", "weighted_chash",
         "heuristic", "optimal"]


def build_multihub_topologies():
    """Multi-hub variants (topology diversity) with the same degraded-host construction as
    the star scenarios. Returns list of (name, hosts, primaries, graph, overrides, degraded)."""
    topos = []
    for n_hosts, n_primaries, n_hubs in [(16, 10, 3), (32, 20, 3)]:
        hosts, primaries, graph = topology_features.generate_multihub_topology(
            n_hosts=n_hosts, n_primaries=n_primaries, n_hubs=n_hubs, seed=0)
        pool = [h for h in hosts if h not in primaries]
        degraded = set(pool[:len(pool) // 3])
        overrides = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded}
        topos.append(("{}-host/{}-hub".format(n_hosts, n_hubs),
                      hosts, primaries, graph, overrides, degraded))
    return topos


def run(n_seeds=5):
    import run_logging
    run_logging.capture_to_findings("baseline_comparison")

    topos = build_topologies() + build_multihub_topologies()
    print("Structural baseline comparison: {} methods x {} topologies x {} seeds\n".format(
        len(ORDER), len(topos), n_seeds))
    print("Per method (5-seed avg): degraded-host hits (of |P|), max per-backup load, "
          "distinct backups used.\n")

    for name, hosts, primaries, graph, overrides, degraded in topos:
        n_prim = len(primaries)
        ctxs = [
            topology_features.build_feature_context(
                hosts=hosts, seed=s, graph=graph, overrides=overrides)
            for s in range(n_seeds)
        ]
        print("=== {} ({} primaries, {} degraded backup hosts) ===".format(
            name, n_prim, len(degraded)))
        print("{:<16} {:>14} {:>10} {:>16}".format(
            "method", "degraded hits", "max load", "distinct backups"))
        for m in ORDER:
            dh = ml = di = 0.0
            for sd, ctx in enumerate(ctxs):
                mapping = METHODS[m](primaries, hosts, ctx, sd)
                d, l, i = evaluate(mapping, degraded)
                dh += d; ml += l; di += i
            print("{:<16} {:>14.2f} {:>10.2f} {:>16.2f}".format(
                m, dh / n_seeds, ml / n_seeds, di / n_seeds))
        print()

    print("[output captured to findings/baseline_comparison.txt]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    run(n_seeds=args.seeds)
