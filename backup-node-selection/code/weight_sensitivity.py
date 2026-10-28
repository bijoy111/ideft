"""Heuristic scoring-weight sensitivity sweep.

Addresses the concern that the multi-criteria heuristic's default scoring weights
(health=0.35, capacity=0.35, latency=0.20, hops=0.10) are chosen by intuition. We
grid-search the (health, capacity, latency, hops) weight simplex -- 286 points on a
0.1-step grid (all non-negative integer 4-tuples summing to 10, i.e. C(13,3)=286),
plus the exact published default (which is not on that grid), for 287 configurations
total -- and evaluate each STRUCTURALLY (no RPC calls): degraded-host hit rate, maximum
per-backup load, and number of distinct backups used. Evaluated across the real 9-host
DEFT topology and the synthetic 16/32/128-host topologies, 5 context seeds each.

This reuses backup_selector.select_backups() and topology_features directly, so the sweep
exercises the exact production heuristic, and requires no RPC harness, so it is exact and
reproducible (deterministic per seed).

Usage:
    python weight_sensitivity.py            # full sweep, prints summary + writes findings
    python weight_sensitivity.py --seeds 5  # number of context seeds (default 5)
"""

import argparse
from collections import defaultdict

import config
import topology_features
import backup_selector
import degradation
from topology_features import INFEASIBLE_HOSTS

DEFAULT = (0.35, 0.35, 0.20, 0.10)
WEIGHT_KEYS = ("health", "capacity", "latency", "hops")


def simplex_grid(step_denominator=10):
    """All non-negative integer 4-tuples summing to `step_denominator`, as fractions.
    For denominator 10 this is C(13,3)=286 points on the 0.1-step 4-simplex."""
    n = step_denominator
    grid = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            for c in range(n - a - b + 1):
                d = n - a - b - c
                grid.append((a / n, b / n, c / n, d / n))
    return grid


def all_configs():
    grid = simplex_grid(10)
    # The published default is off-grid (0.35/0.35/0.20/0.10); add it explicitly.
    return grid + [DEFAULT]


def build_topologies(degradation_model="binary"):
    """Returns list of (name, hosts, primaries, graph, overrides, degraded_set).

    `degradation_model` selects how degraded hosts are constructed; see
    degradation.py. Under "binary" every degraded host sits at one extreme point on
    both scoring features at once, which is why the original sweep found that every
    weight configuration avoids them. "realistic" draws a continuous severity and one
    of several failure kinds, including kinds that are invisible to resource gauges.
    """
    topos = []

    # Real 9-host DEFT topology, homogeneous (no engineered degradation).
    topos.append((
        "9-host (real)", config.HOSTS, config.PRIMARIES,
        topology_features.build_topology_graph(), None, set(),
    ))

    # Synthetic scaled star topologies with floor(|H\P|/3) degraded backup-eligible hosts,
    # matching rpc_contention_experiment.py's degraded construction exactly.
    for n_hosts, n_primaries in [(16, 10), (32, 20), (128, 80)]:
        hosts, primaries, graph = topology_features.generate_scaled_topology(
            n_hosts=n_hosts, n_primaries=n_primaries, seed=0)
        backup_pool = [h for h in hosts if h not in primaries]
        n_degraded = len(backup_pool) // 3
        degraded = set(backup_pool[:n_degraded])
        overrides, _, _ = degradation.build(
            sorted(degraded), model=degradation_model, seed=0,
            healthy_hosts=[h for h in backup_pool if h not in degraded])
        topos.append((
            "{}-host".format(n_hosts), hosts, primaries, graph, overrides, degraded,
        ))
    return topos


def evaluate(mapping, degraded):
    """Structural metrics for one mapping."""
    load = defaultdict(int)
    for b in mapping.values():
        load[b] += 1
    degraded_hits = sum(1 for b in mapping.values() if b in degraded)
    max_load = max(load.values()) if load else 0
    distinct = len(set(mapping.values()))
    return degraded_hits, max_load, distinct


def run(n_seeds=5):
    import run_logging
    run_logging.capture_to_findings("weight_sensitivity")

    configs = all_configs()
    topos = build_topologies(args.degradation_model)

    print("Weight-sensitivity sweep: {} configurations x {} topologies x {} seeds\n".format(
        len(configs), len(topos), n_seeds))

    summary_rows = []      # per-topology Table-X row
    claim_lines = []       # supporting-claim lines

    for name, hosts, primaries, graph, overrides, degraded in topos:
        n_prim = len(primaries)

        # Pre-build one context per seed (resources depend only on seed + overrides).
        ctxs = [
            topology_features.build_feature_context(
                hosts=hosts, seed=s, graph=graph, overrides=overrides)
            for s in range(n_seeds)
        ]

        # Per-config aggregates (averaged over seeds).
        per_config = []  # (weights, avg_degraded_hits, avg_max_load, avg_distinct)
        default_stats = None
        for w in configs:
            weights = dict(zip(WEIGHT_KEYS, w))
            dh_sum = ml_sum = di_sum = 0.0
            for ctx in ctxs:
                import copy
                mapping = backup_selector.select_backups(
                    primaries=primaries, hosts=hosts, ctx=copy.deepcopy(ctx), weights=weights)
                dh, ml, di = evaluate(mapping, degraded)
                dh_sum += dh; ml_sum += ml; di_sum += di
            row = (w, dh_sum / n_seeds, ml_sum / n_seeds, di_sum / n_seeds)
            per_config.append(row)
            if w == DEFAULT:
                default_stats = row

        # Degraded-host avoidance across ALL configs.
        configs_hitting = sum(1 for (_, dh, _, _) in per_config if dh > 0)
        # Load-balancing spread.
        best_max_load = min(ml for (_, _, ml, _) in per_config)
        worst_max_load = max(ml for (_, _, ml, _) in per_config)

        # Rank of default by (degraded_hits, max_load) ascending.
        order = sorted(per_config, key=lambda r: (r[1], r[2]))
        default_rank = 1 + next(i for i, r in enumerate(order) if r[0] == DEFAULT)

        summary_rows.append((name, default_stats[2], best_max_load, default_stats[3], n_prim))
        claim_lines.append(
            "  {:<14} configs-hitting-degraded={}/{}  default_maxload={:.2f}  "
            "best_maxload={:.2f}  worst_maxload={:.2f}  default_backups={:.2f}/{}  "
            "default_rank={}/{}".format(
                name, configs_hitting, len(configs), default_stats[2], best_max_load,
                worst_max_load, default_stats[3], n_prim, default_rank, len(configs)))

    # ---- Table X ----
    header = ("Scale", "Default max load", "Best max load", "Default backups used", "Max possible")
    print("=== TABLE: published default vs. best achievable (max load=1), {}-seed avg ===".format(n_seeds))
    print("{:<14} {:>16} {:>14} {:>21} {:>13}".format(*header))
    table_lines = ["{:<14} {:>16} {:>14} {:>21} {:>13}".format(*header)]
    for name, dml, bml, dbu, mx in summary_rows:
        line = "{:<14} {:>16.2f} {:>14.2f} {:>21.2f} {:>13}".format(name, dml, bml, dbu, mx)
        print(line); table_lines.append(line)

    print("\n=== supporting claims ===")
    for l in claim_lines:
        print(l)

    print("\n[output captured to findings/weight_sensitivity.txt]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--degradation-model", choices=["binary", "realistic"], default="binary",
                    help="see degradation.py; 'binary' reproduces the published sweep")
    args = ap.parse_args()
    run(n_seeds=args.seeds)
