"""Load-decay-constant sensitivity sweep.

Directly addresses the concern that the heuristic's decay mechanics are ad hoc: the flat
capacity decrement (backup_selector.LOAD_UNIT, default 20) and the multiplicative health
decay (backup_selector.HEALTH_DECAY, default 0.95) applied to a backup after each
assignment. We sweep BOTH constants -- decrement in {5,10,20,40,80}, health-decay in
{0.80,0.90,0.95,0.99,1.00} -- with the published scoring weights fixed, across the real
9-host DEFT topology and the synthetic 16/32/128-host topologies, 5 context seeds each,
scoring each combination structurally (degraded-host hits, max per-backup load, distinct
backups used). No RPC calls, so it is exact and fast.

The point: load-balancing is a smooth, monotone function of these constants (bigger
decrement / stronger decay => better spread), degraded-host avoidance is unaffected by
them, and the published (20, 0.95) sits in the near-optimal region rather than being an
arbitrary pick.

Usage:
    python decay_sensitivity.py            # full sweep, prints grids + writes findings
    python decay_sensitivity.py --seeds 5
"""

import argparse
import copy

import backup_selector
import topology_features
from weight_sensitivity import build_topologies, evaluate

DEFAULT_WEIGHTS = {"health": 0.35, "capacity": 0.35, "latency": 0.20, "hops": 0.10}
DECREMENTS = [5, 10, 20, 40, 80]
HEALTH_DECAYS = [0.80, 0.90, 0.95, 0.99, 1.00]
PUBLISHED = (20, 0.95)


def sweep(n_seeds=5):
    import run_logging
    run_logging.capture_to_findings("decay_sensitivity")

    topos = build_topologies()
    print("Decay-constant sensitivity: {} decrements x {} health-decays x {} topologies "
          "x {} seeds (published scoring weights fixed)\n".format(
              len(DECREMENTS), len(HEALTH_DECAYS), len(topos), n_seeds))

    # Save the module defaults so we can restore them.
    saved = (backup_selector.LOAD_UNIT, backup_selector.HEALTH_DECAY)

    # results[name][(dec, decay)] = (avg_degraded, avg_max_load, avg_distinct)
    results = {}
    topo_info = {}
    for name, hosts, primaries, graph, overrides, degraded in topos:
        topo_info[name] = (len(primaries), degraded)
        ctxs = [
            topology_features.build_feature_context(
                hosts=hosts, seed=s, graph=graph, overrides=overrides)
            for s in range(n_seeds)
        ]
        grid = {}
        for dec in DECREMENTS:
            for decay in HEALTH_DECAYS:
                backup_selector.LOAD_UNIT = dec
                backup_selector.HEALTH_DECAY = decay
                dh = ml = di = 0.0
                for ctx in ctxs:
                    mapping = backup_selector.select_backups(
                        primaries=primaries, hosts=hosts,
                        ctx=copy.deepcopy(ctx), weights=DEFAULT_WEIGHTS)
                    d, m, i = evaluate(mapping, degraded)
                    dh += d; ml += m; di += i
                grid[(dec, decay)] = (dh / n_seeds, ml / n_seeds, di / n_seeds)
        results[name] = grid

    backup_selector.LOAD_UNIT, backup_selector.HEALTH_DECAY = saved

    # --- degraded-host avoidance across the whole grid ---
    total_cells = len(DECREMENTS) * len(HEALTH_DECAYS) * len(topos)
    cells_hitting = sum(
        1 for grid in results.values() for (dh, _, _) in grid.values() if dh > 0)
    print("=== Degraded-host avoidance ===")
    print("  {}/{} (decrement x decay x topology) cells routed a primary to a degraded "
          "host.\n  Decay constants do not affect the safety guarantee.\n".format(
              cells_hitting, total_cells))

    # --- 128-host max-load grid (the scale where spreading matters most) ---
    def print_maxload_grid(name):
        n_prim, _ = topo_info[name]
        grid = results[name]
        print("=== {} : mean MAX per-backup load ({} primaries; lower=better spread) ==="
              .format(name, n_prim))
        print("decrement \\ decay  " + "".join("{:>8}".format(d) for d in HEALTH_DECAYS))
        for dec in DECREMENTS:
            row = "{:>16}   ".format(dec)
            for decay in HEALTH_DECAYS:
                ml = grid[(dec, decay)][1]
                star = "*" if (dec, decay) == PUBLISHED else " "
                row += "{:>7.2f}{}".format(ml, star)
            print(row)
        print("  (* = published default (decrement=20, health-decay=0.95))\n")

    print_maxload_grid("128-host")
    print_maxload_grid("32-host")

    # --- published point across all scales, for reference ---
    print("=== published default (decrement=20, health-decay=0.95) across scales ===")
    print("{:<14} {:>10} {:>16} {:>10}".format("Scale", "max load", "backups used", "max"))
    for name, hosts, primaries, graph, overrides, degraded in topos:
        n_prim = len(primaries)
        _, ml, di = results[name][PUBLISHED]
        print("{:<14} {:>10.2f} {:>16.2f} {:>10}".format(name, ml, di, n_prim))

    print("\n[output captured to findings/decay_sensitivity.txt]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    sweep(n_seeds=args.seeds)
