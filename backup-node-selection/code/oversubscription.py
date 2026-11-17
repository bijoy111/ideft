"""Does the greedy heuristic still match the exact assignment when primaries
outnumber backups?

Everywhere else in this study every host is a backup candidate, so hosts
outnumber primaries, the exact assignment gives each primary a backup of its own
(k = 1) and a perfect spread always exists. Here the pool is restricted to the
hosts that are not primaries (6, 12 and 48 hosts for 10, 20 and 80 primaries),
so k = ceil(|P| / |pool|) = 2 and backups must be shared.

Structural: no RPC calls. For each method we report the total score of its
mapping (the exact assignment's own objective, evaluated on the features before
any load penalty), degraded assignments, peak load and distinct backups.

The exact assignment caps every backup at k primaries. The heuristic has no cap,
only a load penalty. With degraded hosts in the pool the healthy hosts' k slots
can be fewer than the primaries, and the two methods then resolve the shortage
differently; `forced` is the number of primaries the cap leaves no healthy slot
for.

Usage:
    python oversubscription.py              # 30 seeds
"""

import argparse
import copy
from collections import Counter

import numpy as np

import backup_selector
import degradation
import run_logging
import topology_features

SCALES = [(16, 10), (32, 20), (128, 80)]


def total_score(mapping, ctx, pool):
    total = 0.0
    for p, b in mapping.items():
        total += backup_selector.score_candidates(
            p, pool, ctx, backup_selector.DEFAULT_WEIGHTS)[b]
    return total


def run(n_seeds=30):
    run_logging.capture_to_findings("oversubscription")
    print("Oversubscribed pool: backups restricted to hosts that are not primaries.")
    print("{} seeds, default weights, structural. Score gap = (exact - heuristic) / |exact|.\n".format(n_seeds))
    for model in ("none", "binary", "realistic"):
        for n_hosts, n_prim in SCALES:
            hosts, primaries, graph = topology_features.generate_scaled_topology(
                n_hosts=n_hosts, n_primaries=n_prim, seed=0)
            pool = [h for h in hosts if h not in primaries]
            k = -(-n_prim // len(pool))
            n_deg = 0 if model == "none" else len(pool) // 4
            forced = max(0, n_prim - k * (len(pool) - n_deg))
            rows = {"heuristic": [], "exact": []}
            gaps = []
            for seed in range(1, n_seeds + 1):
                # The degraded backups, and under the realistic model their kinds
                # and severities, are redrawn for every seed.
                deg, overrides = set(), None
                if n_deg:
                    deg = set(topology_features.pick_degraded(pool, [], n_deg, "random", seed=seed))
                    overrides, _, _ = degradation.build(
                        sorted(deg), model=model, seed=seed,
                        healthy_hosts=[h for h in hosts if h not in deg])
                ctx = topology_features.build_feature_context(
                    hosts=hosts, seed=seed, graph=graph, overrides=overrides)
                maps = {
                    "heuristic": backup_selector.select_backups(primaries, pool, ctx=copy.deepcopy(ctx)),
                    "exact": backup_selector.optimal_mapping(primaries, pool, ctx=copy.deepcopy(ctx)),
                }
                score = {}
                for name, mapping in maps.items():
                    load = Counter(mapping.values())
                    score[name] = total_score(mapping, ctx, pool)
                    rows[name].append((score[name],
                                       sum(1 for b in mapping.values() if b in deg),
                                       max(load.values()), len(load)))
                gaps.append((score["exact"] - score["heuristic"]) / abs(score["exact"]))
            print("=== degradation: {:<9} {} primaries on {} backups (k={}), {} degraded, "
                  "forced onto degraded by the cap: {} ===".format(
                      model, n_prim, len(pool), k, n_deg, forced))
            print("{:<10} {:>10} {:>14} {:>10} {:>10}".format(
                "method", "score", "degraded", "peak load", "distinct"))
            for name in ("exact", "heuristic"):
                a = np.array(rows[name], dtype=float)
                print("{:<10} {:>10.2f} {:>14.2f} {:>10} {:>10.2f}".format(
                    name, a[:, 0].mean(), a[:, 1].mean(),
                    "{:.1f} ({:.0f}-{:.0f})".format(a[:, 2].mean(), a[:, 2].min(), a[:, 2].max()),
                    a[:, 3].mean()))
            g = np.array(gaps) * 100
            print("score gap: mean {:.2f}%  max {:.2f}%  (heuristic above exact in {}/{} seeds)\n".format(
                g.mean(), g.max(), int((g < 0).sum()), len(g)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    run(n_seeds=ap.parse_args().seeds)
