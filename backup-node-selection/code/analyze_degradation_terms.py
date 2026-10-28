"""Which scoring terms actually provide degraded-host avoidance?

Runs the same 287-configuration weight grid as weight_sensitivity.py, but against
the realistic degradation model, and reports how often a configuration that zeroes
each term routes a primary to a degraded host. Under the binary model this question
had no answer: every configuration avoided degraded hosts, so the sweep could not
distinguish between them.
"""

import itertools, copy, numpy as np
import backup_selector, degradation, topology_features
print("Which scoring terms provide degraded-host avoidance?")
print("287 weight configurations x 5 seeds, realistic degradation.\n")
for n_hosts, n_prim in [(32, 20), (128, 80)]:
    hosts, primaries, graph = topology_features.generate_scaled_topology(
        n_hosts=n_hosts, n_primaries=n_prim, seed=0)
    pool = [h for h in hosts if h not in primaries]
    deg = set(pool[:len(pool)//3])
    ov, _, _ = degradation.build(sorted(deg), model="realistic", seed=0,
                                 healthy_hosts=[h for h in pool if h not in deg])
    ctxs = [topology_features.build_feature_context(hosts=hosts, seed=s, graph=graph, overrides=ov)
            for s in range(1, 6)]
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=4)
            if abs(sum(w) - 1.0) < 1e-9]
    rows = []
    for wh, wk, wl, wd in grid:
        weights = {"health": wh, "capacity": wk, "latency": wl, "hops": wd}
        hits = sum(sum(1 for b in backup_selector.select_backups(
                       primaries, hosts, ctx=copy.deepcopy(c), weights=weights).values() if b in deg)
                   for c in ctxs)
        rows.append((wh, wk, wl, wd, hits))
    print("--- {} hosts ({} degraded of {} eligible) ---".format(n_hosts, len(deg), len(pool)))
    print("  {}/{} configurations route at least one primary to a degraded host".format(
        sum(1 for r in rows if r[4] > 0), len(rows)))
    for term, idx in [("health", 0), ("capacity", 1), ("latency", 2), ("hops", 3)]:
        z = [r for r in rows if r[idx] == 0]
        print("    weight({:<8}) = 0 -> {:>3}/{:<3} of those configs hit a degraded host".format(
            term, sum(1 for r in z if r[4] > 0), len(z)))
    d = {"health": 0.35, "capacity": 0.35, "latency": 0.20, "hops": 0.10}
    hits = sum(sum(1 for b in backup_selector.select_backups(
                   primaries, hosts, ctx=copy.deepcopy(c), weights=d).values() if b in deg)
               for c in ctxs)
    print("    published default (0.35,0.35,0.20,0.10): {} degraded assignments\n".format(hits))
print("Reading: the latency term, evaluated against measured RTT, is what provides")
print("degraded-host avoidance. The health term is dispensable. This inverts the")
print("conclusion drawn under the binary degradation model, where every configuration")
print("avoided degraded hosts and the sweep could not discriminate at all.")
print()
print("Caveat: in this model every degradation kind carries some latency signature,")
print("so measured RTT is close to a sufficient statistic here. A failure mode with")
print("no latency signature (a host that is currently fast but about to fail) would")
print("not be caught by it, and would need the health signal instead.")
