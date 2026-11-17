"""Which scoring terms provide degraded-host avoidance?

Sweeps the 286 weight configurations of the 0.1-step grid over the 4-simplex and
reports, for each, the share of assignments that go to a degraded host.

Every seed is a new problem instance in full: new host features, a new random set
of degraded hosts and, under the realistic model, new failure kinds and
severities. Earlier versions of this sweep fixed the degraded hosts and their
failure kinds once and varied only the healthy features, which had two
consequences. The result rested on a single draw of the degradation. And a score
that reads no host state at all, which assigns every primary to the same host,
"avoided" degradation simply because that host was never among the degraded ones.
With the layout redrawn per seed such a score meets degraded hosts at the base
rate, as it should, and serves as the reference the other rows are read against.

The measure is a rate (degraded assignments / all assignments, over all seeds),
not "failed in at least one seed", so it does not grow with the number of seeds.

Each configuration is scored with the latency feature derived from topology and
again with it measured (topology latency plus the delay the candidate is adding,
as an RTT probe would see it). Structural: no RPC calls are made.

Usage:
    python analyze_degradation_terms.py             # 30 seeds
    python analyze_degradation_terms.py --seeds 5
"""

import argparse
import copy
import itertools
import multiprocessing

import numpy as np

import backup_selector
import degradation
import run_logging
import topology_features

SCALES = [(16, 10), (32, 20), (128, 80)]
TERMS = ["health", "capacity", "latency", "hops"]
DEFAULT = (0.35, 0.35, 0.20, 0.10)
GRID = [tuple(round(x, 1) for x in w)
        for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=4)
        if abs(sum(w) - 1.0) < 1e-9]

_INSTANCES = None


def instances(n_hosts, n_prim, model, n_seeds):
    """One (ctx, degraded set) per seed: features, layout and degradation all redrawn."""
    hosts, primaries, graph = topology_features.generate_scaled_topology(
        n_hosts=n_hosts, n_primaries=n_prim, seed=0)
    out = []
    for seed in range(1, n_seeds + 1):
        deg, overrides, _, profiles = degradation.scenario(
            hosts, primaries, n_hosts // 8, selection="random", model=model, seed=seed)
        ctx = topology_features.build_feature_context(
            hosts=hosts, seed=seed, graph=graph, overrides=overrides)
        out.append((ctx, set(deg), profiles))
    return hosts, primaries, out


def _rate(job):
    weights, measured = job
    hosts, primaries, inst = _INSTANCES
    backup_selector.USE_MEASURED_LATENCY = measured
    hits = 0
    by_kind = {}
    for ctx, deg, profiles in inst:
        mapping = backup_selector.select_backups(
            primaries, hosts, ctx=copy.deepcopy(ctx), weights=dict(zip(TERMS, weights)))
        for b in mapping.values():
            if b in deg:
                hits += 1
                kind = profiles[b]["kind"]
                by_kind[kind] = by_kind.get(kind, 0) + 1
    return 100.0 * hits / (len(inst) * len(primaries)), by_kind


def main():
    global _INSTANCES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    run_logging.capture_to_findings("degradation_weight_terms")
    print("Share of assignments that go to a degraded host (%), by weight configuration.")
    print("{} configurations x {} seeds; each seed redraws features, the degraded hosts".format(
        len(GRID), args.seeds))
    print("(1 in 8, uniformly at random) and their failure kinds and severities.\n")

    for model in ("binary", "realistic"):
        for n_hosts, n_prim in SCALES:
            _INSTANCES = instances(n_hosts, n_prim, model, args.seeds)
            inst = _INSTANCES[2]
            kinds = [p["kind"] for _, _, profiles in inst for p in profiles.values()]
            print("=== {} degradation, {} hosts, {} primaries, {} degraded per seed ===".format(
                model, n_hosts, n_prim, n_hosts // 8))
            if model == "realistic":
                print("    degraded hosts drawn over all seeds, by kind: {}".format(
                    ", ".join("{} {}".format(k, kinds.count(k)) for k in sorted(set(kinds)))))
            for measured in (False, True):
                jobs = [(w, measured) for w in GRID + [DEFAULT]]
                with multiprocessing.get_context("fork").Pool() as pool:
                    results = pool.map(_rate, jobs, chunksize=4)
                rates = np.array([r for r, _ in results[:-1]])
                print("--- latency feature: {} ---".format(
                    "measured RTT" if measured else "derived from topology"))
                print("    {:<38} {:>7} {:>7} {:>7} {:>12}".format(
                    "configurations", "mean %", "median", "max", "above 1%"))

                def row(label, idx):
                    r = rates[idx]
                    print("    {:<38} {:>7.2f} {:>7.2f} {:>7.2f} {:>8}/{:<3}".format(
                        label, r.mean(), np.median(r), r.max(), int((r > 1.0).sum()), len(r)))
                row("all", np.arange(len(GRID)))
                for t, term in enumerate(TERMS):
                    row("w({}) = 0".format(term), np.array([i for i, w in enumerate(GRID) if w[t] == 0]))
                    row("w({}) > 0".format(term), np.array([i for i, w in enumerate(GRID) if w[t] > 0]))
                # Health and capacity read within one latency stratum at a time. The
                # marginal groups above mix configurations with and without latency
                # weight in different proportions (11 of 66 against 55 of 220), so
                # with measured RTT they partly compare latency weight, not the term.
                for cond, keep in (("w(latency) > 0", lambda w: w[2] > 0),
                                   ("w(latency) = 0", lambda w: w[2] == 0)):
                    for t in (0, 1):
                        for sign, test in (("= 0", lambda x: x == 0), ("> 0", lambda x: x > 0)):
                            idx = np.array([i for i, w in enumerate(GRID)
                                            if keep(w) and test(w[t])])
                            row("{}, w({}) {}".format(cond, TERMS[t], sign), idx)
                for t, term in enumerate(TERMS):
                    only = GRID.index(tuple(1.0 if i == t else 0.0 for i in range(4)))
                    print("    {:<38} {:>7.2f}".format("only {}".format(term), rates[only]))
                rate, by_kind = results[-1]
                print("    {:<38} {:>7.2f}   by kind: {}".format(
                    "default (0.35,0.35,0.20,0.10)", rate,
                    ", ".join("{} {}".format(k, v) for k, v in sorted(by_kind.items())) or "-"))
            backup_selector.USE_MEASURED_LATENCY = True
            print()

    print("Reading. The reference is the share of hosts that are degraded, 12.5%: what a")
    print("score that reads no host state is expected to give. 'only hops' is such a score")
    print("in a star (every candidate is the same distance away), but it sends all of a")
    print("seed's primaries to one host, so its 30-seed estimate is noisy and the analytic")
    print("12.5% is the figure to compare against. Read health and capacity in the")
    print("'w(latency) > 0' and 'w(latency) = 0' strata, not in the marginal groups.")


if __name__ == "__main__":
    main()
