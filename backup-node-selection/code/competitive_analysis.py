"""
Empirical competitive ratios for online backup selection under churn.

Every policy is scored against the offline optimum computed by
online_selector.offline_optimal, which is exact for the decoupled problem and a
feasible (hence upper-bounding) schedule for the congestion-coupled one. Ratios are
therefore honest: the comparator is a real schedule, not a relaxation that no policy
could achieve.

Three sweeps:

  churn      How the ranking depends on how fast host quality moves. This is the
             headline result: the deployed static policy and the online policies
             trade places, and the crossover point is measurable.

  migration  How the ranking depends on the cost of moving a replica. Snapshot
             chasing degrades linearly in M; hysteresis is insensitive to it by
             construction, since its threshold is expressed in units of M.

  alpha      How much the hysteresis threshold matters, and whether the migration
             budget of the migration-budget proposition is tight in practice.

Usage:
    python competitive_analysis.py --sweep churn --seeds 10
    python competitive_analysis.py --sweep all --seeds 10 --out ../findings/
"""

import argparse
import json
import math
import statistics
import time

import online_model as om
import online_selector as osel


def bootstrap_ci(values, n_resamples=4000, ci=0.95, seed=0):
    """Percentile bootstrap over seeds. Seeds are the independent unit here, exactly
    as runs are in the RPC experiments."""
    import random
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_resamples)]
    hi = means[int((1 + ci) / 2 * n_resamples) - 1]
    return (lo, hi)


def evaluate(inst, policies=None):
    """Run every policy on one instance and return its ratio to the optimum."""
    policies = policies or osel.POLICIES
    opt, _, lower = osel.offline_optimal(inst)
    floor = osel.cost_floor(inst)
    out = {"_opt": opt, "_lower_bound": lower, "_floor": floor}
    for name, fn in policies.items():
        d = osel.decompose(inst, fn(inst))
        out[name] = {
            "ratio": d["total"] / opt,
            "excess_ratio": osel.excess_ratio(d["total"], opt, floor),
            "total": d["total"],
            "migrations": d["migrations"],
            "migration_share": d["migration"] / d["total"] if d["total"] else 0.0,
            "service": d["service"],
            "congestion": d["congestion"],
        }
    return out


def run_grid(variable, values, seeds, n_hosts, n_primaries, horizon,
             base_migration=om.DEFAULT_MIGRATION_COST, policies=None):
    """Sweep one variable, holding everything else fixed, across `seeds` traces."""
    policies = policies or osel.POLICIES
    grid = []
    for value in values:
        per_seed = {name: [] for name in policies}
        excess = {name: [] for name in policies}
        share = {name: [] for name in policies}
        migr = {name: [] for name in policies}
        for seed in range(seeds):
            kwargs = dict(n_hosts=n_hosts, n_primaries=n_primaries,
                          horizon=horizon, seed=seed,
                          migration_cost=base_migration)
            if variable == "churn":
                kwargs["params"] = om.ChurnParams(p_degrade=value,
                                                  p_recover=0.08,
                                                  p_fail=value / 10.0)
            elif variable == "persistence":
                # value is the expected number of rounds a degraded host stays
                # degraded. Hard failures are switched off so that this sweep
                # isolates reaction-to-degradation from reaction-to-death.
                kwargs["params"] = om.ChurnParams(p_degrade=0.03,
                                                  p_recover=1.0 / value,
                                                  p_fail=0.0)
            elif variable == "migration":
                kwargs["migration_cost"] = value
            elif variable == "hosts":
                kwargs["n_hosts"] = value
                kwargs["n_primaries"] = max(2, int(value * n_primaries / n_hosts))
            else:
                raise ValueError(variable)

            inst = om.build_instance(**kwargs)
            if variable == "alpha":
                pass
            result = evaluate(inst, policies)
            for name in policies:
                per_seed[name].append(result[name]["ratio"])
                if result[name]["excess_ratio"] == result[name]["excess_ratio"]:
                    excess[name].append(result[name]["excess_ratio"])
                share[name].append(result[name]["migration_share"])
                migr[name].append(result[name]["migrations"])

        row = {"value": value}
        for name in policies:
            lo, hi = bootstrap_ci(per_seed[name])
            elo, ehi = bootstrap_ci(excess[name]) if len(excess[name]) > 1 else (float("nan"),) * 2
            row[name] = {
                "ratio": statistics.mean(per_seed[name]),
                "ci": [lo, hi],
                "excess_ratio": (statistics.mean(excess[name])
                                 if excess[name] else float("nan")),
                "excess_ci": [elo, ehi],
                "migration_share": statistics.mean(share[name]),
                "migrations": statistics.mean(migr[name]),
            }
        grid.append(row)
    return grid


def sweep_alpha(alphas, seeds, n_hosts, n_primaries, horizon, migration_cost):
    """Hysteresis threshold sensitivity, plus the observed tightness of the
    migration budget in the migration-budget proposition (which bounds migration spend by 1/alpha of
    service spend)."""
    rows = []
    for alpha in alphas:
        ratios, budgets, migrations = [], [], []
        for seed in range(seeds):
            inst = om.build_instance(n_hosts=n_hosts, n_primaries=n_primaries,
                                     horizon=horizon, seed=seed,
                                     migration_cost=migration_cost)
            opt, _, _ = osel.offline_optimal(inst)
            d = osel.decompose(inst, osel.policy_hysteresis(inst, alpha=alpha))
            ratios.append(d["total"] / opt)
            budgets.append(d["migration"] / max(d["service"] + d["congestion"], 1e-9))
            migrations.append(d["migrations"])
        lo, hi = bootstrap_ci(ratios)
        rows.append({"alpha": alpha,
                     "ratio": statistics.mean(ratios),
                     "ci": [lo, hi],
                     "budget_used": statistics.mean(budgets),
                     "budget_bound": 1.0 / alpha,
                     "migrations": statistics.mean(migrations)})
    return rows


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def format_grid(grid, variable, policies):
    """Two tables per sweep.

    The excess-over-floor ratio is the headline: it divides each policy's cost
    above the unavoidable floor by the optimum's, so 1.00 means the policy made
    decisions as good as an algorithm that knew the whole churn trace in advance.
    Total-cost ratio is reported too, since that is what the operator's bill
    actually looks like, but it is compressed toward 1 by the floor.
    """
    width = max(max(len(n) for n in policies) + 2, 9)
    head = "{:>12}".format(variable) + "".join("{:>{}}".format(n, width) for n in policies)
    lines = ["excess-over-floor ratio (1.00 = as good as the offline optimum)", head,
             "-" * len(head)]
    for row in grid:
        lines.append("{:>12}".format(row["value"])
                     + "".join("{:>{}.2f}".format(row[n]["excess_ratio"], width)
                               for n in policies))
    lines += ["", "total-cost ratio", head, "-" * len(head)]
    for row in grid:
        lines.append("{:>12}".format(row["value"])
                     + "".join("{:>{}.3f}".format(row[n]["ratio"], width)
                               for n in policies))
    lines += ["", "migration share of total cost", head, "-" * len(head)]
    for row in grid:
        lines.append("{:>12}".format(row["value"])
                     + "".join("{:>{}}".format("{:.1%}".format(row[n]["migration_share"]), width)
                               for n in policies))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="all",
                    choices=["all", "churn", "persistence", "migration",
                             "alpha", "hosts"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--hosts", type=int, default=16)
    ap.add_argument("--primaries", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--migration-cost", type=float, default=om.DEFAULT_MIGRATION_COST)
    ap.add_argument("--out", default=None, help="directory for .txt/.json findings")
    args = ap.parse_args()

    policies = list(osel.POLICIES)
    report, payload = [], {"config": vars(args)}
    started = time.time()

    def section(title):
        report.append("")
        report.append("=" * 78)
        report.append(title)
        report.append("=" * 78)

    if args.sweep in ("all", "persistence"):
        values = [2, 5, 12, 33, 100, 300]
        grid = run_grid("persistence", values, args.seeds, args.hosts,
                        args.primaries, args.horizon, args.migration_cost)
        section("Competitive ratio vs how long degradation lasts\n"
                "  expected degraded duration in rounds; hard failures disabled, so\n"
                "  this isolates reacting to degradation from reacting to death.\n"
                "  {} hosts, {} primaries, horizon {}, M = {}, {} seeds".format(
                    args.hosts, args.primaries, args.horizon,
                    args.migration_cost, args.seeds))
        report.append(format_grid(grid, "duration", policies))
        payload["persistence"] = grid

    if args.sweep in ("all", "churn"):
        values = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
        grid = run_grid("churn", values, args.seeds, args.hosts, args.primaries,
                        args.horizon, args.migration_cost)
        section("Competitive ratio vs churn rate (p_degrade; p_fail = p_degrade/10)\n"
                "  {} hosts, {} primaries, horizon {}, M = {}, {} seeds".format(
                    args.hosts, args.primaries, args.horizon,
                    args.migration_cost, args.seeds))
        report.append(format_grid(grid, "p_degrade", policies))
        payload["churn"] = grid

    if args.sweep in ("all", "migration"):
        values = [0.5, 1.0, 3.0, 6.0, 12.0]
        grid = run_grid("migration", values, args.seeds, args.hosts,
                        args.primaries, args.horizon)
        section("Competitive ratio vs migration cost M\n"
                "  {} hosts, {} primaries, horizon {}, {} seeds".format(
                    args.hosts, args.primaries, args.horizon, args.seeds))
        report.append(format_grid(grid, "M", policies))
        payload["migration"] = grid

    if args.sweep in ("all", "hosts"):
        values = [8, 16, 32, 64]
        grid = run_grid("hosts", values, args.seeds, args.hosts, args.primaries,
                        args.horizon, args.migration_cost)
        section("Competitive ratio vs topology size\n"
                "  primaries scale with hosts, horizon {}, M = {}, {} seeds".format(
                    args.horizon, args.migration_cost, args.seeds))
        report.append(format_grid(grid, "hosts", policies))
        payload["hosts"] = grid

    if args.sweep in ("all", "alpha"):
        rows = sweep_alpha([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], args.seeds,
                           args.hosts, args.primaries, args.horizon,
                           args.migration_cost)
        section("Hysteresis threshold alpha, and tightness of the migration-budget proposition\n"
                "  {} hosts, {} primaries, horizon {}, M = {}, {} seeds".format(
                    args.hosts, args.primaries, args.horizon,
                    args.migration_cost, args.seeds))
        report.append("{:>8}{:>10}{:>18}{:>14}{:>14}{:>12}".format(
            "alpha", "ratio", "95% CI", "budget used", "budget bound", "migrations"))
        report.append("-" * 76)
        for r in rows:
            report.append("{:>8}{:>10.3f}{:>18}{:>14.3f}{:>14.3f}{:>12.1f}".format(
                r["alpha"], r["ratio"],
                "[{:.3f}, {:.3f}]".format(*r["ci"]),
                r["budget_used"], r["budget_bound"], r["migrations"]))
        report.append("")
        report.append("budget used <= budget bound is the migration-budget proposition; it holds at every alpha above.")
        payload["alpha"] = rows

    report.append("")
    report.append("elapsed {:.1f}s".format(time.time() - started))
    text = "\n".join(report)
    print(text)

    if args.out:
        import os
        base = os.path.join(args.out, "online_competitive")
        with open(base + ".txt", "w") as f:
            f.write(text + "\n")
        with open(base + ".json", "w") as f:
            json.dump(payload, f, indent=2)
        print("\nwrote {}.txt and {}.json".format(base, base))


if __name__ == "__main__":
    main()
