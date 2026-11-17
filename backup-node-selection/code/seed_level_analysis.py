"""Seed-level re-analysis of the bare-metal runs.

A run is one (method, context seed, trial) execution, and each configuration has
5 context seeds x 3 trials = 15 runs. The three trials of a seed share that
seed's resource realization and therefore its mapping: they repeat the RPC
timing, not the problem instance. Treating the 15 runs as independent repeats,
one level up, the error the per-call analysis makes. This script averages each
method's per-run statistic over the trials of a seed and analyses the 5
seed-level values that result.

With five paired values an exact two-sided Wilcoxon signed-rank test cannot fall
below p = 0.0625, so no rank test at this n can reject at 0.05. We therefore
report the mean paired difference, its 95% confidence interval (paired t, 4
d.f.), and the number of seeds agreeing in sign. For the heuristic against the
exact assignment the interval is the point: it bounds how large a difference the
data leave room for, which a failure to reject does not.

Reads the raw per-call records in results_rpc/ (not committed; produced by
rpc_contention_experiment.py --results-csv). Writes findings/seed_level_analysis.txt
and paper/tab_seedlevel.tex.

Usage:
    python seed_level_analysis.py
"""

import collections
import csv
import os

import numpy as np
from scipy import stats

import run_logging

RESULTS = "results_rpc"
RUNS = [
    ("16 hosts", "raw_16hosts_cloudlab.csv"),
    ("32 hosts", "raw_32hosts_cloudlab_n15.csv"),
    ("128 hosts", "raw_128hosts_degraded_spread.csv"),
    ("128 hosts, homogeneous", "raw_128hosts_backlog512.csv"),
]
PAIRS = [("round_robin", "optimal"), ("random", "optimal"), ("heuristic", "optimal"),
         ("rl", "optimal"), ("rl", "heuristic"), ("concentrated", "optimal")]
NAME = {"optimal": "exact", "heuristic": "heuristic", "round_robin": "round_robin",
        "random": "random", "rl": "PPO", "concentrated": "concentrated"}
TEX = {"optimal": r"\texttt{exact}", "heuristic": r"\texttt{heuristic}",
       "round_robin": r"\texttt{round\_robin}", "random": r"\texttt{random}",
       "rl": r"\texttt{PPO}", "concentrated": r"\texttt{concentrated}"}


def load(path):
    """-> (per-run latency lists, per-(method, seed) mapping)."""
    lat = collections.defaultdict(list)
    mapping = collections.defaultdict(dict)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["succeeded"] != "True":
                continue
            seed = int(row["context_seed"])
            lat[(row["method"], seed, int(row["trial"]))].append(float(row["service_ms"]))
            mapping[(row["method"], seed)][row["primary"]] = row["backup"]
    return lat, mapping


def seed_values(lat, stat):
    """method -> array of one value per seed: the per-run statistic averaged over trials."""
    methods = sorted({k[0] for k in lat})
    seeds = sorted({k[1] for k in lat})
    out = {}
    for m in methods:
        out[m] = np.array([
            np.mean([stat(np.asarray(v)) for (mm, s, _), v in lat.items() if mm == m and s == seed])
            for seed in seeds])
    return out, seeds


def paired(a, b):
    d = a - b
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    agree = int(max((d > 0).sum(), (d < 0).sum()))
    return d.mean(), lo, hi, agree, len(d)


def main():
    run_logging.capture_to_findings("seed_level_analysis")
    print("Seed-level analysis of the bare-metal runs: per-run statistics averaged over")
    print("the 3 trials of each context seed, then compared across the 5 seeds.")
    print("Differences are a - b in ms with a 95% paired-t interval (4 d.f.).\n")

    table = collections.defaultdict(dict)
    for label, name in RUNS:
        lat, mapping = load(os.path.join(RESULTS, name))
        mean, seeds = seed_values(lat, np.mean)
        p99, _ = seed_values(lat, lambda a: np.percentile(a, 99))
        print("=== {} (seeds {}) ===".format(label, seeds))
        print("{:<13} {:>9} {:>9}   {:<14} {:<10}".format(
            "method", "mean", "p99", "distinct", "peak load"))
        for m in ("optimal", "heuristic", "round_robin", "random", "rl", "concentrated"):
            distinct = [len(set(mapping[(m, s)].values())) for s in seeds]
            peak = [max(collections.Counter(mapping[(m, s)].values()).values()) for s in seeds]
            print("{:<13} {:>9.2f} {:>9.1f}   {:<14} {:<10}".format(
                NAME[m], mean[m].mean(), p99[m].mean(),
                "{}-{} of {}".format(min(distinct), max(distinct), len(mapping[(m, seeds[0])])),
                "{}-{}".format(min(peak), max(peak))))
        print()
        for what, tab in (("mean", mean), ("p99", p99)):
            for a, b in PAIRS:
                diff, lo, hi, agree, n = paired(tab[a], tab[b])
                table[(what, a, b)][label] = (diff, lo, hi, agree, n)
                print("  {:<4} {:>12} - {:<9} {:>8.2f}  [{:>7.2f}, {:>7.2f}]  {}/{} seeds agree in sign".format(
                    what, NAME[a], NAME[b], diff, lo, hi, agree, n))
        print()

    cols = [r[0] for r in RUNS[:3]]
    rows = [("mean", "round_robin", "optimal"), ("mean", "random", "optimal"),
            ("mean", "heuristic", "optimal"), ("mean", "rl", "optimal"),
            ("p99", "round_robin", "optimal"), ("p99", "heuristic", "optimal")]
    lines = [
        "% Generated by seed_level_analysis.py -- do not edit by hand.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Paired differences at the level of the context seed ($n = 5$ per scale): "
        r"each method's per-run statistic is averaged over a seed's three trials, and the "
        r"difference is taken within the seed. Mean difference with its 95\% confidence "
        r"interval (paired $t$, 4 d.f.). Milliseconds; positive means the first method is slower.}",
        r"\label{tab:seedlevel}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        "Statistic & Comparison & " + " & ".join(cols) + r" \\", r"\midrule"]
    last = None
    for what, a, b in rows:
        if last and what != last:
            lines.append(r"\midrule")
        cells = ["{:.2f} [{:.2f}, {:.2f}]".format(*table[(what, a, b)][c][:3]).replace("-", "$-$")
                 for c in cols]
        lines.append("{} & {} $-$ {} & {} \\\\".format(
            "" if what == last else what.capitalize() if what == "mean" else what,
            TEX[a], TEX[b], " & ".join(cells)))
        last = what
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = os.path.join("..", "paper", "tab_seedlevel.tex")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote {}".format(path))


if __name__ == "__main__":
    main()
