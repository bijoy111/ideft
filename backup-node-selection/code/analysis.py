"""Run-level statistical analysis for the RPC-contention benchmark.

Why this module exists
----------------------
The original analysis pooled every individual RPC call into one sample and ran
Mann-Whitney U with n in the tens of thousands. That is pseudo-replication: calls
inside one burst share a TCP connection, a single-threaded backup server process,
an OS scheduler, and one resource realization, so they are nowhere near
independent. Inflating n that way shrinks p-values without adding information,
which is why the old output reported "significant" almost everywhere and then had
to tell the reader to ignore the p-values.

The independent unit is a *run*: one (method, context_seed, trial) execution.
So the analysis works in two stages:

1. Reduce each run to per-run statistics over its own calls (mean/median/p95/p99
   service latency, failure rate). Within-run dependence is absorbed here.
2. Compare methods across runs, paired by (context_seed, trial), because every
   method is executed under each of those conditions. Pairing cancels the
   between-run variance (background OS load, how hard a given seed's topology is)
   that would otherwise swamp a comparison with only a handful of samples.

Tests are Wilcoxon signed-rank on the paired per-run values, with the
matched-pairs rank-biserial correlation as effect size, Holm-Bonferroni
correction across the pairwise family, and a cluster bootstrap that resamples
whole runs rather than individual calls.

Latency decomposition
---------------------
Each call carries three timings (see rpc_contention_experiment.replicate_burst):

  service_ms     time inside the RPC attempt that actually succeeded. This is
                 the backup's real service + queueing delay, and it is the
                 metric every comparison below uses.
  retry_wait_ms  failed attempts plus the client's own exponential-backoff
                 sleep. This is a property of the client retry policy, not of
                 the backup, and must never be reported as backup latency.
  wall_ms        service_ms + retry_wait_ms: what the previous harness recorded
                 as "latency". Retained only so the old numbers remain
                 reconstructible.

Failed calls are censored, not imputed. The old harness substituted a 10,000 ms
sentinel into the latency sample, which contaminates every percentile computed
from it. Here a failure contributes to the failure rate and to nothing else.

Standalone use:
    python analysis.py results_rpc/rpc_contention_raw.csv
"""

import csv
import gzip
import itertools
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# Per-run statistics computed over the successful calls of a single run.
RUN_METRICS = ["mean", "median", "p95", "p99"]


def _percentile_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def summarize_runs(rows, metric="service_ms"):
    """Reduce raw per-call rows to one record per (method, context_seed, trial).

    Returns {method: {(context_seed, trial): {metric stats..., 'failure_rate', 'n_calls'}}}.
    Latency statistics are computed over successful calls only; failures are
    carried separately as a rate, never imputed as a latency value.
    """
    ok_values = defaultdict(list)
    totals = defaultdict(int)
    failures = defaultdict(int)
    for row in iter_rows(rows):
        key = (row["method"], int(row["context_seed"]), int(row["trial"]))
        totals[key] += 1
        if _is_ok(row) and row[metric] != "":
            ok_values[key].append(float(row[metric]))
        else:
            failures[key] += 1

    out = defaultdict(dict)
    for key, ok in ok_values.items():
        method, seed, trial = key
        n_total = totals[key]
        n_failed = failures[key]
        if not ok:
            continue
        rec = _percentile_stats(ok)
        rec["failure_rate"] = 100.0 * n_failed / n_total if n_total else 0.0
        rec["n_calls"] = n_total
        out[method][(seed, trial)] = rec
    return out


def _is_ok(row):
    value = row.get("succeeded", "True")
    return str(value).strip().lower() in ("1", "true", "yes")


def cluster_bootstrap_ci(run_values, n_resamples=5000, ci=0.95, seed=0):
    """Confidence interval on the mean of per-run values, resampling whole runs.

    Resampling runs (not calls) is what makes the interval honest: it propagates
    between-run variance instead of treating correlated calls as fresh evidence.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(run_values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def matched_pairs_rank_biserial(diffs):
    """Effect size for a Wilcoxon signed-rank test: (T+ - T-) / (T+ + T-) in [-1, 1].

    Unlike the rank-biserial correlation the old code computed over pooled calls,
    this is defined on the paired per-run differences, so it measures how
    consistently one method beats the other across independent runs.
    """
    diffs = np.asarray([d for d in diffs if d != 0], dtype=float)
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    t_pos = ranks[diffs > 0].sum()
    t_neg = ranks[diffs < 0].sum()
    total = t_pos + t_neg
    return float((t_pos - t_neg) / total) if total else 0.0


def holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    The pairwise comparison of m methods is a family of m(m-1)/2 tests; without
    correction the chance of at least one spurious "significant" result across
    that family is far above 5%.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * pvalues[i]
        running = max(running, min(value, 1.0))
        adjusted[i] = running
    return adjusted


def paired_comparisons(runs, metric="mean"):
    """Wilcoxon signed-rank over runs shared by both methods, paired by (seed, trial)."""
    methods = list(runs)
    results = []
    for a, b in itertools.combinations(methods, 2):
        shared = sorted(set(runs[a]) & set(runs[b]))
        if len(shared) < 2:
            continue
        va = np.array([runs[a][k][metric] for k in shared], dtype=float)
        vb = np.array([runs[b][k][metric] for k in shared], dtype=float)
        diffs = va - vb
        if np.allclose(diffs, 0):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p = 1.0
        results.append({
            "a": a, "b": b, "n_pairs": len(shared),
            "median_diff": float(np.median(diffs)),
            "r": matched_pairs_rank_biserial(diffs),
            "p": float(p),
        })
    if results:
        adj = holm_bonferroni([r["p"] for r in results])
        for r, p_adj in zip(results, adj):
            r["p_holm"] = p_adj
    return results


def min_attainable_p(n_pairs):
    """Smallest two-sided p a signed-rank test on n pairs can produce.

    Reported so a null result is not misread as evidence of no effect when the
    design simply cannot resolve one.
    """
    return min(1.0, 2.0 / (2 ** n_pairs))


def load_level_table(rows, metric="service_ms"):
    """Latency vs. backup load, aggregated to one observation per (run, backup).

    The pooled-by-call version of this table had the same pseudo-replication
    problem as the method comparison, and worse: high-load cells are high-load
    precisely because many primaries hammer one server, so their calls are the
    most strongly correlated ones in the dataset.
    """
    by_unit = defaultdict(list)
    for row in iter_rows(rows):
        if not _is_ok(row) or row[metric] == "":
            continue
        key = (int(row["backup_load"]), row["method"], int(row["context_seed"]),
               int(row["trial"]), row["backup"])
        by_unit[key].append(float(row[metric]))

    by_load = defaultdict(list)
    for key, values in by_unit.items():
        by_load[key[0]].append(float(np.mean(values)))
    return by_load


def print_report(rows, metric="service_ms", title="service latency"):
    """`rows` may be a CSV path (streamed, three passes) or an in-memory sequence."""
    runs = summarize_runs(rows, metric=metric)
    if not runs:
        print("no successful calls recorded; nothing to analyse")
        return runs

    n_runs = {m: len(v) for m, v in runs.items()}
    print("\n===== Run-level summary: {} =====".format(title))
    print("One observation per (method, context_seed, trial). Latency statistics are")
    print("computed within each run, then averaged across runs; CIs are cluster")
    print("bootstraps over runs. Failed calls are censored (counted in failure rate,")
    print("excluded from latency), not imputed as a sentinel value.\n")
    header = "{:<12} {:>6} {:>10} {:>10} {:>10} {:>10} {:>9} {:>22}"
    print(header.format("method", "runs", "mean_ms", "median_ms", "p95_ms", "p99_ms",
                        "fail_%", "95% CI of mean"))
    for method in runs:
        per_run = list(runs[method].values())
        means = [r["mean"] for r in per_run]
        lo, hi = cluster_bootstrap_ci(means)
        print(header.format(
            method, n_runs[method],
            "{:.2f}".format(np.mean(means)),
            "{:.2f}".format(np.mean([r["median"] for r in per_run])),
            "{:.2f}".format(np.mean([r["p95"] for r in per_run])),
            "{:.2f}".format(np.mean([r["p99"] for r in per_run])),
            "{:.2f}".format(np.mean([r["failure_rate"] for r in per_run])),
            "[{:.2f}, {:.2f}]".format(lo, hi),
        ))

    for metric_name in ("mean", "p99"):
        comparisons = paired_comparisons(runs, metric=metric_name)
        if not comparisons:
            continue
        n_pairs = comparisons[0]["n_pairs"]
        print("\n===== Paired Wilcoxon signed-rank on per-run {} ({}) =====".format(
            metric_name, title))
        print("Paired by (context_seed, trial); Holm-Bonferroni across the {} comparisons.".format(
            len(comparisons)))
        print("r is the matched-pairs rank-biserial correlation: how consistently one")
        print("method beats the other across runs, +1 or -1 meaning every run agrees.")
        floor = min_attainable_p(n_pairs)
        if floor > 0.01:
            print("[power] with n={} pairs the smallest attainable two-sided p is {:.4f};".format(
                n_pairs, floor))
            print("        a non-significant result here is weak evidence, not evidence of no effect.")
        print()
        fmt = "{:<12} vs {:<12} n={:<3} {:>11} {:>8} {:>11} {:>11}  {}"
        print(fmt.format("method A", "method B", "", "d_median", "r", "p_raw", "p_holm", ""))
        for c in comparisons:
            verdict = "significant" if c["p_holm"] < 0.05 else "n.s."
            print(fmt.format(
                c["a"], c["b"], c["n_pairs"],
                "{:.3f}".format(c["median_diff"]),
                "{:.3f}".format(c["r"]),
                "{:.4f}".format(c["p"]),
                "{:.4f}".format(c["p_holm"]),
                verdict,
            ))

    by_load = load_level_table(rows, metric=metric)
    if by_load:
        print("\n===== {} vs. backup load (one observation per run x backup) =====".format(
            title.capitalize()))
        print("Does sharing a backup among more primaries raise latency, independent of")
        print("which method produced the sharing? Aggregated per (run, backup) so that")
        print("correlated calls against one loaded server count once, not thousands of times.\n")
        print("{:>6} {:>7} {:>10} {:>10} {:>22}".format(
            "load", "n_units", "mean_ms", "median_ms", "95% CI of mean"))
        for load in sorted(by_load):
            values = by_load[load]
            lo, hi = cluster_bootstrap_ci(values)
            print("{:>6} {:>7} {:>10.2f} {:>10.2f} {:>22}".format(
                load, len(values), float(np.mean(values)), float(np.median(values)),
                "[{:.2f}, {:.2f}]".format(lo, hi)))
    return runs


def print_decomposition(rows):
    """Show how much of the old wall-clock 'latency' was client-side retry wait.

    The previous harness started its timer outside the retry loop, so every
    backoff sleep landed inside the reported latency. This table separates the
    two so the inflation is visible rather than silently attributed to the backup.
    """
    by_method = defaultdict(lambda: {"service": [], "retry": [], "wall": [], "retried": 0, "n": 0})
    for row in iter_rows(rows):
        if not _is_ok(row):
            continue
        acc = by_method[row["method"]]
        acc["service"].append(float(row["service_ms"]))
        acc["retry"].append(float(row["retry_wait_ms"]))
        acc["wall"].append(float(row["wall_ms"]))
        acc["n"] += 1
        if int(row["attempts"]) > 1:
            acc["retried"] += 1

    print("\n===== Latency decomposition: backup service time vs. client retry wait =====")
    print("wall_ms is what the previous harness reported as 'latency'. Any gap between")
    print("wall and service is the client's own backoff sleep, not backup delay.\n")
    header = "{:<12} {:>12} {:>12} {:>12} {:>12} {:>10}"
    print(header.format("method", "service p99", "retry p99", "wall p99", "wall-service", "retried %"))
    for method, acc in by_method.items():
        if not acc["service"]:
            continue
        s99 = float(np.percentile(acc["service"], 99))
        r99 = float(np.percentile(acc["retry"], 99))
        w99 = float(np.percentile(acc["wall"], 99))
        print(header.format(
            method,
            "{:.2f}".format(s99), "{:.2f}".format(r99), "{:.2f}".format(w99),
            "{:.2f}".format(w99 - s99),
            "{:.2f}".format(100.0 * acc["retried"] / acc["n"]),
        ))


def iter_rows(source):
    """Yield call records from a CSV path or re-yield an in-memory sequence.

    The 128-host scenario produces on the order of a million call records, so every
    consumer below streams the file and keeps only the floats it needs, rather than
    materialising a list of row dicts.
    """
    if isinstance(source, str):
        opener = gzip.open if source.endswith(".gz") else open
        with opener(source, "rt", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        for row in source:
            yield row


def load_rows(path):
    """Materialise all rows. Prefer iter_rows for large files."""
    return list(iter_rows(path))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_rpc/rpc_contention_raw.csv"
    print("analysing call records from {}".format(path))
    print_decomposition(path)
    print_report(path, metric="service_ms", title="service latency")
