"""Checks for the run-level statistics in analysis.py.

Run: python3 test_analysis.py

The last check is the important one. It builds two methods with *no* true
difference and correlated calls within each run, then shows that the pooled-by-call
test the harness used previously reports an overwhelmingly significant difference
anyway, while the run-level paired test correctly finds nothing. That is the
pseudo-replication failure the analysis rewrite exists to remove.
"""

import numpy as np
from scipy import stats

import analysis


def check(name, condition, detail=""):
    assert condition, "FAILED: {} {}".format(name, detail)
    print("ok   {}{}".format(name, (" -- " + detail) if detail else ""))


def test_holm_bonferroni():
    p = [0.01, 0.04, 0.03, 0.005]
    got = analysis.holm_bonferroni(p)
    expected = {0.005: 0.02, 0.01: 0.03, 0.03: 0.06, 0.04: 0.06}
    check("holm matches worked example",
          all(abs(got[i] - expected[p[i]]) < 1e-12 for i in range(len(p))), str(got))
    order = sorted(range(len(p)), key=lambda i: p[i])
    check("holm is monotone in sorted order",
          [got[i] for i in order] == sorted(got[i] for i in order))
    check("holm never reduces a p-value", all(got[i] >= p[i] for i in range(len(p))))


def test_rank_biserial():
    check("r=+1 when every pair agrees", analysis.matched_pairs_rank_biserial([1, 2, 3]) == 1.0)
    check("r=-1 when every pair disagrees", analysis.matched_pairs_rank_biserial([-1, -2, -3]) == -1.0)
    check("r=0 when pairs cancel",
          abs(analysis.matched_pairs_rank_biserial([1, -1, 2, -2])) < 1e-12)
    check("zero differences are dropped, not counted",
          analysis.matched_pairs_rank_biserial([0, 0, 5]) == 1.0)


def test_cluster_bootstrap():
    rng = np.random.default_rng(0)
    wide = analysis.cluster_bootstrap_ci(list(rng.normal(100, 20, 5)))
    narrow = analysis.cluster_bootstrap_ci(list(rng.normal(100, 20, 200)))
    check("CI narrows as the number of runs grows",
          (narrow[1] - narrow[0]) < (wide[1] - wide[0]),
          "width {:.1f} -> {:.1f}".format(wide[1] - wide[0], narrow[1] - narrow[0]))


def test_failures_are_censored():
    rows = [
        {"method": "m", "context_seed": "1", "trial": "0", "primary": "p", "backup": "b",
         "backup_load": "1", "service_ms": "10", "retry_wait_ms": "0", "wall_ms": "10",
         "attempts": "1", "succeeded": "True"},
        {"method": "m", "context_seed": "1", "trial": "0", "primary": "p", "backup": "b",
         "backup_load": "1", "service_ms": "", "retry_wait_ms": "9000", "wall_ms": "9000",
         "attempts": "8", "succeeded": "False"},
    ]
    rec = analysis.summarize_runs(rows)["m"][(1, 0)]
    check("a failed call does not enter the latency sample", rec["mean"] == 10.0, str(rec["mean"]))
    check("a failed call is counted in the failure rate",
          abs(rec["failure_rate"] - 50.0) < 1e-9, "{:.1f}%".format(rec["failure_rate"]))


def test_pseudo_replication():
    """Two methods, no true difference, 1000 correlated calls per run."""
    rng = np.random.default_rng(1)
    calls_a, calls_b, runs_a, runs_b = [], [], {}, {}
    for seed in range(5):
        for trial in range(3):
            # A per-run offset shared by every call in that run: background load,
            # how hard this seed's topology is, which core the server landed on.
            offset_a, offset_b = rng.normal(0, 10), rng.normal(0, 10)
            a = list(rng.normal(50 + offset_a, 1, 1000))
            b = list(rng.normal(50 + offset_b, 1, 1000))
            calls_a += a
            calls_b += b
            runs_a[(seed, trial)] = {"mean": float(np.mean(a))}
            runs_b[(seed, trial)] = {"mean": float(np.mean(b))}

    _, p_pooled = stats.mannwhitneyu(calls_a, calls_b, alternative="two-sided")
    paired = analysis.paired_comparisons({"a": runs_a, "b": runs_b}, metric="mean")[0]
    print("     pooled-by-call Mann-Whitney, n={:6d} calls: p={:.2e}".format(len(calls_a), p_pooled))
    print("     run-level paired Wilcoxon,   n={:6d} runs : p={:.4f}".format(paired["n_pairs"], paired["p"]))
    check("pooled-by-call test invents significance where there is none", p_pooled < 1e-10)
    check("run-level test correctly finds nothing", paired["p"] > 0.05)


if __name__ == "__main__":
    for fn in [test_holm_bonferroni, test_rank_biserial, test_cluster_bootstrap,
               test_failures_are_censored, test_pseudo_replication]:
        print("\n== {} ==".format(fn.__name__))
        fn()
    print("\nall checks passed")
