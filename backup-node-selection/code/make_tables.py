"""Emit the paper's result tables as LaTeX directly from the archived call records.

Every number in the paper's results section comes from here, so no value is
transcribed by hand. Run:  python3 make_tables.py > ../paper/tables_generated.tex
"""

import numpy as np

import analysis

ARCHIVE = "../findings/raw/{}hosts.csv.gz"
# Order methods as the paper discusses them.
ORDER = ["baseline", "concentrated", "random", "round_robin", "heuristic", "optimal", "rl"]
PRETTY = {"rl": "PPO"}


def ordered(runs):
    return [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]


def name(m):
    return "\\texttt{{{}}}".format(PRETTY.get(m, m).replace("_", "\\_"))


def summary_table(scale, caption, label):
    runs = analysis.summarize_runs(ARCHIVE.format(scale), metric="service_ms")
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Method & Mean (ms) & 95\\% CI of mean & Median (ms) & p95 (ms) \\\\",
        "\\midrule",
    ]
    for m in ordered(runs):
        per_run = list(runs[m].values())
        means = [r["mean"] for r in per_run]
        lo, hi = analysis.cluster_bootstrap_ci(means)
        lines.append("{} & {:.2f} & [{:.2f}, {:.2f}] & {:.2f} & {:.2f} \\\\".format(
            name(m), np.mean(means), lo, hi,
            np.mean([r["median"] for r in per_run]),
            np.mean([r["p95"] for r in per_run])))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def paired_table(scale, caption, label, against="heuristic"):
    runs = analysis.summarize_runs(ARCHIVE.format(scale), metric="service_ms")
    comps = analysis.paired_comparisons(runs, metric="mean")
    rows = []
    for c in comps:
        if against not in (c["a"], c["b"]):
            continue
        # Orient every row as "other method minus the reference".
        other, d, r = (c["a"], c["median_diff"], c["r"]) if c["b"] == against \
            else (c["b"], -c["median_diff"], -c["r"])
        rows.append((other, d, r, c["p_holm"]))
    rows.sort(key=lambda t: ORDER.index(t[0]) if t[0] in ORDER else 99)
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrl}", "\\toprule",
        "Method vs.\\ \\texttt{heuristic} & $\\Delta$ median (ms) & $r$ & $p_{\\mathrm{Holm}}$ & Verdict \\\\",
        "\\midrule",
    ]
    for other, d, r, p in rows:
        lines.append("{} & {:+.2f} & {:+.3f} & {:.4f} & {} \\\\".format(
            name(other), d, r, p, "significant" if p < 0.05 else "n.s."))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def load_table(scales, caption, label):
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{clrrr}", "\\toprule",
        "Scenario & Load & $n$ units & Mean (ms) & 95\\% CI \\\\",
        "\\midrule",
    ]
    for i, (scale, tag) in enumerate(scales):
        by_load = analysis.load_level_table(ARCHIVE.format(scale), metric="service_ms")
        keys = sorted(by_load)
        if i:
            lines.append("\\midrule")
        for j, load in enumerate(keys):
            vals = by_load[load]
            lo, hi = analysis.cluster_bootstrap_ci(vals)
            head = "\\multirow{{{}}}{{*}}{{{}}}".format(len(keys), tag) if j == 0 else ""
            lines.append("{} & {} & {} & {:.2f} & [{:.2f}, {:.2f}] \\\\".format(
                head, load, len(vals), np.mean(vals), lo, hi))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def decomposition_table(scale, caption, label):
    """Service time vs. client retry wait: only meaningful at 128 hosts."""
    from collections import defaultdict
    acc = defaultdict(lambda: {"s": [], "w": [], "n": 0, "ret": 0})
    for row in analysis.iter_rows(ARCHIVE.format(scale)):
        if not analysis._is_ok(row):
            continue
        a = acc[row["method"]]
        a["s"].append(float(row["service_ms"]))
        a["w"].append(float(row["wall_ms"]))
        a["n"] += 1
        a["ret"] += int(row["attempts"]) > 1
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Method & Service p99 & Wall p99 & Retry wait & Calls retried (\\%) \\\\",
        "\\midrule",
    ]
    for m in ordered(acc):
        a = acc[m]
        s99, w99 = np.percentile(a["s"], 99), np.percentile(a["w"], 99)
        lines.append("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            name(m), s99, w99, w99 - s99, 100.0 * a["ret"] / a["n"]))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print("% Generated by make_tables.py from findings/raw/*.csv.gz -- do not edit by hand.\n")
    print(summary_table(9, "Homogeneous scale (9 hosts, 5 primaries), 15 runs "
                           "(5 context seeds $\\times$ 3 trials), 200 calls/primary, burst 10. "
                           "Per-run service latency, averaged across runs.", "tab:homogeneous"))
    print(paired_table(9, "Homogeneous scale: paired comparisons against the heuristic.",
                       "tab:homogeneous-paired"))
    print(summary_table(16, "Scaled heterogeneous, 16 hosts (10 primaries, 2 degraded), 15 runs.",
                        "tab:scaled16"))
    print(paired_table(16, "16 hosts: paired comparisons against the heuristic.", "tab:scaled16-paired"))
    print(summary_table(32, "Scaled heterogeneous, 32 hosts (20 primaries, 4 degraded), 15 runs.",
                        "tab:scaled32"))
    print(paired_table(32, "32 hosts: paired comparisons against the heuristic.", "tab:scaled32-paired"))
    print(summary_table(128, "Scaled heterogeneous, 128 hosts (80 primaries, 16 degraded), 15 runs.",
                        "tab:scaled128"))
    print(paired_table(128, "128 hosts: paired comparisons against the heuristic.", "tab:scaled128-paired"))
    print(load_table([(9, "Homog."), (16, "16-host")],
                     "Service latency vs.\\ backup load, one observation per (run $\\times$ backup).",
                     "tab:backupload"))
    print(decomposition_table(128,
          "128 hosts: backup service time vs.\\ client retry wait. Only at this scale does "
          "client-side backoff contribute measurably; at 9/16/32 hosts no call retries.",
          "tab:decomposition"))
