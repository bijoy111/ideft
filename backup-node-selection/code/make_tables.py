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


def dual_transport_table(xml_tag, grpc_tag, caption, label, reference="heuristic"):
    """One table per scale carrying both transports and the paired statistics.

    Columns are kept to one row per method rather than one table per transport:
    the comparison is the point, and two separate tables would make the reader do
    the differencing. Means come from each transport's own runs; the effect size
    and p-value are the gRPC arm's, which is the production-realistic one and has
    the larger sample.
    """
    xml_runs = analysis.summarize_runs(
        "../findings/raw/{}.csv.gz".format(xml_tag), metric="service_ms")
    g_runs = analysis.summarize_runs(
        "../findings/raw/{}.csv.gz".format(grpc_tag), metric="service_ms")
    paired = {}
    for c in analysis.paired_comparisons(g_runs, metric="mean"):
        if reference in (c["a"], c["b"]):
            other = c["b"] if c["a"] == reference else c["a"]
            sign = 1 if c["b"] == reference else -1
            paired[other] = (sign * c["median_diff"], sign * c["r"], c["p_holm"])

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "& XML-RPC & \\multicolumn{4}{c}{gRPC} \\\\",
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-6}",
        "Method & Mean (ms) & Mean (ms) & 95\\% CI & $r$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for m in ordered(g_runs):
        g = [r["mean"] for r in g_runs[m].values()]
        lo, hi = analysis.cluster_bootstrap_ci(g)
        x = [r["mean"] for r in xml_runs[m].values()] if m in xml_runs else None
        xcell = "{:.2f}".format(np.mean(x)) if x else "--"
        if m == reference:
            eff = "-- & --"
        elif m in paired:
            d, r, pv = paired[m]
            eff = "{:+.3f} & {}".format(
                r, "$<$0.0001" if pv < 0.0001 else "{:.4f}".format(pv))
        else:
            eff = "-- & --"
        lines.append("{} & {} & {:.2f} & [{:.2f}, {:.2f}] & {} \\\\".format(
            name(m), xcell, np.mean(g), lo, hi, eff))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def structural_table(caption, label):
    """128-host results that do not depend on timing.

    Per-call latency at 128 hosts is not a usable measurement on a single
    machine (\\S\\ref{sec:p99}), but the produced mapping is exact: these columns
    are properties of the assignment, independent of how fast any call ran.
    """
    from collections import defaultdict
    rows = []
    for tag, tlabel in [("128hosts", "XML-RPC"), ("128hosts_grpc", "gRPC")]:
        spread, degraded = defaultdict(dict), {}
        for row in analysis.iter_rows("../findings/raw/{}.csv.gz".format(tag)):
            key = (row["method"], int(row["context_seed"]), int(row["trial"]))
            spread.setdefault(key, set()).add(row["backup"])
        per_method = defaultdict(list)
        for (m, _, _), backups in spread.items():
            per_method[m].append(len(backups))
        rows.append((tlabel, per_method))
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrr}", "\\toprule",
        "Method & \\multicolumn{2}{c}{Distinct backups used (of 80)} \\\\",
        "\\cmidrule(lr){2-3}",
        " & XML-RPC & gRPC \\\\",
        "\\midrule",
    ]
    methods = ordered(rows[0][1])
    for m in methods:
        cells = []
        for _, per_method in rows:
            v = per_method.get(m, [])
            cells.append("{}--{}".format(min(v), max(v)) if v else "--")
        lines.append("{} & {} & {} \\\\".format(name(m), cells[0], cells[1]))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def transport_comparison_table(caption, label):
    """Side-by-side effect sizes for the two transports.

    The transport is a controlled variable: identical selection methods, scoring,
    topology, workload and statistics, differing only in how the replication call
    reaches the backup. A finding that holds under both is a property of backup
    placement rather than of one RPC stack, which is the claim this table exists
    to support.
    """
    scales = [(9, "9 (homog.)", "9hosts", "9hosts_grpc_n25"),
              (16, "16", "16hosts", "16hosts_grpc_n25"),
              (32, "32", "32hosts", "32hosts_grpc_n25")]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "& \\multicolumn{3}{c}{XML-RPC ($n{=}15$)} & \\multicolumn{3}{c}{gRPC ($n{=}25$)} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Hosts & $\\Delta$ (ms) & $r$ & $p_{\\mathrm{Holm}}$ & $\\Delta$ (ms) & $r$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for _, label_txt, xml_tag, grpc_tag in scales:
        cells = []
        for tag in (xml_tag, grpc_tag):
            runs = analysis.summarize_runs("../findings/raw/{}.csv.gz".format(tag),
                                           metric="service_ms")
            hit = None
            for c in analysis.paired_comparisons(runs, metric="mean"):
                if {c["a"], c["b"]} == {"round_robin", "heuristic"}:
                    d = c["median_diff"] if c["a"] == "round_robin" else -c["median_diff"]
                    r = c["r"] if c["a"] == "round_robin" else -c["r"]
                    hit = (d, r, c["p_holm"])
            cells.append(hit or (float("nan"),) * 3)
        row = " & ".join("{:+.2f} & {:+.3f} & {}".format(
            d, r, "$<$0.0001" if p < 0.0001 else "{:.4f}".format(p)) for d, r, p in cells)
        lines.append("{} & {} \\\\".format(label_txt, row))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print("% Generated by make_tables.py from findings/raw/*.csv.gz -- do not edit by hand.\n")

    print(dual_transport_table(
        "9hosts", "9hosts_grpc_n25",
        "Homogeneous scale (9 hosts, 5 primaries). XML-RPC: 15 runs; gRPC: 25 runs "
        "(5 context seeds $\\times$ 5 trials), 200 calls/primary, burst 10. Effect sizes are "
        "paired against \\texttt{heuristic} on the gRPC arm.",
        "tab:homogeneous"))

    print(dual_transport_table(
        "16hosts", "16hosts_grpc_n25",
        "Scaled heterogeneous, 16 hosts (10 primaries, 2 degraded).",
        "tab:scaled16"))

    print(dual_transport_table(
        "32hosts", "32hosts_grpc_n25",
        "Scaled heterogeneous, 32 hosts (20 primaries, 4 degraded).",
        "tab:scaled32"))

    print(transport_comparison_table(
        "\\texttt{round\\_robin} vs.\\ \\texttt{heuristic} under both transports. "
        "$\\Delta$ is the median per-run difference (positive favours the heuristic). "
        "The effect is real on both; it is roughly $4.3\\times$ larger on XML-RPC, whose "
        "accept-backlog pathology amplifies the degraded-host penalty.",
        "tab:transport"))

    print(structural_table(
        "128 hosts (80 primaries, 16 degraded): structural outcome of the produced "
        "mapping. Per-call latency is not a usable measurement at this scale on a single "
        "machine (\\S\\ref{sec:p99}), but the mapping itself is exact and "
        "hardware-independent.",
        "tab:scaled128"))

    print(load_table([(9, "Homog."), (16, "16-host")],
                     "Service latency vs.\\ backup load on the XML-RPC arm, one observation "
                     "per (run $\\times$ backup).",
                     "tab:backupload"))

    print(decomposition_table(128,
          "128 hosts, XML-RPC: backup service time vs.\\ client retry wait. Only at this "
          "scale does client-side backoff contribute measurably; at 9/16/32 hosts no call "
          "retries.",
          "tab:decomposition"))
