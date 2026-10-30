"""
Generate the CloudLab testbed tables from the findings files.

Every number in the testbed section of the paper comes from here, so that a
rerun regenerates the tables rather than inviting a hand transcription. The
inputs are the findings files themselves, which carry the command that produced
them in their header, so a table cannot drift from the run it claims to report.
"""

import collections
import csv
import json
import os
import re
import sys

FINDINGS = "../findings"
RESULTS = "results_rpc"

RUNS = {
    "asshipped": "latency_128hosts_cloudlab.txt",
    "backlog": "latency_128hosts_cloudlab_backlog512.txt",
    "degraded": "latency_128hosts_cloudlab_degraded_spread.txt",
}

# The degraded set the spread selection produces at n_hosts=128, n_degraded=16.
DEGRADED = {"w1", "w2", "w27", "w34", "w41", "w49", "w56", "w63", "w70", "w78",
            "w85", "w92", "w106", "w113", "w120", "w128"}

ORDER = ["optimal", "heuristic", "round_robin", "random", "rl", "concentrated"]
TEX = {"optimal": r"\texttt{optimal}", "heuristic": r"\texttt{heuristic}",
       "round_robin": r"\texttt{round\_robin}", "random": r"\texttt{random}",
       "rl": r"\texttt{PPO}", "concentrated": r"\texttt{concentrated}"}


def read(name):
    with open(os.path.join(FINDINGS, RUNS[name])) as handle:
        return handle.read()


def _section(text, title):
    """The body of a banner-delimited block.

    Blocks are introduced by a line of the form `===== Title ... =====`, and the
    title may be followed by further words on that same line, so the split has
    to be line-oriented: find the banner line containing the title, then take
    every line up to the next banner. Splitting on the title string alone
    returns the remainder of the banner line, which parses as nothing and fails
    silently.
    """
    body, collecting = [], False
    for line in text.splitlines():
        banner = line.startswith("=====")
        if banner and title in line:
            collecting = True
            continue
        if collecting:
            if banner:
                break
            body.append(line)
    if not collecting:
        raise ValueError("no section titled {!r}".format(title))
    return "\n".join(body)


def summary(text):
    """The per-method run-level summary: method -> dict of statistics."""
    # The Wilcoxon tables that follow also begin their rows with a method name,
    # so the parse has to stop at the next banner.
    block = _section(text, "Run-level summary: service latency")
    rows = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] in TEX:
            rows[parts[0]] = {
                "runs": int(parts[1]), "mean": float(parts[2]),
                "median": float(parts[3]), "p95": float(parts[4]),
                "p99": float(parts[5]), "fail": float(parts[6]),
            }
    if len(rows) != len(TEX):
        raise ValueError("expected {} methods, parsed {}".format(len(TEX), len(rows)))
    return rows


def load_curve(text):
    """The latency-vs-backup-load table: load -> (n_units, mean)."""
    block = _section(text, "Service latency vs. backup load")
    out = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            out[int(parts[0])] = (int(parts[1]), float(parts[2]))
    return out


def degraded_share(csv_name):
    """Fraction of a method's calls that landed on a degraded backup."""
    hits, total = collections.Counter(), collections.Counter()
    with open(os.path.join(RESULTS, csv_name)) as handle:
        for row in csv.DictReader(handle):
            total[row["method"]] += 1
            if row["backup"] in DEGRADED:
                hits[row["method"]] += 1
    return {m: 100.0 * hits[m] / total[m] for m in total}


def table_main(homog, degr, share):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{128 hosts, 80 primaries, on 10 bare-metal nodes with every "
        r"backup pinned to its own physical core. Left: all hosts identical. "
        r"Right: 16 of 128 hosts degraded by \texttt{tc netem} on the server's "
        r"source port, and the share of each method's calls that reached one. "
        r"Mean and p99 in milliseconds.}",
        r"\label{tab:cloudlab}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Homogeneous} & \multicolumn{3}{c}{16 degraded} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Method & Mean & p99 & Mean & p99 & Degraded (\%) \\",
        r"\midrule",
    ]
    for m in ORDER:
        lines.append("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            TEX[m], homog[m]["mean"], homog[m]["p99"],
            degr[m]["mean"], degr[m]["p99"], share[m]))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_backlog(asshipped, backlog):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{The effect of DEFT's listen backlog, all hosts homogeneous and "
        r"every other variable held fixed. DEFT constructs "
        r"\texttt{SimpleXMLRPCServer} unmodified, so its accept queue holds five "
        r"connections. p99 in milliseconds.}",
        r"\label{tab:backlog}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Backlog 5 (as shipped)} & \multicolumn{2}{c}{Backlog 512} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Method & p95 & p99 & p95 & p99 \\",
        r"\midrule",
    ]
    for m in ORDER:
        lines.append("{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
            TEX[m], asshipped[m]["p95"], asshipped[m]["p99"],
            backlog[m]["p95"], backlog[m]["p99"]))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_load(asshipped, backlog):
    loads = [1, 2, 4, 16, 45, 79]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean service latency against the number of primaries sharing a "
        r"backup, homogeneous hosts. The steep response under the shipped backlog "
        r"is dropped SYNs and the client's 1\,s retransmission timeout, not "
        r"queueing at the backup. Milliseconds.}",
        r"\label{tab:loadcurve}",
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Primaries sharing & Backlog 5 & Backlog 512 \\",
        r"\midrule",
    ]
    for load in loads:
        if load in asshipped and load in backlog:
            lines.append("{} & {:.1f} & {:.1f} \\\\".format(
                load, asshipped[load][1], backlog[load][1]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    texts = {k: read(k) for k in RUNS}
    stats = {k: summary(v) for k, v in texts.items()}
    share = degraded_share("raw_128hosts_degraded_spread.csv")
    curves = {k: load_curve(texts[k]) for k in ("asshipped", "backlog")}

    out = "\n\n".join([
        "% Generated by make_testbed_tables.py -- do not edit by hand.",
        table_main(stats["backlog"], stats["degraded"], share),
        table_backlog(stats["asshipped"], stats["backlog"]),
        table_load(curves["asshipped"], curves["backlog"]),
    ]) + "\n"

    path = "../paper/tables_testbed.tex"
    with open(path, "w") as handle:
        handle.write(out)
    print("wrote {} ({} bytes)".format(path, len(out)))

    facts = {
        "homogeneous": {m: stats["backlog"][m] for m in ORDER},
        "degraded": {m: stats["degraded"][m] for m in ORDER},
        "as_shipped": {m: stats["asshipped"][m] for m in ORDER},
        "degraded_share_pct": share,
        "load_curve_asshipped": {str(k): v[1] for k, v in curves["asshipped"].items()},
        "load_curve_backlog": {str(k): v[1] for k, v in curves["backlog"].items()},
    }
    with open(os.path.join(FINDINGS, "cloudlab_facts.json"), "w") as handle:
        json.dump(facts, handle, indent=1, sort_keys=True)
    print("wrote {}/cloudlab_facts.json".format(FINDINGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
