"""Generate the LaTeX tables for the online-selection section from the findings JSON.

Numbers in the paper are never typed by hand; they come from here, exactly as the
one-shot tables come from make_tables.py.

Usage:
    python make_online_tables.py ../findings/online_competitive.json \
        > ../paper/tables_online.tex
"""

import json
import sys

# Column order and display names. `static` is kept out of the headline tables and
# discussed in prose, because a policy that sits on a dead host is not a baseline
# anyone would defend; `failover` is the honest version of the deployed policy.
COLUMNS = [
    ("failover", r"\textsc{failover}"),
    ("snapshot", r"\textsc{snapshot}"),
    ("periodic20", r"\textsc{periodic}"),
    ("hysteresis1", r"\textsc{hyst}($1$)"),
    ("hysteresis2", r"\textsc{hyst}($2$)"),
]


def fmt(x, places=2):
    if x != x:          # NaN
        return "---"
    return "{:.{}f}".format(x, places)


def grid_table(grid, variable, caption, label, include_static=False):
    cols = ([("static", r"\textsc{static}")] if include_static else []) + COLUMNS
    spec = "r" + "r" * len(cols)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}",
             r"\label{" + label + "}",
             r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{" + spec + "}",
             r"\toprule",
             variable + " & " + " & ".join(n for _, n in cols) + r" \\",
             r"\midrule"]
    for row in grid:
        cells = [fmt(row[k]["excess_ratio"]) for k, _ in cols]
        best = min(range(len(cells)),
                   key=lambda i: row[cols[i][0]]["excess_ratio"])
        cells[best] = r"\textbf{" + cells[best] + "}"
        lines.append("{} & ".format(row["value"]) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def migration_share_table(grid, variable, caption, label):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}",
             r"\label{" + label + "}",
             r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{r" + "r" * len(COLUMNS) + "}",
             r"\toprule",
             variable + " & " + " & ".join(n for _, n in COLUMNS) + r" \\",
             r"\midrule"]
    for row in grid:
        cells = ["{:.0f}\\%".format(100 * row[k]["migration_share"]) for k, _ in COLUMNS]
        lines.append("{} & ".format(row["value"]) + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def alpha_table(rows):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Hysteresis threshold $\alpha$: competitive ratio, and the "
             r"fraction of the Proposition~\ref{prop:budget} migration budget that "
             r"is actually used. The bound $1/\alpha$ holds at every $\alpha$, with "
             r"substantial slack, so the guarantee is not what limits the policy.}",
             r"\label{tab:online-alpha}",
             r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{rrrrr}",
             r"\toprule",
             r"$\alpha$ & excess ratio & migrations & budget used & bound $1/\alpha$ \\",
             r"\midrule"]
    for r in rows:
        lines.append("{} & {} & {:.0f} & {} & {} \\\\".format(
            r["alpha"], fmt(r["ratio"], 3), r["migrations"],
            fmt(r["budget_used"], 3), fmt(r["budget_bound"], 3)))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    payload = json.load(open(sys.argv[1]))
    cfg = payload["config"]
    setup = ("{} hosts, {} primaries, horizon {}, {} seeds. Entries are the "
             "excess-over-floor ratio: cost above the unavoidable floor divided by "
             "the offline optimum's, so $1.00$ means the policy decided as well as "
             "an algorithm that knew the entire trace in advance. Best in each row "
             "is bold.").format(cfg["hosts"], cfg["primaries"], cfg["horizon"],
                                cfg["seeds"])

    out = []
    out.append(grid_table(
        payload["churn"], r"$p_{\mathrm{deg}}$",
        "Competitive ratio against churn rate, with hard failures at "
        "$p_{\\mathrm{fail}} = p_{\\mathrm{deg}}/10$. " + setup + " "
        "\\textsc{static} is shown here to quantify what failover alone is worth.",
        "tab:online-churn", include_static=True))
    out.append(grid_table(
        payload["persistence"], "duration",
        "Competitive ratio against how long degradation lasts (expected degraded "
        "duration, in rounds), with hard failures disabled so the sweep isolates "
        "reacting to degradation from reacting to death. " + setup,
        "tab:online-persistence"))
    out.append(grid_table(
        payload["migration"], "$M$",
        "Competitive ratio against migration cost $M$. \\textsc{snapshot} degrades "
        "without limit, as Theorem~\\ref{thm:snapshot} requires; the hysteresis "
        "policies are flat because their threshold is expressed in units of $M$. "
        + setup, "tab:online-migration"))
    out.append(migration_share_table(
        payload["migration"], "$M$",
        "Share of each policy's total cost spent on migration, against $M$. This is "
        "the quantity Proposition~\\ref{prop:budget} bounds for hysteresis and "
        "Theorem~\\ref{thm:snapshot} shows is unbounded for \\textsc{snapshot}.",
        "tab:online-migshare"))
    out.append(grid_table(
        payload["hosts"], "hosts",
        "Competitive ratio against topology size. The thrashing of "
        "\\textsc{hyst}($0.5$) at 64 hosts, omitted here and reported in the text, "
        "is what a weak migration budget looks like in practice. " + setup,
        "tab:online-hosts"))
    out.append(alpha_table(payload["alpha"]))
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
