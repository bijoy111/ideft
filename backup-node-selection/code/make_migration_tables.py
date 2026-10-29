"""Generate the LaTeX tables for the measured migration cost.

Reads ../findings/migration_cost.json, written by measure_migration_cost.py.
As elsewhere in this repository, no number in the paper is typed by hand.

Usage:
    python make_migration_tables.py ../findings/migration_cost.json \
        > ../paper/tables_migration.tex
"""

import json
import sys

ENC = ["packed", "json", "xmlrpc"]
ENC_NAME = {"packed": r"\texttt{packed}", "json": r"\texttt{json}",
            "xmlrpc": r"\texttt{xmlrpc} (DEFT)"}


def occupancy_table(records):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Flow-table occupancy at steady state, and the share of it "
             r"that changes in one replication round. `Little' is the Little's-law "
             r"prediction; its agreement with the simulated occupancy is a check on "
             r"the workload generator, not a separate result. The delta share is "
             r"what makes migration expensive relative to ordinary replication: a "
             r"round ships a few percent of the table, a migration ships all of it.}",
             r"\label{tab:occupancy}", r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{rrrrr}", r"\toprule",
             r"flows/s & entries & Little & delta/round & delta share \\",
             r"\midrule"]
    for r in records:
        lines.append("{:.0f} & {:,.0f} & {:,.0f} & {:,.0f} & {:.1f}\\% \\\\".format(
            r["arrival_rate"], r["entries"], r["predicted_entries"],
            r["delta_entries"], 100 * r["delta_fraction"]))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def encoding_table(records):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Serialized size of the replicated state. "
             r"\texttt{xmlrpc} is the encoding DEFT actually puts on the wire. Its "
             r"\texttt{<int>} is signed 32-bit, so an IPv4 address does not fit and "
             r"has to be sent as text; that and per-value markup account for the "
             r"inflation over a fixed-width binary encoding of the same state.}",
             r"\label{tab:encoding}", r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{rlrrr}", r"\toprule",
             r"flows/s & encoding & B/entry & full table & vs.\ packed \\",
             r"\midrule"]
    for r in records:
        base = r["encodings"]["packed"]["full_bytes"]
        for enc in ENC:
            e = r["encodings"][enc]
            lines.append("{:.0f} & {} & {:.0f} & {:.1f}\\,MB & {:.2f}$\\times$ \\\\".format(
                r["arrival_rate"], ENC_NAME[enc], e["bytes_per_entry"],
                e["full_bytes"] / 1e6, e["full_bytes"] / base))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def m_table(derived, bandwidth=10.0):
    rows = [d for d in derived if d["bandwidth_gbps"] == bandwidth]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{The measured migration cost $M$, in rounds of ordinary "
             r"replication service, at " + "{:.0f}".format(bandwidth) +
             r"\,Gb/s. $M$ is close to the reciprocal of the delta share and is "
             r"therefore a property of traffic churn rather than of the wire "
             r"format: a costlier encoding inflates the full transfer and the "
             r"per-round delta alike. Every value is several times the $M=3$ that "
             r"the online sweeps originally assumed.}",
             r"\label{tab:mrounds}", r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{rlrrr}", r"\toprule",
             r"flows/s & encoding & migration & service/round & $M$ (rounds) \\",
             r"\midrule"]
    for d in rows:
        lines.append("{:.0f} & {} & {:.0f}\\,ms & {:.1f}\\,ms & {:.0f} \\\\".format(
            d["arrival_rate"], ENC_NAME[d["encoding"]],
            1000 * d["migration_s"], 1000 * d["service_s_per_round"],
            d["M_rounds"]))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def crossover_table(crossovers, bandwidth=10.0, rate=None):
    rows = [c for c in crossovers if c["bandwidth_gbps"] == bandwidth
            and (rate is None or c["arrival_rate"] == rate)]
    gaps = sorted({c["gap_ms"] for c in rows})
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{How long degradation must persist before migrating away "
             r"from it is worth the round trip, at " + "{:.0f}".format(bandwidth) +
             r"\,Gb/s and " + "{:.0f}".format(rows[0]["arrival_rate"]) +
             r" flows/s. Columns are the extra per-round cost the degraded backup "
             r"imposes, spanning the delay range of the failure kinds in "
             r"\S\ref{sec:weights}. Under DEFT's own encoding even a severe "
             r"degradation must outlast a minute before reacting to it pays.}",
             r"\label{tab:crossover}", r"\small",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "r" * len(gaps) + "}", r"\toprule",
             "encoding & " + " & ".join("{:.0f}\\,ms".format(g) for g in gaps) + r" \\",
             r"\midrule"]
    for enc in ENC:
        cells = []
        for g in gaps:
            match = [c for c in rows if c["encoding"] == enc and c["gap_ms"] == g]
            cells.append("{:.0f}\\,s".format(match[0]["crossover_s"]) if match else "---")
        lines.append(ENC_NAME[enc] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    payload = json.load(open(sys.argv[1]))
    records = payload["records"]
    mid = records[len(records) // 2]["arrival_rate"]
    print("\n\n".join([
        occupancy_table(records),
        encoding_table(records),
        m_table(payload["derived"]),
        crossover_table(payload["crossovers"], rate=mid),
    ]))


if __name__ == "__main__":
    main()
