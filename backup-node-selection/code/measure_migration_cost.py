"""
Measure the migration cost M, instead of assuming it.

Why
---
The online analysis turns on one quantity: the cost of moving a primary's state
to a different backup, expressed in units of one round of ordinary replication
service. Everything downstream -- whether snapshot-chasing is ruinous, how much
hysteresis buys, and the crossover duration 2M/gap beyond which reacting to
degradation pays -- is a function of it. Up to now it was a swept parameter with
no empirical value, because DEFT's backup discards the state it is sent and the
experiments ship a thirty-byte placeholder.

With a real flow table (flow_state.py) filled by a real workload (workload.py),
M becomes measurable:

    M_seconds = serialize(full table) + full_bytes / bandwidth + load(full table)
    S_seconds = serialize(delta)      + delta_bytes / bandwidth + load(delta)
    M_rounds  = M_seconds / S_seconds

S is the service cost of one round: the work of replicating the flows that
changed during that round. M_rounds is therefore how many rounds of ordinary
operation one migration is worth, which is exactly the unit the online model
uses.

Serialization and load times are measured on this machine. Transfer time is
computed from the serialized size and a link rate, swept across plausible rates,
rather than measured over a loopback that would misrepresent it. A live
end-to-end check over the real transports is available with --live.

Usage:
    python measure_migration_cost.py --out ../findings/
    python measure_migration_cost.py --live --out ../findings/
"""

import argparse
import json
import statistics
import time

import flow_state as fs
import workload as wl

ENCODINGS = ["packed", "json", "xmlrpc"]

# Link rates to report M against. The migration is a bulk transfer, so it is
# bandwidth-bound; the per-round delta is small enough to be latency-bound, which
# is why both terms are carried separately below.
BANDWIDTHS_GBPS = [1.0, 10.0, 25.0]

# Extra per-round service cost imposed by a degraded backup, in milliseconds.
# Spans the delay range of the degradation kinds in degradation.py, whose
# link_congested mode reaches 60 ms at full severity.
DEGRADED_GAP_MS = [5.0, 20.0, 60.0]


def timeit(fn, repeats=5, budget_s=2.0):
    """Best-of-n wall time, in seconds.

    Best rather than mean, because we want the cost of the operation and not of
    whatever else the machine was doing. Repeats stop early once `budget_s` of
    wall time has been spent: at the top of the arrival-rate sweep a single
    XML-RPC round trip over a quarter-million flows takes tens of seconds, and
    five repeats of it would dominate the run without improving the estimate.
    """
    best, spent = float("inf"), 0.0
    for i in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        spent += elapsed
        if i + 1 < repeats and spent + best > budget_s:
            break
    return best


def measure_one(arrival_rate, idle_timeout, warm_rounds, round_s, seed=0):
    """Fill a table to steady state and measure everything about one round."""
    params = wl.WorkloadParams(arrival_rate=arrival_rate, idle_timeout=idle_timeout)
    engine = wl.FlowWorkload(params, seed=seed, round_s=round_s)
    engine.warm(warm_rounds)

    table = engine.table
    delta = engine.step()           # one round's worth of change, at steady state

    record = {
        "arrival_rate": arrival_rate,
        "idle_timeout": idle_timeout,
        "round_s": round_s,
        "entries": len(table),
        "predicted_entries": wl.expected_occupancy(params),
        "delta_entries": len(delta),
        "delta_fraction": len(delta) / max(len(table), 1),
        "encodings": {},
    }

    delta_table = fs.FlowTable(table=dict(delta))
    for encoding in ENCODINGS:
        full_blob = table.serialize(encoding)
        delta_blob = delta_table.serialize(encoding)
        record["encodings"][encoding] = {
            "full_bytes": len(full_blob),
            "delta_bytes": len(delta_blob),
            "bytes_per_entry": len(full_blob) / max(len(table), 1),
            "serialize_full_s": timeit(lambda: table.serialize(encoding)),
            "load_full_s": timeit(
                lambda: fs.FlowTable.deserialize(full_blob, encoding)),
            "serialize_delta_s": timeit(lambda: delta_table.serialize(encoding)),
            "load_delta_s": timeit(
                lambda: fs.FlowTable.deserialize(delta_blob, encoding)),
        }
    return record


def derive_m(record, encoding, bandwidth_gbps):
    """Turn one measurement into M, in seconds and in rounds."""
    e = record["encodings"][encoding]
    rate_bytes_per_s = bandwidth_gbps * 1e9 / 8.0

    migration_s = (e["serialize_full_s"]
                   + e["full_bytes"] / rate_bytes_per_s
                   + e["load_full_s"])
    service_s = (e["serialize_delta_s"]
                 + e["delta_bytes"] / rate_bytes_per_s
                 + e["load_delta_s"])
    return {
        "encoding": encoding,
        "bandwidth_gbps": bandwidth_gbps,
        "migration_s": migration_s,
        "service_s_per_round": service_s,
        "M_rounds": migration_s / service_s if service_s else float("inf"),
    }


def crossover_rounds(m_rounds, service_s, gap_ms):
    """Rounds of degradation beyond which migrating is worth it.

    A policy that leaves a degraded backup must also come back when it recovers,
    so the round trip costs 2M against a saving of `gap` per round; migration
    pays only once degradation outlasts 2M/gap rounds. Expressed here in the
    same round units as M, and converted to seconds by the caller.
    """
    gap_rounds = (gap_ms / 1000.0) / service_s if service_s else 0.0
    if gap_rounds <= 0:
        return float("inf")
    return 2.0 * m_rounds / gap_rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrival-rates", type=float, nargs="+",
                    default=[500.0, 2000.0, 8000.0])
    ap.add_argument("--idle-timeouts", type=float, nargs="+", default=[30.0])
    ap.add_argument("--warm-rounds", type=int, default=150)
    ap.add_argument("--round-s", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--live", action="store_true",
                    help="also time a real update_state round trip over both transports")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report, payload = [], {"config": vars(args)}
    report.append("=" * 78)
    report.append("Migration cost of real network-function state")
    report.append("=" * 78)
    report.append("")
    report.append("A connection-tracking flow table (flow_state.py) filled to steady state")
    report.append("by a flow-arrival workload (workload.py). DEFT's own backup discards the")
    report.append("state it receives, so none of this could be measured before.")
    report.append("")

    records = []
    for rate in args.arrival_rates:
        for timeout in args.idle_timeouts:
            per_seed = [measure_one(rate, timeout, args.warm_rounds,
                                    args.round_s, seed=s)
                        for s in range(args.seeds)]
            merged = dict(per_seed[0])
            merged["entries"] = statistics.mean(r["entries"] for r in per_seed)
            merged["delta_entries"] = statistics.mean(
                r["delta_entries"] for r in per_seed)
            merged["delta_fraction"] = statistics.mean(
                r["delta_fraction"] for r in per_seed)
            records.append(merged)
    payload["records"] = records

    # -- occupancy ------------------------------------------------------
    report.append("Table occupancy at steady state")
    report.append("-" * 78)
    report.append("{:>12}{:>12}{:>14}{:>12}{:>12}{:>14}".format(
        "flows/s", "timeout", "entries", "Little", "delta/rnd", "delta share"))
    for r in records:
        report.append("{:>12.0f}{:>12.0f}{:>14,.0f}{:>12,.0f}{:>12,.0f}{:>13.1%}".format(
            r["arrival_rate"], r["idle_timeout"], r["entries"],
            r["predicted_entries"], r["delta_entries"], r["delta_fraction"]))
    report.append("")
    report.append("'Little' is the Little's-law prediction arrival_rate x (mean lifetime +")
    report.append("idle timeout); agreement is a check on the generator, not a second result.")
    report.append("")

    # -- wire size ------------------------------------------------------
    report.append("Serialized size of the full table")
    report.append("-" * 78)
    report.append("{:>12}{:>12}{:>16}{:>16}{:>14}".format(
        "flows/s", "encoding", "bytes/entry", "full table", "vs packed"))
    for r in records:
        base = r["encodings"]["packed"]["full_bytes"]
        for enc in ENCODINGS:
            e = r["encodings"][enc]
            report.append("{:>12.0f}{:>12}{:>16.1f}{:>16,}{:>13.2f}x".format(
                r["arrival_rate"], enc, e["bytes_per_entry"],
                e["full_bytes"], e["full_bytes"] / base))
    report.append("")
    report.append("xmlrpc is what DEFT puts on the wire. Its <int> is signed 32-bit, so an")
    report.append("IPv4 address does not fit and must be sent as text; that, plus per-value")
    report.append("markup, is where the inflation comes from.")
    report.append("")

    # -- M ---------------------------------------------------------------
    report.append("Migration cost M, in rounds of ordinary replication service")
    report.append("-" * 78)
    report.append("{:>12}{:>10}{:>8}{:>14}{:>16}{:>12}".format(
        "flows/s", "encoding", "Gb/s", "migration", "service/round", "M (rounds)"))
    derived = []
    for r in records:
        for enc in ENCODINGS:
            for bw in BANDWIDTHS_GBPS:
                d = derive_m(r, enc, bw)
                d["arrival_rate"] = r["arrival_rate"]
                derived.append(d)
                report.append("{:>12.0f}{:>10}{:>8.0f}{:>13.1f}ms{:>15.2f}ms{:>12.1f}".format(
                    r["arrival_rate"], enc, bw, 1000 * d["migration_s"],
                    1000 * d["service_s_per_round"], d["M_rounds"]))
    payload["derived"] = derived
    report.append("")

    # -- crossover --------------------------------------------------------
    report.append("Crossover: how long degradation must last before migrating pays")
    report.append("-" * 78)
    report.append("A policy that leaves a degraded backup must come back when it recovers,")
    report.append("so the round trip costs 2M against a saving of `gap` per round.")
    report.append("")
    report.append("{:>12}{:>10}{:>8}{:>12}{:>16}{:>14}".format(
        "flows/s", "encoding", "Gb/s", "gap (ms)", "crossover rnds", "crossover s"))
    crossovers = []
    for r in records:
        for enc in ENCODINGS:
            for bw in BANDWIDTHS_GBPS:
                d = derive_m(r, enc, bw)
                for gap in DEGRADED_GAP_MS:
                    rounds = crossover_rounds(d["M_rounds"],
                                              d["service_s_per_round"], gap)
                    crossovers.append({"arrival_rate": r["arrival_rate"],
                                       "encoding": enc, "bandwidth_gbps": bw,
                                       "gap_ms": gap, "crossover_rounds": rounds,
                                       "crossover_s": rounds * r["round_s"]})
                    report.append("{:>12.0f}{:>10}{:>8.0f}{:>12.0f}{:>16.1f}{:>14.1f}".format(
                        r["arrival_rate"], enc, bw, gap, rounds,
                        rounds * r["round_s"]))
    payload["crossovers"] = crossovers

    if args.live:
        report.append("")
        report.extend(live_check(records[len(records) // 2]))

    text = "\n".join(report)
    print(text)
    if args.out:
        import os
        base = os.path.join(args.out, "migration_cost")
        open(base + ".txt", "w").write(text + "\n")
        json.dump(payload, open(base + ".json", "w"), indent=2)
        print("\nwrote {}.txt and {}.json".format(base, base))


def live_check(record):
    """Time a real update_state round trip carrying a real delta, on both
    transports, to confirm the analytic model is not missing a large constant."""
    import multiprocessing
    import xmlrpc.client

    lines = ["Live round trip with a real delta payload (loopback)",
             "-" * 78]
    params = wl.WorkloadParams(arrival_rate=record["arrival_rate"])
    engine = wl.FlowWorkload(params, seed=0)
    engine.warm(60)
    delta = engine.step()
    payload = {fs.key_to_str(k): [str(e.packets), str(e.bytes),
                                  fs._ipv4(e.nat_ip), e.nat_port,
                                  e.last_seen, e.tcp_state]
               for k, e in delta.items()}

    import rpc_contention_experiment as rpc
    port = 18800
    proc = multiprocessing.Process(target=rpc._serve_backup,
                                   args=("127.0.0.1", port, 0.0), daemon=True)
    proc.start()
    time.sleep(1.0)
    try:
        proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:{}".format(port))
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            proxy.update_state(payload)
            samples.append(time.perf_counter() - start)
        lines.append("  xmlrpc: {:,} entries, median {:.1f} ms".format(
            len(delta), 1000 * statistics.median(samples)))
    except Exception as exc:                       # pragma: no cover
        lines.append("  xmlrpc: failed ({})".format(exc))
    finally:
        proc.terminate()
        proc.join(timeout=5)
    return lines


if __name__ == "__main__":
    main()
