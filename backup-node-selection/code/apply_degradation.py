"""
Impose degradation physically, on the hosts a run marks as degraded.

Why this exists
---------------
The single-machine harness makes a degraded backup slower by sleeping inside the
server. That measures our model of degradation, not degradation: the sleep is
exactly as long as we said it would be, costs no CPU, queues nothing, and cannot
interact with the transport. `RemoteDeployment` therefore refuses a `slow_hosts`
argument, and this script supplies the mechanism it refuses to fake.

Why per-port rather than per-node
---------------------------------
`tc netem` attaches to an interface, but a node carries many backups. Delaying
the interface would degrade every backup on that node, including the healthy
ones the run depends on as a control. Each backup owns a distinct TCP port, so a
`prio` qdisc with a `u32` filter on source port delays exactly the degraded
servers' traffic and leaves their co-tenants alone.

Band 4 of a 4-band `prio` is unreachable through the default priomap (which uses
bands 0-2 only), so traffic reaches netem only by matching a filter. Healthy
flows are not merely delayed by 0ms -- they never enter the netem queue at all.

Per-host delays
---------------
Under the realistic degradation model each degraded host has its own delay. A
node then gets one extra band per distinct delay (bands 4, 5, ...), each with its
own netem, and each degraded port is filtered into the band for its delay. prio
allows 16 bands, so a node can carry up to 13 distinct delays, which is every
server on a 13-server node. With a single delay the commands are exactly those of
the original single-delay form.

The delay applied is the model's delay_ms. netem delays every packet the server
sends, so a connection-per-call protocol pays it twice, on the SYN-ACK and on the
response: the binary model's 50 ms measured 101 ms per call. The same holds here,
and the RTT probe in rpc_contention_experiment.py measures what was actually
imposed rather than what the model said.

Usage
-----
    # see the commands, change nothing
    python3 apply_degradation.py --manifest ../findings/m.json \\
        --hosts w81 w82 --delay-ms 50 --iface enp65s0f0np0

    # the hosts and per-host delays a scenario defines (realistic model)
    python3 apply_degradation.py --manifest ../findings/m.json \\
        --scenario --n-hosts 128 --n-primaries 80 --n-degraded 16 \\
        --degradation-model realistic --iface enp65s0f0np0

    # apply them
    python3 apply_degradation.py --manifest ../findings/m.json \\
        --hosts w81 w82 --delay-ms 50 --iface enp65s0f0np0 --apply

    # check what the kernel actually has, and how much traffic matched
    python3 apply_degradation.py --manifest ../findings/m.json --iface enp65s0f0np0 --show

    # remove it
    python3 apply_degradation.py --manifest ../findings/m.json --iface enp65s0f0np0 --clear
"""

import argparse
import json
import subprocess
import sys

BAND = 4            # 1:4 -- the band the default priomap never selects
PRIO = 4


def load(path):
    with open(path) as handle:
        return json.load(handle)


def by_node(manifest, host_ids):
    """Group the requested hosts by the node they run on."""
    hosts = manifest["hosts"]
    missing = sorted(set(host_ids) - set(hosts))
    if missing:
        raise KeyError("manifest has no server for: {}".format(", ".join(missing)))
    grouped = {}
    for host_id in host_ids:
        spec = hosts[host_id]
        grouped.setdefault(spec["node"], []).append((host_id, spec["port"]))
    for node in grouped:
        grouped[node].sort(key=lambda hp: hp[1])
    return grouped


MAX_BANDS = 16      # the prio qdisc's limit (TCQ_PRIO_BANDS)


def setup_commands(iface, port_delays, loss_pct=None):
    """tc commands for one node. `port_delays` is a list of (port, delay_ms).

    Ports sharing a delay share a netem band. Bands 1-3 stay with the default
    priomap, so the first netem band is 4, which keeps the single-delay output
    identical to what the October runs applied."""
    delays = sorted({round(float(d), 3) for _, d in port_delays})
    n_bands = 3 + len(delays)
    if n_bands > MAX_BANDS:
        raise ValueError("{} distinct delays on one node; prio allows at most {}".format(
            len(delays), MAX_BANDS - 3))
    band_of = {d: BAND + i for i, d in enumerate(delays)}
    cmds = [
        # Idempotent: drop whatever is there before rebuilding.
        "sudo tc qdisc del dev {} root".format(iface),
        "sudo tc qdisc add dev {} root handle 1: prio bands {}".format(iface, n_bands),
    ]
    for d in delays:
        netem = "delay {:g}ms".format(d)
        if loss_pct:
            netem += " loss {}%".format(loss_pct)
        cmds.append("sudo tc qdisc add dev {} parent 1:{:x} handle {:x}0: netem {}".format(
            iface, band_of[d], band_of[d], netem))
    for port, d in sorted(port_delays):
        cmds.append(
            "sudo tc filter add dev {} protocol ip parent 1:0 prio {} u32 "
            "match ip sport {} 0xffff flowid 1:{:x}".format(
                iface, PRIO, port, band_of[round(float(d), 3)]))
    return cmds


def scenario_delays(args):
    """host -> delay_ms for the hosts a scenario degrades (degradation.scenario)."""
    import degradation
    import topology_features
    hosts, primaries, _ = topology_features.generate_scaled_topology(
        n_hosts=args.n_hosts, n_primaries=args.n_primaries, seed=0)
    _, _, delays, profiles = degradation.scenario(
        hosts, primaries, args.n_degraded, selection=args.degraded_selection,
        model=args.degradation_model, seed=args.scale_seed)
    return {h: round(s * 1000.0, 3) for h, s in delays.items()}, profiles


def run(node, command, apply_it):
    if not apply_it:
        print("    {}".format(command))
        return 0
    done = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", node, command],
        capture_output=True, text=True, timeout=60)
    # `tc qdisc del` fails when there is nothing to delete, which is the normal
    # first-run case and not an error worth stopping for.
    if done.returncode != 0 and "qdisc del" not in command:
        sys.stderr.write("  {}: FAILED: {}\n".format(node, done.stderr.strip()))
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hosts", nargs="*", default=[],
                    help="host ids to degrade (e.g. w81 w82 ...), all by --delay-ms")
    ap.add_argument("--scenario", action="store_true",
                    help="degrade the hosts degradation.scenario() picks, each by its "
                         "own modelled delay; use the same scenario flags as the run")
    ap.add_argument("--n-hosts", type=int)
    ap.add_argument("--n-primaries", type=int)
    ap.add_argument("--n-degraded", type=int)
    ap.add_argument("--degradation-model", choices=["binary", "realistic"], default="binary")
    ap.add_argument("--degraded-selection", choices=["prefix", "spread", "random"], default="spread")
    ap.add_argument("--scale-seed", type=int, default=0)
    ap.add_argument("--delay-ms", type=float, default=50.0,
                    help="one-way added delay; the binary degradation model's 50ms")
    ap.add_argument("--loss-pct", type=float, default=None,
                    help="optional packet loss, for an 'unhealthy' host")
    ap.add_argument("--iface", required=True, help="the node's testbed NIC")
    ap.add_argument("--apply", action="store_true", help="run the commands")
    ap.add_argument("--clear", action="store_true", help="remove all shaping")
    ap.add_argument("--show", action="store_true",
                    help="print the live qdisc and filter counters")
    args = ap.parse_args(argv)

    manifest = load(args.manifest)
    all_nodes = sorted({spec["node"] for spec in manifest["hosts"].values()})

    if args.show:
        for node in all_nodes:
            print("=== {} ===".format(node))
            subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", node,
                            "tc -s qdisc show dev {}; tc -s filter show dev {}".format(
                                args.iface, args.iface)])
        return 0

    if args.clear:
        failures = 0
        for node in all_nodes:
            failures += run(node, "sudo tc qdisc del dev {} root".format(args.iface),
                            True)
        print("cleared shaping on {} nodes".format(len(all_nodes)))
        return 0

    if args.scenario:
        if None in (args.n_hosts, args.n_primaries, args.n_degraded):
            ap.error("--scenario needs --n-hosts, --n-primaries and --n-degraded")
        delay_of, profiles = scenario_delays(args)
        print("scenario: {} model, {} selection, scale seed {}".format(
            args.degradation_model, args.degraded_selection, args.scale_seed))
        for h in sorted(delay_of, key=lambda x: int(x[1:])):
            print("  {:<6} {:<20} severity {:.2f}  delay {:.3f}ms".format(
                h, profiles[h]["kind"], profiles[h]["severity"], delay_of[h]))
    elif args.hosts:
        delay_of = {h: args.delay_ms for h in args.hosts}
    else:
        ap.error("--hosts or --scenario is required unless --clear or --show")

    grouped = by_node(manifest, sorted(delay_of))
    print("degrading {} hosts on {} node(s){}\n".format(
        len(delay_of), len(grouped),
        "" if not args.loss_pct else " with {}% loss".format(args.loss_pct)))

    failures = 0
    for node in sorted(grouped):
        pairs = grouped[node]
        print("  {}: {}".format(
            node, ", ".join("{}:{}={:g}ms".format(h, p, delay_of[h]) for h, p in pairs)))
        port_delays = [(p, delay_of[h]) for h, p in pairs]
        for command in setup_commands(args.iface, port_delays, args.loss_pct):
            failures += run(node, command, args.apply)

    untouched = [n for n in all_nodes if n not in grouped]
    if untouched:
        print("\n  not shaped (no degraded host): {}".format(", ".join(untouched)))

    if not args.apply:
        print("\n(dry run -- nothing was changed; pass --apply to run these)")
    elif failures:
        print("\n{} command(s) failed".format(failures))
        return 1
    else:
        print("\napplied. verify with --show, and record it in the paper as "
              "tc netem on the server's source port.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
