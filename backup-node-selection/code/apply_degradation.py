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

Usage
-----
    # see the commands, change nothing
    python3 apply_degradation.py --manifest ../findings/m.json \\
        --hosts w81 w82 --delay-ms 50 --iface enp65s0f0np0

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


def setup_commands(iface, ports, delay_ms, loss_pct=None):
    netem = "delay {}ms".format(delay_ms)
    if loss_pct:
        netem += " loss {}%".format(loss_pct)
    cmds = [
        # Idempotent: drop whatever is there before rebuilding.
        "sudo tc qdisc del dev {} root".format(iface),
        "sudo tc qdisc add dev {} root handle 1: prio bands {}".format(iface, BAND),
        "sudo tc qdisc add dev {} parent 1:{} handle {}0: netem {}".format(
            iface, BAND, BAND, netem),
    ]
    for port in ports:
        cmds.append(
            "sudo tc filter add dev {} protocol ip parent 1:0 prio {} u32 "
            "match ip sport {} 0xffff flowid 1:{}".format(iface, PRIO, port, BAND))
    return cmds


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
                    help="host ids to degrade (e.g. w81 w82 ...)")
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

    if not args.hosts:
        ap.error("--hosts is required unless --clear or --show")

    grouped = by_node(manifest, args.hosts)
    print("degrading {} hosts on {} node(s) with {}ms{}\n".format(
        len(args.hosts), len(grouped), args.delay_ms,
        "" if not args.loss_pct else " + {}% loss".format(args.loss_pct)))

    failures = 0
    for node in sorted(grouped):
        pairs = grouped[node]
        print("  {}: {}".format(
            node, ", ".join("{}:{}".format(h, p) for h, p in pairs)))
        ports = [p for _, p in pairs]
        for command in setup_commands(args.iface, ports, args.delay_ms, args.loss_pct):
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
