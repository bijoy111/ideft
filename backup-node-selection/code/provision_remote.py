"""
Place backup servers across testbed nodes, pin them to cores, write the manifest.

The single-machine harness could start its servers itself. A multi-node run
cannot: the servers live on machines the client does not control, they must be
pinned so that the "each server had a dedicated core" claim is checkable, and the
client needs an address book. This script produces all three.

What it does
------------
1. Assigns each topology host id to a (node, port, core) slot, packing
   `--per-node` servers onto each node in turn.
2. Launches them over SSH under `taskset`, so the pinning is visible in the
   remote process table rather than asserted in prose.
3. Writes the manifest that RemoteDeployment reads.
4. Optionally records how each degraded host was degraded, so a run states the
   mechanism rather than a number we chose.

Why pinning is not optional
---------------------------
The 128-host latency results in this paper had to be withdrawn because 128
single-threaded servers and the client contended for one machine's cores, so the
measurement reflected CPU scheduling rather than backup placement. Simply using
more machines does not answer that objection; showing that each server owned a
core does. `--per-node` should therefore not exceed the node's core count less a
margin, and `--verify` reports the affinity each server actually got.

Degradation
-----------
`--degrade w7=link_congested:20ms` records that w7's degradation is to be imposed
with `tc netem delay 20ms` on its node, and prints the command. This is
deliberately physical: the weight-sensitivity conclusions in this paper currently
rest on a degradation model we wrote, and reproducing them against real tc and
cgroup mechanisms is what turns them from a property of the model into a property
of degradation. The script prints these commands rather than running them, since
they need root on the node and are worth reading before they run.

Usage
-----
    # dry run: see the plan and the commands, touch nothing
    python3 provision_remote.py --nodes 10.10.1.2 10.10.1.3 --hosts 16 \
        --per-node 8 --dry-run

    # launch for real over SSH, then check
    python3 provision_remote.py --nodes node0 node1 --hosts 16 --per-node 8 \
        --user myuser --remote-dir ~/ideft/backup-node-selection/code \
        --manifest ../findings/manifest.json
    python3 provision_remote.py --manifest ../findings/manifest.json --verify

    # exercise the whole remote path on this machine, no SSH, no testbed
    python3 provision_remote.py --local --hosts 8 --per-node 8 \
        --manifest /tmp/manifest_local.json

    # stop everything
    python3 provision_remote.py --manifest ../findings/manifest.json --teardown
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

import deployment

DEFAULT_BASE_PORT = 19000


def natural(host_id):
    """Sort w2 before w10. Plain string order puts w10 first, which makes a
    printed placement plan hard to check by eye."""
    digits = "".join(c for c in host_id if c.isdigit())
    return ("".join(c for c in host_id if not c.isdigit()),
            int(digits) if digits else 0)

# Physical mechanisms for each degradation kind in degradation.py. These are
# printed for the operator to apply on the node; the script does not run them,
# since each needs root and each is worth reading first.
DEGRADE_COMMANDS = {
    "link_congested": "sudo tc qdisc replace dev {iface} root netem delay {arg}",
    "unhealthy": "sudo tc qdisc replace dev {iface} root netem loss {arg}",
    "cpu_starved": "sudo systemd-run --scope -p CPUQuota={arg} --uid={user} <server cmd>",
    "capacity_exhausted": "sudo sysctl -w net.netfilter.nf_conntrack_max={arg}",
    "storage_pressured": "sudo systemd-run --scope -p IOReadBandwidthMax='{arg}' <server cmd>",
}


def plan(hosts, nodes, per_node, base_port, reserve_cores=2):
    """Assign host ids to (node, port, core) slots.

    Cores are numbered from `reserve_cores` upward, leaving the low cores for the
    OS and the SSH session so that a server is not pinned onto a core the node is
    already using for something else.
    """
    if per_node < 1:
        raise ValueError("--per-node must be at least 1")
    capacity = len(nodes) * per_node
    if len(hosts) > capacity:
        raise ValueError(
            "{} hosts need more than {} slots ({} nodes x {} per node). Add nodes "
            "or raise --per-node, but do not raise it past the node's core count: "
            "oversubscribing cores is the defect this arrangement exists to "
            "remove.".format(len(hosts), capacity, len(nodes), per_node))

    slots = {}
    for i, host_id in enumerate(hosts):
        node = nodes[i // per_node]
        index_on_node = i % per_node
        slots[host_id] = {
            "ip": node,
            "node": node,
            "port": base_port + index_on_node,
            "core": reserve_cores + index_on_node,
        }
    return slots


def server_command(host_id, spec, transport, grpc_workers, remote_dir,
                   python="python3", backlog=None):
    """The command to run on the node for one backup."""
    inner = [python, "backup_server.py",
             "--host-id", host_id,
             "--bind", spec["ip"],
             "--port", str(spec["port"]),
             "--transport", transport]
    if grpc_workers is not None:
        inner += ["--grpc-workers", str(grpc_workers)]
    if backlog is not None:
        inner += ["--backlog", str(backlog)]
    pinned = ["taskset", "-c", str(spec["core"])] + inner
    logfile = "/tmp/ideft_backup_{}.log".format(host_id)
    body = "cd {} || exit 1; nohup {} > {} 2>&1 < /dev/null & echo $!".format(
        shlex.quote(remote_dir), " ".join(shlex.quote(x) for x in pinned),
        shlex.quote(logfile))
    return body


def ssh(node, command, user=None, dry_run=False):
    target = "{}@{}".format(user, node) if user else node
    argv = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            target, command]
    if dry_run:
        print("  " + " ".join(shlex.quote(a) for a in argv))
        return None
    done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError("ssh to {} failed: {}".format(node, done.stderr.strip()))
    return done.stdout.strip()


def launch_local(slots, transport, grpc_workers):
    """Start the servers on this machine as ordinary subprocesses.

    This is not a substitute for a testbed -- it reintroduces exactly the core
    contention that makes single-machine latency unreliable -- but it drives the
    full manifest and RemoteDeployment path without needing a second host, which
    is what makes that path testable.
    """
    pids = {}
    for host_id, spec in sorted(slots.items(), key=lambda kv: natural(kv[0])):
        argv = [sys.executable, "backup_server.py", "--host-id", host_id,
                "--bind", spec["ip"], "--port", str(spec["port"]),
                "--transport", transport]
        if grpc_workers is not None:
            argv += ["--grpc-workers", str(grpc_workers)]
        log = open("/tmp/ideft_backup_{}.log".format(host_id), "w")
        proc = subprocess.Popen(argv, stdout=log, stderr=log)
        pids[host_id] = proc.pid
    return pids


def verify(manifest_path):
    """Check every server in the manifest answers, and report its affinity."""
    deploy = deployment.RemoteDeployment.from_file(manifest_path)
    hosts = sorted(deploy.hosts, key=natural)
    endpoints = deploy.endpoints_for(hosts)
    transport = deploy.manifest.get("transport", "xmlrpc")
    print("verifying {} servers ({})".format(len(endpoints), transport))
    for line in deploy.describe():
        print("  " + line)
    deploy.wait_ready(endpoints, transport)
    print("  all {} endpoints answered".format(len(endpoints)))

    unpinned = [h for h, s in deploy.hosts.items() if s.get("core") is None]
    if unpinned:
        print("  WARNING: {} servers have no recorded core. A latency result at "
              "scale needs each server pinned, or it cannot be distinguished "
              "from CPU contention.".format(len(unpinned)))
    cores = {}
    for host_id, spec in deploy.hosts.items():
        key = (spec.get("node"), spec.get("core"))
        cores.setdefault(key, []).append(host_id)
    shared = {k: v for k, v in cores.items() if len(v) > 1 and k[1] is not None}
    if shared:
        print("  WARNING: cores shared by more than one server:")
        for (node, core), ids in sorted(shared.items()):
            print("    {} core {}: {}".format(node, core, ", ".join(sorted(ids))))
    else:
        print("  every server has its own core on its node")
    return 0


def teardown(manifest_path, user=None, dry_run=False):
    deploy = deployment.RemoteDeployment.from_file(manifest_path)
    if deploy.manifest.get("launched") == "local":
        subprocess.run(["pkill", "-f", "backup_serve[r].py --host-id"], check=False)
        print("stopped local backup servers")
        return 0
    nodes = sorted({s["node"] for s in deploy.hosts.values()})
    for node in nodes:
        ssh(node, "pkill -f 'backup_serve[r].py --host-id' || true",
            user=user, dry_run=dry_run)
    print("stop signal sent to {} nodes".format(len(nodes)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", nargs="+", default=[],
                    help="node hostnames or testbed IPs to place servers on")
    ap.add_argument("--hosts", type=int, default=None,
                    help="number of topology hosts (w1..wN); all are provisioned, "
                         "since the mapping is chosen at run time")
    ap.add_argument("--host-ids", nargs="+", default=None,
                    help="explicit host ids instead of w1..wN")
    ap.add_argument("--per-node", type=int, default=8,
                    help="servers per node; keep below the node's core count")
    ap.add_argument("--reserve-cores", type=int, default=2,
                    help="low cores left for the OS and the login session")
    ap.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    ap.add_argument("--transport", choices=["xmlrpc", "grpc"], default="xmlrpc")
    ap.add_argument("--grpc-workers", type=int, default=None)
    ap.add_argument("--user", default=None, help="SSH user")
    ap.add_argument("--remote-dir", default="~/ideft/backup-node-selection/code")
    ap.add_argument("--manifest", default="../findings/manifest.json")
    ap.add_argument("--degrade", nargs="*", default=[],
                    metavar="HOST=KIND:ARG",
                    help="record a physical degradation, e.g. w7=link_congested:20ms")
    ap.add_argument("--backlog", type=int, default=None,
                    help="TCP listen backlog for each XML-RPC backup. Unset "
                         "reproduces DEFT as shipped (SimpleXMLRPCServer's default "
                         "of 5), whose accept queue overflows under concurrent "
                         "replication and produces a 1s RTO plateau at p99. "
                         "Recorded in the manifest either way.")
    ap.add_argument("--iface", default="eth1", help="testbed NIC for tc commands")
    ap.add_argument("--local", action="store_true",
                    help="launch on this machine instead of over SSH, to exercise "
                         "the manifest and remote code path without a testbed")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands without running them")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--teardown", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        return verify(args.manifest)
    if args.teardown:
        return teardown(args.manifest, user=args.user, dry_run=args.dry_run)

    host_ids = args.host_ids or ["w{}".format(i) for i in range(1, (args.hosts or 0) + 1)]
    if not host_ids:
        ap.error("give --hosts N or --host-ids")
    nodes = args.nodes or (["127.0.0.1"] if args.local else [])
    if not nodes:
        ap.error("give --nodes, or --local to place everything on this machine")

    slots = plan(host_ids, nodes, args.per_node, args.base_port,
                 reserve_cores=args.reserve_cores)

    print("plan: {} hosts over {} node(s), {} per node".format(
        len(host_ids), len(nodes), args.per_node))
    by_node = {}
    for host_id, spec in sorted(slots.items(), key=lambda kv: natural(kv[0])):
        by_node.setdefault(spec["node"], []).append((host_id, spec))
    for node, entries in sorted(by_node.items()):
        print("  {}: {}".format(node, ", ".join(
            "{}->:{} core {}".format(h, s["port"], s["core"])
            for h, s in sorted(entries, key=lambda kv: natural(kv[0])))))

    degradations = {}
    for item in args.degrade:
        host, _, rest = item.partition("=")
        kind, _, arg = rest.partition(":")
        if kind not in DEGRADE_COMMANDS:
            ap.error("unknown degradation kind {!r}; choose from {}".format(
                kind, ", ".join(sorted(DEGRADE_COMMANDS))))
        if host not in slots:
            ap.error("--degrade names {} which is not in the plan".format(host))
        command = DEGRADE_COMMANDS[kind].format(
            iface=args.iface, arg=arg or "", user=args.user or "$USER")
        degradations[host] = {"kind": kind, "arg": arg, "command": command,
                              "node": slots[host]["node"]}

    if degradations:
        print("\nphysical degradation -- run these on the named node, as root:")
        for host, spec in sorted(degradations.items()):
            print("  # {} ({} on {})".format(host, spec["kind"], spec["node"]))
            print("  " + spec["command"])

    if args.dry_run:
        print("\nserver commands (not run):")
        for host_id, spec in sorted(slots.items(), key=lambda kv: natural(kv[0])):
            print("  # {} on {}".format(host_id, spec["node"]))
            ssh(spec["node"],
                server_command(host_id, spec, args.transport, args.grpc_workers,
                               args.remote_dir, backlog=args.backlog),
                user=args.user, dry_run=True)
        print("\nmanifest would be written to {}".format(args.manifest))
        return 0

    launched = "ssh"
    if args.local:
        pids = launch_local(slots, args.transport, args.grpc_workers)
        for host_id, pid in pids.items():
            slots[host_id]["pid"] = pid
            # Nothing is pinned in local mode, and recording a core we did not
            # set would make the manifest claim an isolation it does not have.
            slots[host_id]["core"] = None
        launched = "local"
        print("\nstarted {} local servers".format(len(pids)))
    else:
        print("\nlaunching over SSH:")
        for host_id, spec in sorted(slots.items(), key=lambda kv: natural(kv[0])):
            out = ssh(spec["node"],
                      server_command(host_id, spec, args.transport,
                                     args.grpc_workers, args.remote_dir,
                                     backlog=args.backlog),
                      user=args.user)
            if out:
                spec["pid"] = int(out.splitlines()[-1])
            print("  {} on {}:{} core {} pid {}".format(
                host_id, spec["node"], spec["port"], spec["core"],
                spec.get("pid", "?")))

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    deployment.save_manifest(
        args.manifest, slots, transport=args.transport,
        extra={"launched": launched,
               "per_node": args.per_node,
               "grpc_workers": args.grpc_workers,
               "backlog": args.backlog,
               "degradations": degradations})
    print("\nwrote {}".format(args.manifest))
    print("run the experiment against it with:")
    print("  IDEFT_MANIFEST={} python3 rpc_contention_experiment.py ...".format(
        args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
