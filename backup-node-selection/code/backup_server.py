"""
One backup server, as its own process, bound to a real address.

The single-machine harness starts its servers with multiprocessing.Process and
binds them to 127.0.0.1. On a multi-machine testbed the servers have to exist
before the client runs, on nodes the client does not control, bound to routable
addresses, and ideally pinned to a known core. This is that entry point.

It serves exactly the same handlers as the in-process path in
rpc_contention_experiment.py, so a remote run and a local run exercise the same
server code and remain comparable.

Usage (on a backup node):
    python3 backup_server.py --host-id w7 --bind 10.10.1.3 --port 19006
    python3 backup_server.py --host-id w7 --bind 10.10.1.3 --port 19006 \
        --transport grpc --grpc-workers 8

Pinning is the caller's job, so that the pinning is visible in the process
table and can be reported:
    taskset -c 6 python3 backup_server.py --host-id w7 --bind 10.10.1.3 --port 19006

On --delay
----------
`--delay` exists for parity with the local harness, where a degraded host's extra
per-call cost is a sleep inside the server. On a testbed that is the wrong
mechanism: a congested link should be imposed with `tc netem`, a starved CPU with
a cgroup, so that the degradation is physical and the finding is a property of
real degradation rather than of our model of it. The flag therefore warns when
used, rather than being silently convenient.
"""

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-id", required=True,
                    help="topology-level name of the host this serves (e.g. w7)")
    ap.add_argument("--bind", default="0.0.0.0",
                    help="address to bind (default 0.0.0.0; use the node's testbed IP)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--transport", choices=["xmlrpc", "grpc"], default="xmlrpc")
    ap.add_argument("--grpc-workers", type=int, default=None,
                    help="gRPC server thread pool size; 1 reproduces XML-RPC's "
                         "strict serialization")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="artificial per-call delay in seconds (prefer tc/cgroups; "
                         "see module docstring)")
    args = ap.parse_args(argv)

    if args.delay:
        sys.stderr.write(
            "backup_server: --delay {}s is a simulated degradation. On a testbed, "
            "impose it physically (tc netem / cgroup cpu.max) so the result "
            "describes real degradation.\n".format(args.delay))

    # Imported here so that --help works without the experiment module's
    # dependencies (numpy, the DEFT package) being importable.
    import rpc_contention_experiment as rpc

    sys.stderr.write("backup_server: host_id={} transport={} bind={}:{} pid={} "
                     "affinity={}\n".format(
                         args.host_id, args.transport, args.bind, args.port,
                         os.getpid(),
                         sorted(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity") else "n/a"))
    sys.stderr.flush()

    if args.transport == "grpc":
        rpc._serve_backup_grpc(args.bind, args.port, args.delay, args.grpc_workers)
    else:
        rpc._serve_backup(args.bind, args.port, args.delay)


if __name__ == "__main__":
    main()
