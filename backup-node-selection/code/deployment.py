"""
Where the backup servers live, and how the client reaches them.

Why this exists
---------------
The RPC harness identified a backup by a port number on 127.0.0.1 and started
every server with multiprocessing.Process on the measuring machine. That couples
three things that should be separate: which host a backup *is* in the topology,
which process serves it, and which address the client dials. While everything ran
on one box the coupling was invisible, and it is also why the 128-host latency
results had to be withdrawn: 128 single-threaded servers and the client contended
for the same cores, so the numbers measured CPU scheduling rather than placement.

This module introduces the missing indirection. A backup is an `Endpoint`
(host id, IP, port). A `Deployment` maps the backups a mapping needs onto
endpoints. `LocalDeployment` reproduces the original behaviour exactly, so
previously published runs stay reproducible and serve as the regression check.
`RemoteDeployment` reads a manifest of servers already running elsewhere, which
is what a multi-machine testbed needs.

Nothing here assumes the remote servers are on a testbed in particular: a
manifest pointing at 127.0.0.1 exercises the remote code path on one machine,
which is how test_deployment.py covers it without a second host.
"""

import json
import multiprocessing
import os
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    """One backup server's identity and address.

    `host_id` is the topology-level name ("w7") and is what mappings, scoring and
    results are keyed by. `ip`/`port` are where to dial. Keeping them distinct is
    the whole point of this module: on one machine many host ids share an IP and
    differ by port, and on a testbed they differ by IP and can share a port.
    """
    host_id: str
    ip: str
    port: int

    def address(self):
        return "{}:{}".format(self.ip, self.port)

    def url(self):
        return "http://{}:{}".format(self.ip, self.port)


class Deployment:
    """Base class. A deployment knows how to produce endpoints for a set of
    backups, wait until they answer, and release whatever it started."""

    #: Set by subclasses. True when this deployment owns the server processes and
    #: must tear them down; False when they are managed outside the harness.
    owns_servers = False

    def endpoints_for(self, unique_backups, slow_hosts=None, slow_delay_s=0.05):
        raise NotImplementedError

    def wait_ready(self, endpoints, transport):
        """Block until every endpoint answers, or raise."""
        if transport == "grpc":
            import grpc_transport
            for endpoint in endpoints.values():
                grpc_transport.wait_until_ready(endpoint.port, host=endpoint.ip)
        else:
            # SimpleXMLRPCServer offers no readiness signal, so probe the socket.
            _wait_for_tcp(endpoints.values())

    def teardown(self):
        pass


def _wait_for_tcp(endpoints, timeout_s=15.0):
    """Wait until a TCP connect succeeds on each endpoint.

    The original harness slept 0.3 s and hoped. That is adequate for a handful of
    local processes and not for dozens of servers starting across several
    machines, where a slow binder would otherwise be measured as a connection
    failure in the first burst.
    """
    import socket
    deadline = time.time() + timeout_s
    pending = list(endpoints)
    while pending and time.time() < deadline:
        still = []
        for endpoint in pending:
            try:
                with socket.create_connection((endpoint.ip, endpoint.port), 0.5):
                    pass
            except OSError:
                still.append(endpoint)
        pending = still
        if pending:
            time.sleep(0.05)
    if pending:
        raise RuntimeError("backup servers did not come up: {}".format(
            ", ".join(e.address() for e in pending)))


# ----------------------------------------------------------------------------
# Local: one process per backup on the measuring machine
# ----------------------------------------------------------------------------

class LocalDeployment(Deployment):
    """The original arrangement: a server process per backup on 127.0.0.1,
    assigned consecutive ports from `base_port`.

    Retained unchanged in behaviour so that the published single-machine results
    remain reproducible, and so that a run through the refactored code path can be
    diffed against a run through the old one.
    """

    owns_servers = True

    def __init__(self, base_port, transport="xmlrpc", grpc_workers=None,
                 serve_xmlrpc=None, serve_grpc=None):
        self.base_port = base_port
        self.transport = transport
        self.grpc_workers = grpc_workers
        # Injected by the experiment module, which owns the server entry points.
        self._serve_xmlrpc = serve_xmlrpc
        self._serve_grpc = serve_grpc
        self.processes = []

    def endpoints_for(self, unique_backups, slow_hosts=None, slow_delay_s=0.05):
        slow_hosts = slow_hosts or {}
        endpoints = {}
        for i, host_id in enumerate(sorted(unique_backups)):
            port = self.base_port + i
            delay = (slow_hosts.get(host_id, 0.0) if isinstance(slow_hosts, dict)
                     else (slow_delay_s if host_id in slow_hosts else 0.0))
            if self.transport == "grpc":
                target = self._serve_grpc
                args = ("127.0.0.1", port, delay, self.grpc_workers)
            else:
                target = self._serve_xmlrpc
                args = ("127.0.0.1", port, delay)
            proc = multiprocessing.Process(target=target, args=args, daemon=True)
            proc.start()
            self.processes.append(proc)
            endpoints[host_id] = Endpoint(host_id, "127.0.0.1", port)
        return endpoints

    def teardown(self):
        for proc in self.processes:
            proc.terminate()
        for proc in self.processes:
            proc.join()
        self.processes = []
        # Let the OS release sockets before the next method reuses the ports.
        time.sleep(0.2)


# ----------------------------------------------------------------------------
# Remote: servers already running, described by a manifest
# ----------------------------------------------------------------------------

MANIFEST_VERSION = 1


class RemoteDeployment(Deployment):
    """Backups served by processes this harness did not start.

    The manifest is a JSON document mapping topology host ids to addresses::

        {
          "version": 1,
          "transport": "grpc",
          "hosts": {
            "w1": {"ip": "10.10.1.2", "port": 19000, "node": "node-0", "core": 0},
            "w2": {"ip": "10.10.1.2", "port": 19001, "node": "node-0", "core": 1}
          }
        }

    `node` and `core` are recorded for reporting, not used for dialing: a run that
    claims to have given each server a dedicated core should be able to show which
    core that was.

    A degraded host's extra delay cannot be applied from here, because the server
    is already running. On a real testbed that is the right division anyway:
    degradation should be imposed physically on the node (tc netem for a congested
    link, a cgroup cpu.max for a starved one) rather than by a sleep inside the
    server, and `provision_remote.py` writes those mechanisms into the manifest
    so a run records how degradation was produced. `endpoints_for` therefore
    rejects a slow_hosts argument it cannot honour, instead of silently ignoring
    it and reporting a degradation that was never applied.
    """

    owns_servers = False

    def __init__(self, manifest):
        self.manifest = manifest
        self.hosts = manifest["hosts"]

    @classmethod
    def from_file(cls, path):
        with open(path) as handle:
            manifest = json.load(handle)
        version = manifest.get("version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                "manifest version {} is not supported (expected {})".format(
                    version, MANIFEST_VERSION))
        if not manifest.get("hosts"):
            raise ValueError("manifest lists no hosts: {}".format(path))
        return cls(manifest)

    def endpoints_for(self, unique_backups, slow_hosts=None, slow_delay_s=0.05):
        missing = sorted(set(unique_backups) - set(self.hosts))
        if missing:
            raise KeyError(
                "manifest has no server for: {}. A remote run must provision every "
                "backup-eligible host, since the mapping is chosen at run time and "
                "any host may be selected.".format(", ".join(missing)))
        if slow_hosts:
            offenders = sorted(slow_hosts) if not isinstance(slow_hosts, dict) \
                else sorted(h for h, d in slow_hosts.items() if d)
            if offenders:
                raise ValueError(
                    "RemoteDeployment cannot inject a per-call delay into servers it "
                    "did not start ({}). Impose degradation on the node instead "
                    "(see provision_remote.py --degrade) so the run records how it "
                    "was produced.".format(", ".join(offenders)))
        return {host_id: Endpoint(host_id,
                                  self.hosts[host_id]["ip"],
                                  int(self.hosts[host_id]["port"]))
                for host_id in sorted(unique_backups)}

    def describe(self):
        """One line per distinct node, for the run log."""
        by_node = {}
        for host_id, spec in sorted(self.hosts.items()):
            by_node.setdefault(spec.get("node", spec["ip"]), []).append(host_id)
        return ["{}: {} backups ({})".format(node, len(ids),
                                             ", ".join(ids[:4]) + ("..." if len(ids) > 4 else ""))
                for node, ids in sorted(by_node.items())]


def save_manifest(path, hosts, transport="xmlrpc", extra=None):
    """Write a manifest. `hosts` maps host id -> dict with at least ip and port."""
    manifest = {"version": MANIFEST_VERSION, "transport": transport,
                "hosts": hosts}
    if extra:
        manifest.update(extra)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------

MANIFEST_ENV = "IDEFT_MANIFEST"


def make_deployment(base_port, transport="xmlrpc", grpc_workers=None,
                    manifest=None, serve_xmlrpc=None, serve_grpc=None):
    """Pick a deployment.

    An explicit `manifest` path wins; otherwise the IDEFT_MANIFEST environment
    variable is honoured, so an existing experiment command runs unchanged against
    a testbed. With neither, the local single-machine arrangement is used, which
    is what every published result so far was produced with.
    """
    manifest = manifest or os.environ.get(MANIFEST_ENV)
    if manifest:
        return RemoteDeployment.from_file(manifest)
    return LocalDeployment(base_port, transport=transport,
                           grpc_workers=grpc_workers,
                           serve_xmlrpc=serve_xmlrpc, serve_grpc=serve_grpc)
