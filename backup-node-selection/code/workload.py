"""
A flow-arrival workload that fills a FlowTable to a realistic occupancy.

What this is for
----------------
The migration cost of a primary's state depends almost entirely on how many flow
entries the table holds, and that number is a property of the traffic, not of the
selector. This module generates traffic with the features that determine it:
flows arrive at some rate, live for a heavy-tailed duration, emit packets while
alive, and are evicted once idle. Steady-state occupancy then follows Little's
law, occupancy sets the serialized size, and serialized size sets the migration
cost.

On the parameter values
-----------------------
The defaults below are a plausible mid-size enterprise edge: a few thousand new
flows per second, most of them short, a small fraction long-lived. They are
stated as explicit parameters rather than asserted as measured ground truth,
because they are not measured here. Anything that depends on them is reported as
a sweep, so the conclusion is a curve over plausible deployments rather than a
single number resting on a guess. `measure_migration_cost.py` sweeps arrival rate
and idle timeout for exactly this reason.

Replacing the generator with a packet trace is the obvious next step and needs
only `replay_trace` below to be pointed at a pcap; the rest of the pipeline is
unchanged. Public traces suitable for this include the MAWI archive and
CIC-IDS2017, neither of which requires an access application.
"""

import random
from dataclasses import dataclass

import flow_state as fs


@dataclass
class WorkloadParams:
    """Traffic shape. Durations are in seconds."""
    arrival_rate: float = 2000.0      # new flows per second
    # Flow lifetime is heavy-tailed: most connections are brief, a few persist.
    # A Pareto tail with alpha near 1.5 reproduces the mice/elephants split
    # without needing the full empirical distribution.
    lifetime_min: float = 0.05
    lifetime_alpha: float = 1.5
    lifetime_cap: float = 600.0       # truncate, or the mean does not exist
    packet_rate: float = 20.0         # packets per second while a flow is active
    mean_packet_bytes: float = 700.0
    idle_timeout: float = 30.0
    tcp_fraction: float = 0.85


def _pareto_lifetime(rng, params):
    """Truncated Pareto. Truncation matters: with alpha < 2 the untruncated
    distribution has infinite variance, and with alpha <= 1 no finite mean, so an
    untruncated draw would make steady-state occupancy undefined."""
    u = rng.random()
    value = params.lifetime_min / (u ** (1.0 / params.lifetime_alpha))
    return min(value, params.lifetime_cap)


def expected_occupancy(params):
    """Little's law prediction for steady-state table size.

    A flow occupies a table slot for its lifetime plus the idle timeout that must
    elapse before it is evicted, so the expected residency is E[lifetime] +
    idle_timeout and occupancy is arrival_rate times that. Used as a correctness
    check on the simulator rather than as a substitute for it.
    """
    a, lo, cap = params.lifetime_alpha, params.lifetime_min, params.lifetime_cap
    # E[min(X, cap)] for a Pareto(lo, a) variate with a > 1.
    if a <= 1.0:
        raise ValueError("lifetime_alpha must exceed 1 for a finite mean")
    mean_life = (a * lo / (a - 1.0)) * (1.0 - (lo / cap) ** (a - 1.0)) \
        + cap * (lo / cap) ** a
    return params.arrival_rate * (mean_life + params.idle_timeout)


class FlowWorkload:
    """Drives a FlowTable forward in fixed time steps.

    Each call to `step` advances the clock by `round_s`, admits new flows, emits
    packets for the flows that are active, expires the idle ones, and returns the
    replication delta that a primary would have to ship for that interval.
    """

    def __init__(self, params=None, seed=0, round_s=1.0, start_time=0.0):
        self.params = params or WorkloadParams()
        self.rng = random.Random(seed)
        self.round_s = round_s
        self.now = start_time
        self.table = fs.FlowTable()
        self.deadline = {}          # key -> time the flow stops sending
        self._next_src = 0x0A000000
        self._next_port = 1024

    def _new_key(self):
        self._next_src += 1
        self._next_port = 1024 + (self._next_port - 1024 + 1) % 64000
        proto = fs.TCP if self.rng.random() < self.params.tcp_fraction else fs.UDP
        return (self._next_src & 0xFFFFFFFF, 0x0A000002,
                self._next_port, 80, proto)

    def step(self):
        """Advance one round. Returns the delta a primary would replicate."""
        p = self.params
        self.now += self.round_s

        # Arrivals over this interval.
        n_new = self._poisson(p.arrival_rate * self.round_s)
        for _ in range(n_new):
            key = self._new_key()
            self.deadline[key] = self.now + _pareto_lifetime(self.rng, p)
            nat = (0xC0A80001, self.rng.randrange(1024, 65535))
            self.table.observe(key, int(p.mean_packet_bytes), self.now,
                               fs.TCP_NEW, nat)

        # Packets from flows that are still sending. Only flows that emit at
        # least one packet in this interval are touched, which is what keeps the
        # delta smaller than the table.
        expected = p.packet_rate * self.round_s
        for key, stop in list(self.deadline.items()):
            if stop < self.now:
                continue
            count = self._poisson(expected)
            if not count:
                continue
            self.table.observe(key, int(count * p.mean_packet_bytes), self.now,
                               fs.TCP_ESTABLISHED)

        self.table.expire(self.now, p.idle_timeout)
        for key in [k for k, stop in self.deadline.items()
                    if stop < self.now - p.idle_timeout]:
            del self.deadline[key]

        return self.table.take_delta()

    def _poisson(self, mean):
        """Knuth for small means, normal approximation above the point where
        Knuth's loop becomes the bottleneck."""
        if mean <= 0:
            return 0
        if mean > 30:
            return max(0, int(self.rng.gauss(mean, mean ** 0.5) + 0.5))
        target, count, product = 2.718281828459045 ** -mean, 0, 1.0
        while True:
            product *= self.rng.random()
            if product <= target:
                return count
            count += 1

    def warm(self, rounds):
        """Run until occupancy is in steady state, discarding the deltas."""
        for _ in range(rounds):
            self.step()
        return self.table


def replay_trace(path, table=None, limit=None):
    """Drive a FlowTable from a pcap instead of the generator.

    Left as a thin wrapper deliberately: everything downstream consumes a
    FlowTable, so swapping the generator for a real trace changes nothing else.
    Requires `dpkt`, which is not a dependency of the rest of this repository.
    """
    import dpkt                                    # noqa: F401  (optional)
    import socket

    table = table if table is not None else fs.FlowTable()
    with open(path, "rb") as handle:
        for i, (timestamp, buf) in enumerate(dpkt.pcap.Reader(handle)):
            if limit is not None and i >= limit:
                break
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue
            seg = ip.data
            key = (int.from_bytes(ip.src, "big"), int.from_bytes(ip.dst, "big"),
                   seg.sport, seg.dport, ip.p)
            table.observe(key, len(buf), timestamp,
                          fs.TCP_ESTABLISHED if ip.p == fs.TCP else None,
                          (0xC0A80001, seg.sport))
    return table
