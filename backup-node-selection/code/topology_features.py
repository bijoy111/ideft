"""
Real-topology distances + configurable synthetic resource state for backup selection.

build_topology_graph() derives its edges from config.SWITCH_HOST_TO_PORT directly, so it
always matches the actual Mininet topology built in test.py's MultiSwitchTopo.

Note: within the h1..h7 pool every host is exactly 2 hops from every other host (via
switch-2), so hop-count/latency do not discriminate within that pool; capacity and health
are what drive load-spreading decisions among candidate backups.
"""

import random
from dataclasses import dataclass

import networkx as nx

import config

INFEASIBLE_HOSTS = {"client", "stamper"}


def build_topology_graph():
    graph = nx.Graph()
    for switch, port_map in config.SWITCH_HOST_TO_PORT.items():
        for neighbor in port_map:
            graph.add_edge(switch, neighbor)
    return graph


def generate_scaled_topology(n_hosts=20, n_primaries=15, seed=0):
    """Synthetic single-hub star topology, same shape as the real 2-switch topology in
    config.py/test.py (every worker host is 2 hops from every other via a shared hub),
    parametrized to an arbitrary size for offline evaluation (RPC benchmark, RL training)
    decoupled from the fixed 9-host Mininet topology. Returns (hosts, primaries, graph).

    The first n_primaries hosts are treated as primaries; the remaining
    (n_hosts - n_primaries) are backup-eligible only, mirroring config.py's PRIMARIES
    being a subset of HOSTS."""
    if n_primaries >= n_hosts:
        raise ValueError("n_primaries must be less than n_hosts (need spare backup-eligible hosts)")

    hosts = ["w{}".format(i) for i in range(1, n_hosts + 1)]
    primaries = hosts[:n_primaries]

    graph = nx.Graph()
    hub = "hub"
    for host in hosts:
        graph.add_edge(hub, host)

    return hosts, primaries, graph


def generate_multihub_topology(n_hosts=32, n_primaries=20, n_hubs=3, seed=0):
    """Synthetic multi-hub topology: n_hubs hubs connected in a line, with hosts distributed
    round-robin across the hubs. Unlike the single-hub star (generate_scaled_topology), where
    every host is exactly two hops from every other, host-to-host hop-count and latency now
    vary (same-hub pairs are 2 hops apart, different-hub pairs more), giving genuine topology
    and latency diversity for testing whether the findings hold beyond a uniform star.
    Returns (hosts, primaries, graph)."""
    if n_primaries >= n_hosts:
        raise ValueError("n_primaries must be less than n_hosts (need spare backup-eligible hosts)")

    hosts = ["w{}".format(i) for i in range(1, n_hosts + 1)]
    primaries = hosts[:n_primaries]

    graph = nx.Graph()
    hubs = ["hub{}".format(k) for k in range(n_hubs)]
    for a, b in zip(hubs, hubs[1:]):          # connect hubs in a line
        graph.add_edge(a, b)
    for i, host in enumerate(hosts):          # distribute hosts across hubs
        graph.add_edge(hubs[i % n_hubs], host)

    return hosts, primaries, graph


def compute_hop_and_latency(graph, hosts, latency_per_hop_ms=1.5):
    """Returns {host_a: {host_b: {"hops": int, "latency_ms": float}}} for host_a != host_b."""
    distances = {}
    for host in hosts:
        hop_lengths = nx.single_source_shortest_path_length(graph, host)
        distances[host] = {
            other: {
                "hops": hop_lengths[other],
                "latency_ms": hop_lengths[other] * latency_per_hop_ms,
            }
            for other in hosts
            if other != host and other in hop_lengths
        }
    return distances


@dataclass
class HostResourceState:
    host: str
    cpu_availability: float
    storage_availability: float
    health: float
    capacity: float
    max_capacity: float
    # Added round-trip time attributable to the host and its network path, as an
    # active RTT probe would observe it. Topology-derived hop latency cannot see
    # this: a host behind a congested link is the same number of hops away as any
    # other, so in a uniform-hop topology the hop and latency features carry no
    # signal at all. This field is what makes such a host detectable.
    observed_delay_ms: float = 0.0


def generate_synthetic_resources(hosts, seed=None, overrides=None):
    """Draws synthetic health/capacity/CPU/storage from fixed value ranges, seeded for
    reproducibility. Excludes client/stamper (never valid backups). `overrides` maps
    host -> partial dict of field overrides, for pinning specific experiment scenarios."""
    rng = random.Random(seed)
    overrides = overrides or {}

    resources = {}
    for host in hosts:
        if host in INFEASIBLE_HOSTS:
            continue

        max_capacity = rng.randint(50, 100)
        state = HostResourceState(
            host=host,
            cpu_availability=round(rng.uniform(0.4, 1.0), 2),
            storage_availability=round(rng.uniform(0.4, 1.0), 2),
            health=round(rng.uniform(0.5, 1.0), 2),
            capacity=max_capacity,
            max_capacity=max_capacity,
        )

        for field, value in overrides.get(host, {}).items():
            if field == "capacity_ratio":
                # Expressed as a fraction of this host's own drawn max_capacity, so
                # it can only be resolved here. Used by the realistic degradation
                # model, where both healthy baseline utilisation and capacity
                # exhaustion are relative quantities rather than absolute numbers.
                state.capacity = max(1.0, round(state.max_capacity * value, 2))
            elif not hasattr(state, field):
                raise AttributeError(
                    "unknown host resource field {!r} for host {}".format(field, host))
            else:
                setattr(state, field, value)

        resources[host] = state

    return resources


@dataclass
class FeatureContext:
    distances: dict
    resources: dict


def build_feature_context(hosts=None, latency_per_hop_ms=1.5, seed=42, overrides=None, graph=None):
    hosts = hosts if hosts is not None else config.HOSTS
    graph = graph if graph is not None else build_topology_graph()
    distances = compute_hop_and_latency(graph, hosts, latency_per_hop_ms=latency_per_hop_ms)
    resources = generate_synthetic_resources(hosts, seed=seed, overrides=overrides)
    return FeatureContext(distances=distances, resources=resources)
