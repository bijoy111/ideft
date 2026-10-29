"""
Online backup selection under churn: the instance model.

Why this module exists
----------------------
Every selector in backup_selector.py solves a *one-shot* problem: given a snapshot
of host state, choose an assignment. That is the right model only if host state is
static, which it is not. Hosts degrade and recover, primaries are scaled in and out,
and the assignment that was best at t=0 is not the assignment that is best at t=100.

Re-solving the snapshot problem at every round is the obvious fix and it is wrong,
because moving a primary's backup is not free: the new backup starts with no replica
of the primary's state, so a migration costs a full state transfer. A selector that
chases the instantaneous optimum pays that cost on every fluctuation.

This module generates the instance: a sequence of per-round host states produced by
a churn process, together with the cost model that charges for service quality,
congestion, and migration. online_selector.py consumes it.

Cost model
----------
At round t, primary p backed by host b pays

    c_t(p, b)  =  service cost (quality of b as a backup for p, in the same
                  weighted-feature units backup_selector.py already scores in)
               +  LAMBDA * (load_t(b) - 1)      congestion
               +  MIGRATION_COST if b differs from p's backup at round t-1

Service cost is the negated selector score shifted to be non-negative, so a
selector that minimises cost here is optimising exactly what the paper's heuristic
maximises. This keeps the online problem commensurable with the one-shot results
rather than introducing a second, unrelated objective.
"""

import math
import random
from dataclasses import dataclass, field

import degradation

# Weighted-feature cost uses the same weights as the published heuristic, so the
# online objective is the online version of the paper's objective, not a new one.
DEFAULT_WEIGHTS = {"health": 0.35, "capacity": 0.35, "latency": 0.20, "hops": 0.10}

# Congestion charged per additional primary sharing a backup. The one-shot selector
# expresses the same preference through its LOAD_UNIT capacity decrement; here it is
# an explicit per-round cost so that it enters the competitive analysis.
LAMBDA = 0.25

# Cost of moving one primary's backup, in the same units as one round of service
# cost: a migration re-replicates the primary's whole state to the new backup,
# where a round of service replicates only what changed during that round.
#
# This was originally a guessed 3.0, because DEFT's backup discards the state it
# is sent and there was nothing to measure. measure_migration_cost.py now derives
# it from a real connection-tracking table filled by a real workload, and the
# guess was low by a factor of three to five:
#
#     M_rounds  = (serialize + transfer + load of the full table)
#               / (serialize + transfer + load of one round's delta)
#               ~ 23 to 41 rounds, over a 16x range of flow arrival rates
#
# M_rounds is roughly 1/delta_fraction and therefore a property of traffic churn
# rather than of the wire format: a costlier encoding inflates the numerator and
# the denominator alike. The encoding does move the crossover in *seconds*, which
# is what measure_migration_cost.py reports.
#
# Converting to the cost units used here multiplies by the per-primary per-round
# service cost a well-behaved policy actually pays, measured at 0.334, giving a
# range of roughly 7.7 to 13.7. The sweeps are re-run at both ends, and the
# ordering of the policies is the same at each, so nothing below depends on where
# in the measured range one lands. DEFAULT_MIGRATION_COST is kept at the original
# guess so previously published sweeps stay reproducible.
DEFAULT_MIGRATION_COST = 3.0
MEASURED_MIGRATION_COST_LOW = 7.7
MEASURED_MIGRATION_COST_HIGH = 13.7
ROUND_SERVICE_COST = 0.334


@dataclass
class HostState:
    """Per-round observable state of one host."""
    health: float
    capacity_ratio: float
    observed_delay_ms: float
    failed: bool = False

    def quality(self):
        """Scalar in [0,1], higher is better. Used only for reporting."""
        if self.failed:
            return 0.0
        return 0.5 * self.health + 0.5 * self.capacity_ratio


@dataclass
class OnlineInstance:
    """A full problem instance: topology, churn trace, and cost accessors."""
    hosts: list
    primaries: list
    hop_latency: dict          # (p, b) -> topology latency in ms
    hop_count: dict            # (p, b) -> hops
    trace: list                # trace[t][host] -> HostState
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    migration_cost: float = DEFAULT_MIGRATION_COST
    congestion: float = LAMBDA
    _service_cache: dict = field(default_factory=dict, repr=False)

    @property
    def horizon(self):
        return len(self.trace)

    def candidates(self, primary):
        return [h for h in self.hosts if h != primary]

    def service_cost(self, t, primary, backup):
        """Cost of primary using backup at round t, excluding congestion and migration.

        Non-negative, and 0 only for a perfect host at zero distance. A failed host
        costs FAILED_COST, which is large enough that no finite-horizon policy will
        sit on one but finite so the DP stays well-defined."""
        key = (t, primary, backup)
        hit = self._service_cache.get(key)
        if hit is not None:
            return hit
        state = self.trace[t][backup]
        if state.failed:
            value = FAILED_COST
        else:
            w = self.weights
            # Latency is normalised by the instance's own scale so the term is
            # comparable across topologies, matching _normalize() in the one-shot
            # selector.
            lat = (self.hop_latency[(primary, backup)] + state.observed_delay_ms) / LATENCY_SCALE
            hops = self.hop_count[(primary, backup)] / HOP_SCALE
            score = (w["health"] * state.health
                     + w["capacity"] * state.capacity_ratio
                     - w["latency"] * min(lat, 1.0)
                     - w["hops"] * min(hops, 1.0))
            # Shift so cost is non-negative: score lies in [-(w_lat+w_hop), w_h+w_c].
            value = (w["health"] + w["capacity"]) - score
        self._service_cache[key] = value
        return value

    def round_cost(self, t, assignment, previous=None):
        """Total cost of one round under `assignment` (primary -> backup)."""
        load = {}
        for b in assignment.values():
            load[b] = load.get(b, 0) + 1
        total = 0.0
        for p, b in assignment.items():
            total += self.service_cost(t, p, b)
            total += self.congestion * (load[b] - 1)
            if previous is not None and previous.get(p) not in (None, b):
                total += self.migration_cost
        return total

    def total_cost(self, schedule):
        """Cost of a full schedule: a list of per-round assignments."""
        total, previous = 0.0, None
        for t, assignment in enumerate(schedule):
            total += self.round_cost(t, assignment, previous)
            previous = assignment
        return total


FAILED_COST = 50.0
LATENCY_SCALE = 60.0   # ms; a path at or beyond this is maximally penalised
HOP_SCALE = 8.0


# ----------------------------------------------------------------------------
# Churn process
# ----------------------------------------------------------------------------

@dataclass
class ChurnParams:
    """Two-state Markov chain per host, plus hard failures.

    p_degrade / p_recover set how often a healthy host becomes degraded and how
    long degradation lasts. p_fail / p_repair govern hard failures, which make a
    host unusable rather than merely worse. Expected degraded fraction at
    stationarity is p_degrade / (p_degrade + p_recover).
    """
    p_degrade: float = 0.02
    p_recover: float = 0.08
    p_fail: float = 0.002
    p_repair: float = 0.05
    # Within-state drift: even a host that does not change regime wanders, so the
    # instantaneous argmin moves without any regime change. This is what punishes
    # a selector that chases the snapshot optimum.
    drift: float = 0.02


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def generate_trace(hosts, horizon, params=None, seed=0, model="realistic"):
    """Produce trace[t][host] -> HostState under the churn process.

    Degraded hosts draw a kind and severity from degradation.py, so the failure
    modes here are the same ones the one-shot experiments use -- including
    link_congested, whose damage is visible only in observed_delay_ms.
    """
    params = params or ChurnParams()
    rng = random.Random(seed)
    kinds = sorted(degradation.KINDS)

    # Initial regime: everyone healthy, with the realistic model's baseline
    # utilisation so capacity is a measurement rather than a flag.
    regime = {h: "healthy" for h in hosts}
    kind_of = {h: kinds[i % len(kinds)] for i, h in enumerate(sorted(hosts))}
    severity = {h: 0.0 for h in hosts}
    base_health = {h: rng.uniform(*degradation.HEALTHY_HEALTH) for h in hosts}
    base_cap = {h: rng.uniform(*degradation.HEALTHY_CAPACITY_RATIO) for h in hosts}

    trace = []
    for _ in range(horizon):
        snapshot = {}
        for h in hosts:
            # --- regime transitions ---
            if regime[h] == "healthy":
                if rng.random() < params.p_fail:
                    regime[h] = "failed"
                elif rng.random() < params.p_degrade:
                    regime[h] = "degraded"
                    severity[h] = rng.betavariate(degradation.SEVERITY_ALPHA,
                                                  degradation.SEVERITY_BETA)
                    if model == "binary":
                        severity[h] = 1.0
                    kind_of[h] = rng.choice(kinds)
            elif regime[h] == "degraded":
                if rng.random() < params.p_fail:
                    regime[h] = "failed"
                elif rng.random() < params.p_recover:
                    regime[h] = "healthy"
                    severity[h] = 0.0
            else:  # failed
                if rng.random() < params.p_repair:
                    regime[h] = "healthy"
                    severity[h] = 0.0

            # --- within-regime drift ---
            base_health[h] = _clamp(base_health[h] + rng.gauss(0, params.drift),
                                    *degradation.HEALTHY_HEALTH)
            base_cap[h] = _clamp(base_cap[h] + rng.gauss(0, params.drift),
                                 *degradation.HEALTHY_CAPACITY_RATIO)

            # --- realise observable state ---
            if regime[h] == "failed":
                snapshot[h] = HostState(0.0, 0.0, 0.0, failed=True)
                continue
            health, cap, delay = base_health[h], base_cap[h], 0.0
            if regime[h] == "degraded":
                response = degradation.KINDS[kind_of[h]]
                s = severity[h]
                for fieldname, lo_hi in response.items():
                    value = degradation._lerp(lo_hi, s)
                    if fieldname == "delay_ms":
                        delay = value
                    elif fieldname == "health":
                        health = min(health, value)
                    elif fieldname in ("capacity_ratio", "cpu_availability",
                                       "storage_availability"):
                        cap = min(cap, value)
            snapshot[h] = HostState(health, cap, delay)
        trace.append(snapshot)
    return trace


def build_instance(n_hosts=16, n_primaries=10, horizon=200, seed=0,
                   params=None, migration_cost=DEFAULT_MIGRATION_COST,
                   model="realistic", topology="multihub", n_hubs=3):
    """Assemble a full OnlineInstance on a synthetic topology.

    Mirrors topology_features.generate_multihub_topology's shape (hosts spread over
    a line of hubs) so hop-count and latency genuinely vary, rather than the uniform
    star where every pair is two hops apart and topology terms cannot discriminate.
    """
    hosts = ["w{}".format(i) for i in range(1, n_hosts + 1)]
    primaries = hosts[:n_primaries]

    hub_of = {h: (i % n_hubs if topology == "multihub" else 0)
              for i, h in enumerate(hosts)}
    hop_latency, hop_count = {}, {}
    for p in hosts:
        for b in hosts:
            if p == b:
                continue
            hub_distance = abs(hub_of[p] - hub_of[b])
            hops = 2 + 2 * hub_distance
            hop_count[(p, b)] = hops
            # 0.5 ms per hop plus a small per-hub-crossing penalty, the same scale
            # the Mininet topology reports.
            hop_latency[(p, b)] = 0.5 * hops + 2.0 * hub_distance

    trace = generate_trace(hosts, horizon, params=params, seed=seed, model=model)
    return OnlineInstance(hosts=hosts, primaries=primaries,
                          hop_latency=hop_latency, hop_count=hop_count,
                          trace=trace, migration_cost=migration_cost)
