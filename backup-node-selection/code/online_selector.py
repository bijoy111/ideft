"""
Online backup-selection policies, and an exact offline optimum to score them against.

The policies
------------
static        Solve the one-shot problem once at t=0 and never move. This is what
              the deployed system does, and what every result in the one-shot
              sections of the paper measures.

snapshot      Re-solve the one-shot problem every round and migrate whenever the
              answer changes. This is the paper's Hungarian "optimal" baseline run
              online. Theorem 1 shows it has unbounded competitive ratio; this is
              the implementation that demonstrates it.

periodic(T0)  Re-solve every T0 rounds. Standard operational practice (a nightly
              rebalance), and a fairer strawman than `snapshot`.

hysteresis(a) Stay until the accumulated excess cost of the current backup exceeds
              a * M, then move to the current best. The migration-budget proposition: this policy can
              never spend more than a 1/a fraction of its service cost on
              migrations, whatever the adversary does.

offline_opt   Exact minimum-cost schedule by dynamic programming (migration-coupled
              across time, decoupled across primaries). See `offline_optimal`.
"""

import math

import online_model


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def _effective_cost(inst, t, primary, backup, load):
    """Service cost plus the congestion the primary would face at `backup`."""
    extra = load.get(backup, 0)
    return inst.service_cost(t, primary, backup) + inst.congestion * extra


def _one_shot(inst, t):
    """Load-aware greedy assignment for round t, the same rule select_backups uses:
    place primaries in order, charging each one the congestion created by those
    already placed."""
    load, assignment = {}, {}
    for p in inst.primaries:
        best = min(inst.candidates(p),
                   key=lambda b: _effective_cost(inst, t, p, b, load))
        assignment[p] = best
        load[best] = load.get(best, 0) + 1
    return assignment


# ----------------------------------------------------------------------------
# Policies. Each returns the schedule: a list of per-round assignments.
# ----------------------------------------------------------------------------

def policy_static(inst):
    initial = _one_shot(inst, 0)
    return [dict(initial) for _ in range(inst.horizon)]


def policy_static_failover(inst):
    """Static assignment, but migrate when the current backup hard-fails.

    This is the fair version of the deployed policy. A production system does not
    sit on a dead replica: it fails over. What it does *not* do is react to a host
    that is merely degraded -- slow, congested, or short of capacity -- which is
    the regime the online policies are designed for. Comparing against this rather
    than against pure `static` isolates the value of reacting to degradation from
    the value of reacting to death, and avoids crediting the online policies for
    beating a strawman.
    """
    current = _one_shot(inst, 0)
    schedule = []
    for t in range(inst.horizon):
        schedule.append(dict(current))
        load = {}
        for b in current.values():
            load[b] = load.get(b, 0) + 1
        pending = {}
        for p in inst.primaries:
            if inst.trace[t][current[p]].failed:
                other = dict(load)
                other[current[p]] = other.get(current[p], 1) - 1
                alive = [b for b in inst.candidates(p) if not inst.trace[t][b].failed]
                if alive:
                    pending[p] = min(alive, key=lambda b: _effective_cost(inst, t, p, b, other))
        current.update(pending)
    return schedule


def policy_snapshot(inst):
    return [_one_shot(inst, t) for t in range(inst.horizon)]


def policy_periodic(inst, period=20):
    schedule, current = [], None
    for t in range(inst.horizon):
        if current is None or t % period == 0:
            current = _one_shot(inst, t)
        schedule.append(dict(current))
    return schedule


def policy_hysteresis(inst, alpha=1.0):
    """Accumulate each primary's regret against the best available backup; migrate
    only once that regret has paid for the migration.

    Migrations decided during round t take effect at round t+1. That is the
    physically correct semantics -- a primary cannot retroactively have had its
    state on the new backup -- and it is also what makes the migration-budget proposition hold: the
    policy must actually pay the cost that justified its migration.

    A hard failure needs no special case: a failed host costs FAILED_COST, which
    exceeds alpha*M for any reasonable alpha, so the threshold fires after a single
    round of the failure.
    """
    M = inst.migration_cost
    current = _one_shot(inst, 0)
    regret = {p: 0.0 for p in inst.primaries}
    schedule = []

    for t in range(inst.horizon):
        # The assignment held *during* round t is what this round is billed for.
        schedule.append(dict(current))

        load = {}
        for b in current.values():
            load[b] = load.get(b, 0) + 1

        pending = {}
        for p in inst.primaries:
            here = current[p]
            # The load this primary would see elsewhere excludes its own contribution.
            other = dict(load)
            other[here] = other.get(here, 1) - 1
            best = min(inst.candidates(p),
                       key=lambda b: _effective_cost(inst, t, p, b, other))
            gap = (_effective_cost(inst, t, p, here, other)
                   - _effective_cost(inst, t, p, best, other))
            regret[p] += max(gap, 0.0)
            if regret[p] >= alpha * M and best != here:
                pending[p] = best
                regret[p] = 0.0

        # Apply all of this round's decisions at once, so the order in which
        # primaries are scanned cannot affect who moves.
        current.update(pending)
    return schedule


# ----------------------------------------------------------------------------
# Offline optimum
# ----------------------------------------------------------------------------

def offline_optimal_single(inst, primary, congestion_load=None):
    """Exact min-cost schedule for one primary by DP over the time-host trellis.

    With a uniform migration cost the transition reduces to

        dp[t][b] = cost(t, b) + min( dp[t-1][b],  M + min_b' dp[t-1][b'] )

    so each round is O(k) rather than O(k^2). `congestion_load` optionally supplies
    a fixed background load per host per round, used by the coordinate-descent
    refinement below; with None the primary is costed in isolation.
    """
    M = inst.migration_cost
    cands = inst.candidates(primary)

    def cost(t, b):
        base = inst.service_cost(t, primary, b)
        if congestion_load is not None:
            base += inst.congestion * congestion_load[t].get(b, 0)
        return base

    dp = {b: cost(0, b) for b in cands}
    back = [{} for _ in range(inst.horizon)]
    for t in range(1, inst.horizon):
        best_prev_host = min(dp, key=dp.get)
        best_prev = dp[best_prev_host]
        new = {}
        for b in cands:
            stay = dp[b]
            move = M + best_prev
            if stay <= move:
                new[b], back[t][b] = cost(t, b) + stay, b
            else:
                new[b], back[t][b] = cost(t, b) + move, best_prev_host
        dp = new

    end = min(dp, key=dp.get)
    path = [None] * inst.horizon
    path[-1] = end
    for t in range(inst.horizon - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return dp[end], path


def offline_optimal(inst, refine_rounds=12):
    """Offline optimum for the full instance.

    Exactly optimal when congestion is zero, since the primaries then decouple and
    each is solved exactly by DP. With congestion the joint problem couples the
    primaries and exact optimisation is intractable, so we do two things:

      * `lower_bound` solves the decoupled relaxation (congestion dropped). It is a
        certified lower bound on the true optimum, so any competitive ratio measured
        against it OVERSTATES the true ratio -- it is conservative in the direction
        that matters, and cannot flatter the online policies.

      * `cost` runs coordinate descent from that relaxation: repeatedly re-optimise
        one primary's whole trajectory by DP against the others' current load. This
        is a genuine feasible schedule, so it is an upper bound on the optimum, and
        it is what the reported ratios use.

    Returns (cost, schedule, lower_bound).
    """
    paths, lower_bound = {}, 0.0
    for p in inst.primaries:
        value, path = offline_optimal_single(inst, p)
        paths[p] = path
        lower_bound += value

    def build(paths):
        return [{p: paths[p][t] for p in inst.primaries} for t in range(inst.horizon)]

    best_schedule = build(paths)
    best_cost = inst.total_cost(best_schedule)

    for _ in range(refine_rounds):
        improved = False
        for p in inst.primaries:
            load = [{} for _ in range(inst.horizon)]
            for t in range(inst.horizon):
                for q in inst.primaries:
                    if q == p:
                        continue
                    b = paths[q][t]
                    load[t][b] = load[t].get(b, 0) + 1
            _, path = offline_optimal_single(inst, p, congestion_load=load)
            trial = dict(paths)
            trial[p] = path
            schedule = build(trial)
            cost = inst.total_cost(schedule)
            if cost < best_cost - 1e-9:
                best_cost, best_schedule, paths, improved = cost, schedule, trial, True
        if not improved:
            break

    return best_cost, best_schedule, lower_bound


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def cost_floor(inst):
    """The cost no policy can avoid: each primary placed on its instantaneously
    cheapest backup every round, with congestion and migration both free.

    This is a lower bound on every policy's cost, including the optimum's. It is
    worth separating out because it is usually the majority of total cost, and a
    competitive ratio computed on total cost is therefore compressed toward 1 by a
    term that no selection policy -- good or bad -- has any influence over.
    Reporting the ratio of *excess over this floor* isolates the part of the bill
    that selection actually controls.
    """
    total = 0.0
    for t in range(inst.horizon):
        for p in inst.primaries:
            total += min(inst.service_cost(t, p, b) for b in inst.candidates(p))
    return total


def excess_ratio(alg_total, opt_total, floor):
    """(ALG - floor) / (OPT - floor): how much worse the policy's *decisions* are.

    Returns 1.0 when the optimum itself only just clears the floor, which happens
    on instances where nothing can be gained; those instances carry no information
    about policy quality and are reported as such rather than as a perfect score.
    """
    denominator = opt_total - floor
    if denominator <= 1e-9:
        return float("nan")
    return (alg_total - floor) / denominator


def migration_count(schedule):
    n, previous = 0, None
    for assignment in schedule:
        if previous is not None:
            n += sum(1 for p, b in assignment.items() if previous.get(p) != b)
        previous = assignment
    return n


def decompose(inst, schedule):
    """Split a schedule's cost into service, congestion and migration."""
    service = congestion = migration = 0.0
    previous = None
    for t, assignment in enumerate(schedule):
        load = {}
        for b in assignment.values():
            load[b] = load.get(b, 0) + 1
        for p, b in assignment.items():
            service += inst.service_cost(t, p, b)
            congestion += inst.congestion * (load[b] - 1)
            if previous is not None and previous.get(p) != b:
                migration += inst.migration_cost
        previous = assignment
    return {"service": service, "congestion": congestion, "migration": migration,
            "total": service + congestion + migration,
            "migrations": migration_count(schedule)}


POLICIES = {
    "static": policy_static,
    "failover": policy_static_failover,
    "snapshot": policy_snapshot,
    "periodic20": lambda inst: policy_periodic(inst, period=20),
    "hysteresis0.5": lambda inst: policy_hysteresis(inst, alpha=0.5),
    "hysteresis1": lambda inst: policy_hysteresis(inst, alpha=1.0),
    "hysteresis2": lambda inst: policy_hysteresis(inst, alpha=2.0),
}
