"""
Deterministic multi-criteria backup-node selector.

Replaces the static hardcoded primary_to_secondary.json with a feasibility- and
load-aware greedy assignment, using topology_features.py for hop-count/latency and
configurable synthetic health/capacity. Writes the same flat {primary: backup} JSON
format config.py already reads, so no downstream changes are needed.

Usage:
    python backup_selector.py --mode heuristic --seed 42 --out primary_to_secondary.json
    python backup_selector.py --mode baseline --out primary_to_secondary.json
"""

import argparse
import json
import random as random_module

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
import topology_features
from topology_features import INFEASIBLE_HOSTS

DEFAULT_WEIGHTS = {"health": 0.35, "capacity": 0.35, "latency": 0.20, "hops": 0.10}

LOAD_UNIT = 20
HEALTH_DECAY = 0.95

# The mapping DEFT ships with (config.py's primary_to_secondary.json), kept here so the
# original baseline can be regenerated on demand.
BASELINE_MAPPING = {
    "h1": "h4",
    "h2": "h1",
    "h3": "h6",
    "h5": "h4",
    "h7": "h4",
}


def feasible_backups(primary, hosts):
    return [h for h in hosts if h != primary and h not in INFEASIBLE_HOSTS]


def _normalize(values):
    """Min-max normalize a dict of host->value to [0, 1]. Constant inputs -> all 0."""
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {host: 0.0 for host in values}
    return {host: (value - lo) / (hi - lo) for host, value in values.items()}


def score_candidates(primary, candidates, ctx, weights):
    latencies = {c: ctx.distances[primary][c]["latency_ms"] for c in candidates}
    hops = {c: ctx.distances[primary][c]["hops"] for c in candidates}
    norm_latency = _normalize(latencies)
    norm_hops = _normalize(hops)

    scores = {}
    for c in candidates:
        state = ctx.resources[c]
        scores[c] = (
            weights["health"] * state.health
            + weights["capacity"] * (state.capacity / state.max_capacity)
            - weights["latency"] * norm_latency[c]
            - weights["hops"] * norm_hops[c]
        )
    return scores


def _apply_load_penalty(state):
    state.capacity = max(state.capacity - LOAD_UNIT, 0)
    state.health *= HEALTH_DECAY


def select_backups(primaries=None, hosts=None, ctx=None, weights=None):
    primaries = primaries if primaries is not None else config.PRIMARIES
    hosts = hosts if hosts is not None else config.HOSTS
    ctx = ctx if ctx is not None else topology_features.build_feature_context(hosts)
    weights = weights if weights is not None else DEFAULT_WEIGHTS

    mapping = {}
    for primary in primaries:
        candidates = feasible_backups(primary, hosts)
        scores = score_candidates(primary, candidates, ctx, weights)
        best = max(candidates, key=lambda c: scores[c])
        mapping[primary] = best
        _apply_load_penalty(ctx.resources[best])

    return mapping


def get_baseline_mapping():
    return dict(BASELINE_MAPPING)


def concentrated_mapping(primaries=None, hosts=None):
    """Assigns every primary to the same single backup wherever feasible: a maximally
    naive policy with no load awareness. Used as a scale-independent analog of the DEFT
    static mapping (which concentrates 3 of 5 primaries onto h4 in the 9-host topology),
    generalizing that same failure mode to any topology size."""
    primaries = primaries if primaries is not None else config.PRIMARIES
    hosts = hosts if hosts is not None else config.HOSTS

    mapping = {}
    chosen_backup = None
    for primary in primaries:
        candidates = feasible_backups(primary, hosts)
        if chosen_backup is None or chosen_backup not in candidates:
            chosen_backup = candidates[0]
        mapping[primary] = chosen_backup
    return mapping


def random_mapping(primaries=None, hosts=None, seed=None):
    """Uniform-random feasible backup per primary. A sanity-floor baseline: any
    non-random method should at least beat this."""
    primaries = primaries if primaries is not None else config.PRIMARIES
    hosts = hosts if hosts is not None else config.HOSTS
    rng = random_module.Random(seed)

    mapping = {}
    for primary in primaries:
        candidates = feasible_backups(primary, hosts)
        mapping[primary] = rng.choice(candidates)
    return mapping


def round_robin_mapping(primaries=None, hosts=None):
    """Cycles through the sorted feasible-backup pool in order, skipping a candidate
    only when it would equal the primary itself. A common, cost-free load-spreading
    baseline that doesn't need any node features at all."""
    primaries = primaries if primaries is not None else config.PRIMARIES
    hosts = hosts if hosts is not None else config.HOSTS
    all_backups = sorted(h for h in hosts if h not in INFEASIBLE_HOSTS)

    mapping = {}
    idx = 0
    for primary in primaries:
        for _ in range(len(all_backups)):
            candidate = all_backups[idx % len(all_backups)]
            idx += 1
            if candidate != primary:
                mapping[primary] = candidate
                break
    return mapping


def optimal_mapping(primaries=None, hosts=None, ctx=None, weights=None, capacity_per_backup=None):
    """Globally optimal assignment under a hard per-backup capacity constraint, solved
    exactly via the Hungarian algorithm (scipy linear_sum_assignment). Serves as an upper
    bound for the greedy/RL methods. Capacity slots per backup default to a ceil-balanced
    split (e.g. 5 primaries / 2 backups -> 3 slots each), which forces load-spreading into
    the optimum instead of everyone piling onto the single highest-scoring backup.
    """
    primaries = primaries if primaries is not None else config.PRIMARIES
    hosts = hosts if hosts is not None else config.HOSTS
    ctx = ctx if ctx is not None else topology_features.build_feature_context(hosts)
    weights = weights if weights is not None else DEFAULT_WEIGHTS

    all_backups = sorted(h for h in hosts if h not in INFEASIBLE_HOSTS)
    n_primaries = len(primaries)
    n_backups = len(all_backups)

    if capacity_per_backup is None:
        capacity_per_backup = -(-n_primaries // n_backups)  # ceil division

    columns = []
    for backup in all_backups:
        columns.extend([backup] * capacity_per_backup)

    cost = np.full((n_primaries, len(columns)), fill_value=1e6)
    for i, primary in enumerate(primaries):
        candidates = feasible_backups(primary, hosts)
        scores = score_candidates(primary, candidates, ctx, weights)
        for j, backup in enumerate(columns):
            if backup in scores:
                cost[i, j] = -scores[backup]

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[primaries[i]] = columns[j]
    return mapping


def write_mapping(mapping, path="primary_to_secondary.json"):
    with open(path, "w") as f:
        json.dump(mapping, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["heuristic", "baseline", "concentrated", "random", "round_robin", "optimal"],
        default="heuristic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="primary_to_secondary.json")
    args = parser.parse_args()

    if args.mode == "baseline":
        mapping = get_baseline_mapping()
    elif args.mode == "concentrated":
        mapping = concentrated_mapping()
    elif args.mode == "random":
        mapping = random_mapping(seed=args.seed)
    elif args.mode == "round_robin":
        mapping = round_robin_mapping()
    elif args.mode == "optimal":
        ctx = topology_features.build_feature_context(seed=args.seed)
        mapping = optimal_mapping(ctx=ctx)
    else:
        ctx = topology_features.build_feature_context(seed=args.seed)
        mapping = select_backups(ctx=ctx)

    write_mapping(mapping, args.out)
    print(f"[{args.mode}] wrote {args.out}: {mapping}")


if __name__ == "__main__":
    main()
