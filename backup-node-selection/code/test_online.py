"""Correctness tests for the online model, the DP optimum, and the proven bounds.

The DP is the yardstick every competitive ratio in the paper is measured against,
so it is checked against exhaustive enumeration rather than trusted.
"""

import itertools
import random

import online_model as om
import online_selector as osel


def brute_force_single(inst, primary):
    """Exhaustive minimum over all k^T trajectories for one primary. Only tractable
    for tiny instances, which is the point: it validates the DP on those."""
    cands = inst.candidates(primary)
    best = None
    for path in itertools.product(cands, repeat=inst.horizon):
        cost = 0.0
        for t, b in enumerate(path):
            cost += inst.service_cost(t, primary, b)
            if t > 0 and path[t - 1] != b:
                cost += inst.migration_cost
        if best is None or cost < best:
            best = cost
    return best


def test_dp_matches_brute_force():
    for seed in range(6):
        inst = om.build_instance(n_hosts=4, n_primaries=1, horizon=7, seed=seed,
                                 migration_cost=random.Random(seed).uniform(0.5, 6.0))
        p = inst.primaries[0]
        dp_cost, path = osel.offline_optimal_single(inst, p)
        bf_cost = brute_force_single(inst, p)
        assert abs(dp_cost - bf_cost) < 1e-6, (seed, dp_cost, bf_cost)
        # The returned path must actually achieve the returned cost.
        replay = sum(inst.service_cost(t, p, b) for t, b in enumerate(path))
        replay += sum(inst.migration_cost for t in range(1, len(path))
                      if path[t] != path[t - 1])
        assert abs(replay - dp_cost) < 1e-6, (seed, replay, dp_cost)
    print("ok   DP optimum matches exhaustive enumeration (6 instances)")


def test_joint_optimum_is_feasible_and_bounded():
    """Coordinate descent must return a real schedule whose cost it reports
    correctly, and must sit at or above the decoupled lower bound."""
    for seed in range(4):
        inst = om.build_instance(n_hosts=8, n_primaries=4, horizon=40, seed=seed)
        cost, schedule, lower = osel.offline_optimal(inst)
        assert abs(inst.total_cost(schedule) - cost) < 1e-6
        assert lower <= cost + 1e-6, (lower, cost)
        assert len(schedule) == inst.horizon
    print("ok   joint optimum is feasible and >= the certified lower bound")


def test_migration_budget_proposition():
    """Migration-budget proposition: hysteresis(alpha) spends at most a 1/alpha fraction of its
    service cost on migrations. Checked against an adversarially wide sweep."""
    worst = 0.0
    for alpha in (0.5, 1.0, 2.0, 4.0):
        for seed in range(12):
            for M in (0.5, 2.0, 8.0):
                inst = om.build_instance(n_hosts=10, n_primaries=5, horizon=120,
                                         seed=seed, migration_cost=M,
                                         params=om.ChurnParams(p_degrade=0.08,
                                                               p_recover=0.10,
                                                               p_fail=0.01))
                d = osel.decompose(inst, osel.policy_hysteresis(inst, alpha=alpha))
                ratio = d["migration"] / max(d["service"] + d["congestion"], 1e-9)
                worst = max(worst, ratio * alpha)
                assert ratio <= 1.0 / alpha + 1e-6, (alpha, seed, M, ratio)
    print("ok   migration budget holds; tightest observed is {:.3f} of the 1.000 bound"
          .format(worst))


def adversarial_alternating(eps, horizon=100, M=3.0):
    """The instance from the proof of Theorem 1.

    One primary, two candidate backups, no congestion. The cheaper host alternates
    every round, with the two costs separated by only `eps`. Chasing the snapshot
    optimum therefore migrates every round and pays M per round, while simply
    staying put pays at most eps per round.
    """
    inst = om.build_instance(n_hosts=3, n_primaries=1, horizon=horizon,
                             migration_cost=M, seed=0)
    inst.congestion = 0.0
    p = inst.primaries[0]
    a, b = inst.candidates(p)[:2]
    third = inst.candidates(p)[2] if len(inst.candidates(p)) > 2 else None
    cache = {}
    for t in range(horizon):
        cache[(t, p, a)] = 0.0 if t % 2 == 0 else eps
        cache[(t, p, b)] = eps if t % 2 == 0 else 0.0
        if third is not None:
            cache[(t, p, third)] = 10.0      # never attractive
    inst._service_cache = cache
    return inst


def test_snapshot_is_unbounded():
    """Theorem 1: snapshot-chasing has unbounded competitive ratio.

    As eps shrinks the optimum cost falls to zero while the snapshot policy keeps
    paying M per round, so the ratio grows without bound -- independent of horizon.
    """
    ratios = []
    for eps in (1.0, 0.1, 0.01, 0.001):
        inst = adversarial_alternating(eps)
        opt, _, _ = osel.offline_optimal(inst)
        snap = osel.decompose(inst, osel.policy_snapshot(inst))
        ratios.append(snap["total"] / opt)
    assert all(x < y for x, y in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] > 100, ratios
    print("ok   Theorem 1: snapshot/OPT grows without bound as eps -> 0: "
          + " -> ".join("{:.1f}".format(r) for r in ratios))


def test_hysteresis_survives_the_adversary():
    """The instance that defeats snapshot-chasing must not defeat hysteresis.

    As eps shrinks, the regret threshold is reached more and more slowly, so the
    policy migrates less and less and converges to the optimal do-nothing schedule.
    The contrast with Theorem 1 is the point: on the same instances where snapshot
    diverges, hysteresis stays bounded and in the limit is exactly optimal.
    """
    ratios = []
    for eps in (1.0, 0.1, 0.01, 0.001):
        inst = adversarial_alternating(eps)
        opt, _, _ = osel.offline_optimal(inst)
        d = osel.decompose(inst, osel.policy_hysteresis(inst, alpha=1.0))
        ratios.append(d["total"] / opt)
    assert all(r <= 2.5 for r in ratios), ratios
    assert all(x >= y - 1e-9 for x, y in zip(ratios, ratios[1:])), ratios
    assert abs(ratios[-1] - 1.0) < 1e-6, ratios
    print("ok   hysteresis stays bounded where snapshot diverges: "
          + " -> ".join("{:.2f}x".format(r) for r in ratios))


def test_trace_is_reproducible():
    a = om.build_instance(n_hosts=8, n_primaries=4, horizon=30, seed=11)
    b = om.build_instance(n_hosts=8, n_primaries=4, horizon=30, seed=11)
    assert [s["w1"].health for s in a.trace] == [s["w1"].health for s in b.trace]
    print("ok   churn traces are reproducible from the seed")


if __name__ == "__main__":
    test_trace_is_reproducible()
    test_dp_matches_brute_force()
    test_joint_optimum_is_feasible_and_bounded()
    test_migration_budget_proposition()
    test_snapshot_is_unbounded()
    test_hysteresis_survives_the_adversary()
    print("\nall online-model tests passed")
