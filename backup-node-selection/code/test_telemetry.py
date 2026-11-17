"""Checks for the testbed additions: per-host tc delays and the selectors' telemetry.

Run with:  python test_telemetry.py
Needs no network and no root: tc commands are only generated, never run.
"""

import apply_degradation
import degradation
import telemetry
import topology_features

OCTOBER_128 = {"w1", "w2", "w27", "w34", "w41", "w49", "w56", "w63", "w70", "w78",
               "w85", "w92", "w106", "w113", "w120", "w128"}


def scenario(n_hosts, n_primaries, n_degraded, model):
    hosts, primaries, graph = topology_features.generate_scaled_topology(
        n_hosts=n_hosts, n_primaries=n_primaries, seed=0)
    degraded, overrides, delays, profiles = degradation.scenario(
        hosts, primaries, n_degraded, selection="spread", model=model, seed=0)
    return hosts, primaries, graph, degraded, overrides, delays


def test_scenario_reproduces_october_hosts():
    """The shared scenario must degrade the hosts the October runs degraded."""
    _, _, _, degraded, _, _ = scenario(128, 80, 16, "binary")
    assert set(degraded) == OCTOBER_128, degraded
    print("ok   scenario picks the October 128-host degraded set")


def test_single_delay_commands_unchanged():
    """One delay must produce the exact commands the October runs applied."""
    cmds = apply_degradation.setup_commands("eth1", [(19000, 50), (19001, 50)])
    assert cmds == [
        "sudo tc qdisc del dev eth1 root",
        "sudo tc qdisc add dev eth1 root handle 1: prio bands 4",
        "sudo tc qdisc add dev eth1 parent 1:4 handle 40: netem delay 50ms",
        "sudo tc filter add dev eth1 protocol ip parent 1:0 prio 4 u32 "
        "match ip sport 19000 0xffff flowid 1:4",
        "sudo tc filter add dev eth1 protocol ip parent 1:0 prio 4 u32 "
        "match ip sport 19001 0xffff flowid 1:4",
    ], cmds
    print("ok   single-delay tc commands identical to the original form")


def test_per_host_delays_get_their_own_band():
    cmds = apply_degradation.setup_commands("eth1", [(19000, 9.645), (19001, 2.183)])
    assert "prio bands 5" in cmds[1]
    assert any("parent 1:4 handle 40: netem delay 2.183ms" in c for c in cmds)
    assert any("parent 1:5 handle 50: netem delay 9.645ms" in c for c in cmds)
    assert any("sport 19000 0xffff flowid 1:5" in c for c in cmds)
    assert any("sport 19001 0xffff flowid 1:4" in c for c in cmds)
    # thirteen distinct delays fit (16 bands); fourteen do not
    apply_degradation.setup_commands("eth1", [(19000 + i, i + 1) for i in range(13)])
    try:
        apply_degradation.setup_commands("eth1", [(19000 + i, i + 1) for i in range(14)])
    except ValueError:
        pass
    else:
        raise AssertionError("14 distinct delays should exceed prio's 16 bands")
    print("ok   per-host delays map to separate netem bands, limit enforced")


def test_tc_delays_match_what_selectors_are_told():
    """The realistic model's delay for a host is what tc imposes on it."""
    _, _, _, degraded, overrides, delays = scenario(128, 80, 16, "realistic")
    for h in degraded:
        assert abs(delays[h] * 1000.0 - overrides[h]["observed_delay_ms"]) < 1e-3, h
    print("ok   tc delays equal the model's reported delays")


def test_probe_excess():
    rtt = {"a": 1.0, "b": 1.2, "c": 0.9, "d": 31.0}
    ex = telemetry.excess_over_baseline(rtt)
    assert ex["d"] > 29.8 and ex["c"] == 0.0 and ex["a"] < 0.2, ex
    calls = []

    def call(ep):
        calls.append(ep)
        if ep == "dead":
            raise ConnectionRefusedError
    out = telemetry.probe_rtt({"x": "x", "dead": "dead"}, call, samples=3)
    assert set(out) == {"x"} and len(calls) == 6, (out, calls)
    print("ok   probe keeps answering hosts and measures excess over the median")


def test_report_is_deterministic_and_stale_hosts_look_healthy():
    hosts, primaries, graph, degraded, overrides, _ = scenario(32, 20, 4, "binary")
    healthy = {h: v for h, v in overrides.items() if h not in degraded}
    ctx = topology_features.build_feature_context(
        hosts=hosts, seed=3, graph=graph, overrides=overrides)
    excess = {h: (100.0 if h in degraded else 0.0) for h in hosts}
    r1, stale1 = telemetry.report(ctx, degraded, 3, hosts, graph, healthy,
                                  probed_excess=excess, rtt_noise_ms=2.0,
                                  feature_noise=0.05, stale_frac=0.5)
    r2, stale2 = telemetry.report(ctx, degraded, 3, hosts, graph, healthy,
                                  probed_excess=excess, rtt_noise_ms=2.0,
                                  feature_noise=0.05, stale_frac=0.5)
    assert stale1 == stale2 and len(stale1) == 2
    for h in hosts:
        assert r1.resources[h].health == r2.resources[h].health
        assert r1.resources[h].observed_delay_ms == r2.resources[h].observed_delay_ms
    for h in stale1:
        assert r1.resources[h].health > 0.3, (h, r1.resources[h].health)
        assert r1.resources[h].observed_delay_ms < 10.0
    fresh = [h for h in degraded if h not in stale1]
    for h in fresh:
        assert r1.resources[h].observed_delay_ms > 90.0
    # the input context is left untouched
    assert all(ctx.resources[h].health == 0.05 for h in degraded)
    print("ok   telemetry report is reproducible, stale hosts read healthy")


if __name__ == "__main__":
    test_scenario_reproduces_october_hosts()
    test_single_delay_commands_unchanged()
    test_per_host_delays_get_their_own_band()
    test_tc_delays_match_what_selectors_are_told()
    test_probe_excess()
    test_report_is_deterministic_and_stale_hosts_look_healthy()
    print("all telemetry checks passed")
