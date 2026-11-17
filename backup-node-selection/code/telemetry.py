"""What the selectors are told about each host, when it is not simply the truth.

Without this module a run hands the selectors the degradation model's own values:
a degraded host's features and added delay are reported exactly and at once, so a
selector that reads host state avoids it by construction. This module replaces
that with three departures from the truth, each optional:

1. **Probed RTT.** The latency feature becomes what a probe measured against the
   running server, not the delay the model assigned. Each backup is called a few
   times with the replication call itself, one call at a time so that the probe
   measures the host and not its own queue, and the median is kept. A host's
   *excess* is its median less the median over all probed hosts, clipped at zero,
   so a healthy host reads about 0 ms and a degraded one reads what tc imposed
   (twice the netem delay for a connection-per-call protocol).

2. **Noise.** Gaussian noise on the reported RTT excess (ms) and on the reported
   health and capacity ratio (absolute, clipped to [0.01, 1]). It stands for
   probe jitter and for gauges that are sampled rather than exact.

3. **Staleness.** A fraction of the degraded hosts is reported as it was before it
   degraded: its healthy feature draw and an RTT excess of zero. It stands for a
   host that degraded after the orchestrator last looked. The host is still slow
   on the testbed; only the report is old.

Noise and staleness are drawn per context seed from a generator seeded by it, so a
seed's report is the same for every method and every trial, and reproducible.
"""

import copy
import random
import statistics
import time

import topology_features


def probe_rtt(endpoints, call, samples=5):
    """host -> median per-call time in ms, over `samples` sequential calls.

    `call(endpoint)` performs one replication call and raises on failure. Hosts
    are probed one after another, so probes do not queue behind each other at a
    shared node. A host whose every probe fails is left out.
    """
    out = {}
    for host, endpoint in sorted(endpoints.items()):
        times = []
        for _ in range(samples):
            start = time.perf_counter()
            try:
                call(endpoint)
            except Exception:  # a failed probe is a missing sample, not a crash
                continue
            times.append((time.perf_counter() - start) * 1000.0)
        if times:
            out[host] = statistics.median(times)
    return out


def excess_over_baseline(rtt_ms):
    """host -> ms above the median host, clipped at zero."""
    if not rtt_ms:
        return {}
    base = statistics.median(rtt_ms.values())
    return {h: max(0.0, v - base) for h, v in rtt_ms.items()}


def report(ctx, degraded, seed, hosts, graph, healthy_overrides,
           probed_excess=None, rtt_noise_ms=0.0, feature_noise=0.0, stale_frac=0.0):
    """A copy of `ctx` holding what the selectors are told for this context seed.

    `healthy_overrides` are the overrides without the degraded hosts' entries (the
    baseline utilisation of the realistic model, or nothing for the binary one).
    Rebuilding the context from them with the same seed gives every host its
    pre-degradation draw, which is what a stale report shows.

    Returns (reported_ctx, stale_hosts).
    """
    rng = random.Random("telemetry-{}".format(seed))
    out = copy.deepcopy(ctx)

    n_stale = int(round(stale_frac * len(degraded)))
    stale = sorted(rng.sample(sorted(degraded), n_stale)) if n_stale else []
    if stale:
        before = topology_features.build_feature_context(
            hosts=hosts, seed=seed, graph=graph, overrides=healthy_overrides)
        for h in stale:
            out.resources[h] = copy.deepcopy(before.resources[h])

    for h, state in sorted(out.resources.items()):
        if probed_excess is not None:
            state.observed_delay_ms = 0.0 if h in stale else probed_excess.get(h, 0.0)
        if rtt_noise_ms:
            state.observed_delay_ms = max(0.0, state.observed_delay_ms
                                          + rng.gauss(0.0, rtt_noise_ms))
        if feature_noise:
            state.health = min(1.0, max(0.01, state.health + rng.gauss(0.0, feature_noise)))
            ratio = state.capacity / state.max_capacity
            ratio = min(1.0, max(0.01, ratio + rng.gauss(0.0, feature_noise)))
            state.capacity = ratio * state.max_capacity
    return out, stale


def describe(probed_ms, excess, degraded, limit=None):
    """Lines for the run log: what was probed, and how it compares to the model."""
    lines = []
    hosts = sorted(probed_ms, key=lambda h: -excess.get(h, 0.0))
    for h in hosts[:limit] if limit else hosts:
        lines.append("  {:<6} {:>8.2f} ms  excess {:>7.2f} ms{}".format(
            h, probed_ms[h], excess.get(h, 0.0), "  (degraded)" if h in degraded else ""))
    return lines
