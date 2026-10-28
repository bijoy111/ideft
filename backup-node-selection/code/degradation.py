"""Degradation models for backup hosts.

Why a second model exists
-------------------------
The original scenarios mark a host degraded by pinning it to a single extreme
point: health 0.05 against healthy hosts drawn from U[0.5, 1.0], and a capacity
ratio of 0.033 against a healthy ratio of 1.0. Degraded hosts are therefore
separated from healthy ones by roughly an order of magnitude on *both* scoring
features at once, and every degraded host is identical to every other.

That makes the discrimination trivial, and the weight-sensitivity sweep shows it:
all 287 weight configurations avoid degraded hosts at every topology, including a
configuration that puts 100% of its weight on hop-count and therefore cannot see
host quality at all. A result that holds even for a scoring function with no
discriminative power is a property of the benchmark, not of the method.

The `realistic` model fixes three things:

1. **Severity is continuous, not pinned.** Each degraded host draws a severity in
   [0, 1] and its features interpolate from the healthy range down to the extreme.
   A host at severity 0.1 is genuinely marginal, which is where placement decisions
   are actually difficult and where a scoring function can be wrong.

2. **Degradation has kinds, not one kind.** A host starved of CPU, one with a
   failing disk, one near its connection-table limit and one behind a congested
   link present differently. Each kind drives a different subset of the scoring
   features, so a selector that reads only health no longer catches everything.
   This is what the reviewers asked for when they noted that only one type of
   degraded host was evaluated.

3. **Degradation is partial.** A host can be slow but healthy, or healthy but
   nearly full. The binary model moves every feature together, which no real
   failure does.

The binary model is retained verbatim so previously published numbers stay
reproducible; `--degradation-model binary` selects it.
"""

import random

# Healthy hosts are drawn from these ranges in topology_features.generate_synthetic_resources.
# The degradation kinds below interpolate *into* and past the bottom of these ranges, so a
# mildly degraded host overlaps the healthy population rather than sitting far outside it.
HEALTHY_HEALTH = (0.5, 1.0)
HEALTHY_CPU = (0.4, 1.0)
HEALTHY_STORAGE = (0.4, 1.0)

# Per-kind response: field -> (value at severity 0, value at severity 1).
# A field absent from a kind is left at its healthy draw, which is what makes the
# degradation partial.
KINDS = {
    "cpu_starved": {
        "cpu_availability": (0.40, 0.02),
        "health": (0.60, 0.25),
        "delay_ms": (1.0, 45.0),
    },
    "unhealthy": {
        # Failing hardware: errors and retries show up as health and latency,
        # while capacity and CPU look fine. A capacity-only selector misses this.
        "health": (0.45, 0.03),
        "delay_ms": (2.0, 30.0),
    },
    "capacity_exhausted": {
        # Connection-table or session-slot pressure. Health looks fine, so a
        # health-only selector misses this one.
        "capacity_ratio": (0.45, 0.02),
        "delay_ms": (0.5, 12.0),
    },
    "storage_pressured": {
        "storage_availability": (0.35, 0.02),
        "health": (0.55, 0.30),
        "delay_ms": (1.0, 25.0),
    },
    "link_congested": {
        # Purely a network-path problem: every resource gauge reads healthy and
        # only measured latency reveals it.
        "delay_ms": (5.0, 60.0),
    },
}

# Severity is drawn from a distribution that deliberately includes marginal hosts.
# Beta(1.6, 1.6) is unimodal and symmetric on [0,1]: most hosts are moderately
# degraded, with both near-healthy and severe tails represented. A uniform draw
# would under-represent the middle, which is the interesting region.
SEVERITY_ALPHA = 1.6
SEVERITY_BETA = 1.6


def _lerp(lo_hi, severity):
    lo, hi = lo_hi
    return lo + (hi - lo) * severity


# Under the binary model every healthy host starts completely empty
# (capacity == max_capacity, ratio 1.0), so any reduction at all marks a host as
# degraded and the capacity feature is effectively a flag rather than a measurement.
# That is also the formulation a reviewer objected to: hosts already backing other
# primaries are treated as though they were idle. The realistic model gives healthy
# hosts a baseline utilisation, so remaining capacity becomes a continuous quantity
# that a degraded host can overlap.
HEALTHY_CAPACITY_RATIO = (0.55, 1.0)


def healthy_overrides(healthy_hosts, seed=0):
    """Baseline utilisation for hosts that are not degraded (realistic model only)."""
    rng = random.Random(("healthy", seed).__hash__() & 0xFFFFFFFF)
    return {h: {"capacity_ratio": round(rng.uniform(*HEALTHY_CAPACITY_RATIO), 3)}
            for h in sorted(healthy_hosts)}


def binary_profile():
    """The original all-or-nothing degraded host, retained for reproducibility."""
    return {"health": 0.05, "capacity": 2, "max_capacity": 60}, 0.05


def realistic_profiles(degraded_hosts, seed=0):
    """Assign each degraded host a kind and a severity.

    Returns (overrides, delays, profiles):
      overrides -- host -> field overrides for generate_synthetic_resources
      delays    -- host -> added per-call server delay in seconds
      profiles  -- host -> {kind, severity} for reporting

    `capacity_ratio` is resolved against each host's own drawn max_capacity inside
    topology_features.generate_synthetic_resources, since that is where it is known.
    """
    rng = random.Random(seed)
    kinds = sorted(KINDS)
    overrides, delays, profiles = {}, {}, {}

    for i, host in enumerate(sorted(degraded_hosts)):
        # Round-robin over kinds before randomising, so that even a small degraded
        # set covers several distinct failure modes rather than sampling one twice.
        kind = kinds[i % len(kinds)]
        severity = rng.betavariate(SEVERITY_ALPHA, SEVERITY_BETA)
        response = KINDS[kind]

        fields = {}
        for field, lo_hi in response.items():
            value = _lerp(lo_hi, severity)
            if field == "delay_ms":
                delays[host] = value / 1000.0
                # Also expose it as a measurable host property, so a selector with
                # RTT telemetry can see what a resource-gauge-only selector cannot.
                fields["observed_delay_ms"] = round(value, 3)
            else:
                fields[field] = round(value, 3)
        overrides[host] = fields
        profiles[host] = {"kind": kind, "severity": round(severity, 3)}

    return overrides, delays, profiles


def build(degraded_hosts, model="binary", seed=0, healthy_hosts=()):
    """Return (overrides, delays, profiles) for the chosen model.

    `overrides` covers degraded *and*, under the realistic model, healthy hosts,
    since the latter carry a baseline utilisation there. `delays` maps host ->
    seconds of added per-call server delay: a flat 50 ms under the binary model,
    scaled by kind and severity under the realistic one.
    """
    degraded_hosts = list(degraded_hosts)
    if model == "binary":
        if not degraded_hosts:
            return {}, {}, {}
        fields, delay = binary_profile()
        fields = dict(fields, observed_delay_ms=round(delay * 1000.0, 3))
        return ({h: dict(fields) for h in degraded_hosts},
                {h: delay for h in degraded_hosts},
                {h: {"kind": "binary", "severity": 1.0} for h in degraded_hosts})
    if model == "realistic":
        overrides, delays, profiles = realistic_profiles(degraded_hosts, seed=seed)
        overrides.update(healthy_overrides(
            [h for h in healthy_hosts if h not in set(degraded_hosts)], seed=seed))
        return overrides, delays, profiles
    raise ValueError("unknown degradation model: {}".format(model))


def describe(profiles):
    """One-line-per-host summary for the experiment log."""
    if not profiles:
        return "none"
    return ", ".join(
        "{}({}, s={:.2f})".format(h, p["kind"], p["severity"])
        for h, p in sorted(profiles.items()))
