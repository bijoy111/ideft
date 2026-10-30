# Intelligent Backup-Node Selection: Methodology and Results

This document reports the experimental evaluation of replacing DEFT's static, hand-picked
backup-node mapping with feature-aware selection methods, including a trained RL policy,
following on from the motivation in [Thesis Overview.md](Thesis%20Overview.md).

## 1. Problem recap

DEFT replicates each primary VNF's state to a backup node for fault tolerance. The
shipped mapping (`code/primary_to_secondary.json`) is a
hand-picked, static assignment with no load or health awareness — in the real 9-host
topology it concentrates 3 of 5 primaries onto a single backup (`h4`). This document
evaluates whether smarter selection actually improves anything measurable, and if so,
under what conditions.

## 2. Methods compared

All methods are implemented in `code/backup_selector.py`
and `rl_backup_selector.py`.

| Method | Description |
|---|---|
| `baseline` | DEFT's shipped static hardcoded mapping (9-host topology only). |
| `concentrated` | Every primary assigned to the same single backup — a scale-independent, maximally naive worst case. |
| `random` | Uniform-random feasible backup per primary, seeded. |
| `round_robin` | Cycles through the feasible-backup pool in order; load-balances blindly, with zero awareness of host health/capacity/latency. |
| `heuristic` | Deterministic greedy weighted scorer (health, remaining capacity, latency, hop-count), with a load-decay penalty after each assignment so it naturally spreads load. |
| `optimal` | Globally optimal assignment under a hard per-backup capacity constraint, solved exactly via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) — an upper bound for the heuristic/RL to be judged against. |
| `rl` | PPO agent (`stable_baselines3`), trained on the same scoring objective as `heuristic`, with action masking (invalid actions penalized during training, not just corrected post-hoc) and a fixed reward-indexing bug present in the repo's original `predict.ipynb` prototype. |

## 3. Experimental design

### 3.1 Testbed: localhost RPC-isolation, not Mininet

Backup-selection quality is measured by driving the **real, unmodified** `Backup`/`Primary`
2PC classes (`exp_package/Two_phase_commit/backup_2pc.py`, `primary_2pc.py`) directly over
XML-RPC, with each backup host running as its own OS process on localhost
(`rpc_contention_experiment.py`). This is a deliberate departure from the physical Mininet
network path (`test.py`) — see §5 (Threats to Validity) for why, and what was tried first.

Each primary fires a **burst** of `--burst` simultaneous in-flight RPC calls (not one
sequential call-then-wait stream), matching how concurrent primaries would actually load a
shared backup in production. Three implementation bugs were found and fixed during
development of this benchmark itself: (1) running backup servers as threads within one
process shares a GIL, which confounds the very effect being measured — servers now run as
real separate OS processes; (2) creating a new `ServerProxy` per call forces a fresh TCP
handshake every time, dominating the measurement with connection-thrash noise instead of
RPC/queueing latency — each burst worker now keeps one persistent connection; (3) an early
version built a single host-resource context per seed and passed that same object to every
method, but `heuristic` and `rl` apply an in-place load-decay penalty to each host they
pick, so any method evaluated after them scored against an already-depleted context and its
mapping depended on execution order — each method now receives its own `copy.deepcopy` of
the context. (This last bug is why the reproduction re-run supersedes an earlier draft: on
the shared context `rl` had appeared to collapse onto ~13 backups at 128 hosts, but on a
private context it spreads across 24–30, matching the `--check-collapse` diagnostic.)

### 3.2 Four scenarios

**Homogeneous** — the real 9-host DEFT topology (`config.py`), 5 primaries, synthetic
resource values drawn independently per context seed (no engineered heterogeneity).

**Scaled + heterogeneous (16 hosts)** — a synthetic 16-host topology (10 primaries, 6
backup-eligible hosts), with 2 of the 6 backup-eligible hosts deliberately degraded
(health=0.05, capacity ratio ≈0.03, plus an artificial +50ms per-call server-side delay
simulating a genuinely overloaded machine). This isolates whether feature-aware methods
actually avoid known-bad hosts, independent of load-balancing alone.

**Scaled + heterogeneous (32 hosts)** — same construction doubled (20 primaries, 12
backup-eligible hosts, 4 degraded, same proportions as the 16-host case), used to check
whether the 16-host finding is a scale-specific artifact or holds as the candidate pool
grows (§4.5).

**Scaled + heterogeneous (128 hosts)** — same construction again ×4 (80 primaries, 48
backup-eligible hosts, 16 degraded), a full order of magnitude above the 16-host point,
with the same trial count as the original 16-host run rather than a reduced one (§4.6).

### 3.3 Statistical methodology

- **Multiple independent topology/resource realizations** (`--context-seeds`), not just
  repeated trials of one fixed instance — repeated trials of a single instance only
  capture RPC-timing noise, not whether a result generalizes across problem instances.
  Final numbers aggregate across all (context seed × trial) combinations.
- **Per-trial method order is randomized** so no method is systematically favored or
  disadvantaged by *when* it happens to run (background OS load, thermal drift).
- **Mann-Whitney U** (two-sided) for pairwise significance, plus **rank-biserial
  correlation** as an effect size — with n in the tens of thousands, p-values are almost
  always significant regardless of practical magnitude; the effect size is what tells you
  whether a difference actually matters (conventionally |r|≈0.1/0.3/0.5 ≈ small/medium/large).
- **Bootstrap 95% CI** on the mean (5000 resamples).
- Raw per-call latencies are written to
  `code/results_rpc/rpc_contention_raw.csv` for
  independent re-analysis.

## 4. Results

### 4.1 Homogeneous scale (5 context seeds × 3 trials, 200 calls/primary, burst=10)

| method | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| baseline | 15000 | 57.33 | 10.67 | 41.24 | 1071.77 |
| concentrated | 15000 | 94.99 | 6.81 | 221.69 | 3125.54 |
| random | 15000 | 45.98 | 12.35 | 47.50 | 1063.38 |
| round_robin | 15000 | 35.67 | 18.46 | 54.27 | 1035.42 |
| heuristic | 15000 | 41.06 | 17.48 | 50.68 | 1054.36 |
| optimal | 15000 | 37.00 | 19.63 | 56.03 | 1034.23 |
| rl | 15000 | 36.20 | 18.88 | 54.47 | 1038.38 |

Key pairwise effect sizes (rank-biserial r):
- `round_robin` vs `rl`: r=+0.008, **not significant** (p=0.23) — the only non-significant pair, indistinguishable.
- `round_robin` vs `optimal`: r=+0.031, `heuristic` vs `rl`: r=+0.036 — significant on sample size alone, negligible effect.
- `round_robin` vs `heuristic`: r=-0.029, significant but **small** effect.
- `baseline`/`concentrated` vs everything else: r in [0.08, 0.40], significant with **small-to-medium** effect — clearly worse.

**Interpretation**: at small homogeneous scale, any method that avoids gross load
concentration (round_robin, heuristic, optimal, rl) performs similarly well; the
feature-aware methods show no meaningful advantage over blind round-robin here, and the
effect size is small even where "significant." The real gap is between
naive-concentration policies (baseline, concentrated) and everything else. This matches
the underlying mechanism found in §4.3: contention (`backup_load`), not feature-awareness,
is what drives the difference at this scale.

### 4.2 Scaled + heterogeneous (5 context seeds × 3 trials, 200 calls/primary, burst=10)

16 hosts, 10 primaries, 6 backup-eligible hosts, 2 of which (`w11`, `w12` in seed 0's
layout) are deliberately degraded (health=0.05, capacity ratio≈0.03, +50ms artificial
per-call delay). `baseline` is excluded here — it's only defined for the real 9-host
topology.

Degraded-host usage (out of 10 assignments per method per seed, summed check across the
run): `concentrated`, `heuristic`, `optimal`, and `rl` **never** route a primary to a
degraded host across any of the 5 seeds. `round_robin` hits a degraded host **2/10 times
in every single seed** (it cycles the backup pool blindly and has no way to know `w11`/
`w12` are bad). `random` is inconsistent — 0 to 4/10 depending on seed, as expected from
chance alone.

| method | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| concentrated | 30000 | 157.37 | 4.89 | 1029.37 | 3361.56 |
| random | 30000 | 124.97 | 16.77 | 360.19 | 1069.84 |
| round_robin | 30000 | 106.94 | 22.41 | 360.26 | 1040.29 |
| heuristic | 30000 | 39.11 | 19.55 | 53.68 | 1050.06 |
| optimal | 30000 | 37.25 | 21.46 | 54.44 | 1036.45 |
| rl | 30000 | 37.71 | 21.47 | 56.97 | 1039.31 |

Key pairwise effect sizes (rank-biserial r):
- `round_robin` vs `heuristic`: r=-0.1465, **small-to-medium** effect — heuristic clearly better.
- `round_robin` vs `optimal`: r=-0.1008, significant, **small** effect.
- `round_robin` vs `rl`: r=-0.0939, **small** effect.
- `heuristic` vs `optimal`: r=+0.0586, **negligible-to-small** effect (heuristic is marginally behind optimal here).
- `heuristic` vs `rl`: r=+0.0673, **negligible** — RL matches the heuristic.
- `optimal` vs `rl`: r=+0.0089, **not significant** — RL is essentially at the optimal bound.
- `concentrated` vs everything: r in [0.47, 0.56], **large** effect — the naive worst case
  is dramatically worse at this scale, since it now piles 10 primaries onto one backup
  instead of 5.
- `random` vs `heuristic`/`optimal`/`rl`: r in [+0.01, +0.07], small/negligible on
  central-tendency stats, but random's mean (124.97ms) is over 3x heuristic/optimal/rl's
  (~37-38ms) — the rank-based effect size is muted by random's high variance across
  seeds (0 to 4/10 degraded-host hits), while the raw mean makes the practical gap obvious.

**Interpretation**: this is the scenario that isolates *feature awareness*, not just load
balancing. `round_robin` load-balances as well as anyone (it's designed to), but it is
mechanically blind to host quality — it is the only method (besides `random`) that
routinely uses degraded hosts, and this shows up directly in its mean latency (107ms vs
~37-39ms for the feature-aware methods) and its p95 (360ms vs ~54-57ms). `heuristic`,
`optimal`, and `rl` all learn/compute to avoid `w11`/`w12` entirely across every seed, and
end up statistically indistinguishable or nearly so from each other (all pairwise effect
sizes among these three are negligible-to-small) — confirming the RL policy generalizes
across independent topology/resource realizations and matches the Hungarian-algorithm
optimal bound, not just one memorized instance. This is the core empirical result
supporting the thesis's claim: feature-aware backup selection (heuristic or RL) provides
a real, mechanistically explained, statistically robust improvement over blind
load-balancing once hosts differ in quality — which is the realistic case for elastic
scaling in production, where backup capacity is added/removed under varying load and
health conditions, not a uniform fixed pool.

### 4.3 Latency vs. backup_load (mechanism check, pooled across all methods/seeds)

Homogeneous scenario:

| backup_load | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| 1 | 70800 | 35.35 | 17.94 | 52.07 | 1036.52 |
| 2 | 13200 | 53.44 | 10.23 | 49.57 | 1074.48 |
| 3 | 9000 | 74.77 | 8.14 | 221.96 | 1090.20 |
| 4 | 12000 | 111.85 | 6.12 | 1018.58 | 3140.93 |

This is monotonic and mechanistically explicable: p99 latency roughly **triples** going
from load 1 to load 4, consistent with TCP accept-queue backlog overflow at Python's
`SimpleXMLRPCServer` (default backlog, single-threaded) once enough concurrent connection
attempts pile onto one backup — a genuine architectural property of DEFT's real,
unmodified backup implementation, not a benchmark artifact.

Scaled + heterogeneous scenario (pooled across `concentrated`/`random`/`round_robin`/
`heuristic`/`optimal`/`rl`):

| backup_load | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| 1 | 135600 | 58.19 | 20.67 | 355.99 | 1038.28 |
| 2 | 8400 | 49.88 | 13.35 | 49.22 | 1071.96 |
| 3 | 9000 | 238.40 | 14.08 | 361.21 | 1390.65 |
| 9 | 27000 | 172.04 | 4.59 | 1036.05 | 3374.23 |

Load=9 is `concentrated` piling 9 of its 10 primaries onto a single backup (the tenth is that backup's own primary, which cannot back up to itself) — mean/p99 both
degrade sharply relative to load=1-2, reproducing the same contention mechanism at a
larger, more extreme sharing ratio. The load=1 row's elevated p95 (355.99ms, versus
52.07ms in the homogeneous scenario) is not a contention artifact: it's `round_robin`
and `random` pooled in with everyone else at load=1 while still occasionally landing on
the artificially-delayed degraded hosts (+50ms server-side, compounded by retry/backoff
on top), which the *load* column alone doesn't capture — this is precisely why §4.2
reports per-method statistics separately rather than relying on this pooled mechanism
check alone.

### 4.4 Selection overhead (compute time per mapping)

"Elastic scaling" requires the selection algorithm itself to be fast — a method that
matches the optimal bound on placement quality but takes seconds to compute is a
materially worse deployment story than one that's comparably fast. `selection_overhead.py`
times one full mapping computation per method (30 repeats, first call excluded as warmup),
across all four topology sizes (128-host `rl` uses the fixed `rl_model_scaled128_v3.zip`
from §4.6, with its larger `[256,256]` network):

| method | 9 hosts (ms) | 16 hosts (ms) | 32 hosts (ms) | 128 hosts (ms) |
|---|---|---|---|---|
| concentrated | 0.005 | 0.013 | 0.055 | 0.731 |
| random | 0.021 | 0.029 | 0.084 | 0.639 |
| round_robin | 0.004 | 0.005 | 0.015 | 0.030 |
| heuristic | 0.055 | 0.161 | 0.646 | 7.260 |
| optimal (Hungarian) | 0.097 | 0.208 | 0.913 | 8.943 |
| rl (PPO inference) | 1.81 | 3.76 | 11.29 | 49.64 |

Even the slowest method (`rl`) computes a 128-host mapping in about 50ms, and every other
method in under 10ms — negligible relative to any realistic
failure-detection interval (typically hundreds of ms to seconds in NF failure-recovery
literature). `rl` is consistently the slowest (PPO forward passes, one per primary, plus
`stable_baselines3`/PyTorch call overhead), and the jump at 128 hosts (50ms, vs. 11ms at
32 hosts) is the direct cost of the `[256,256]` network needed to fix the collapse in §4.6,
on top of more primaries to loop over. `heuristic` and `optimal` grow roughly linearly/superlinearly
with host count (as expected — the Hungarian algorithm is $O(n^3)$), reaching low
double-digit milliseconds at 128 hosts. **Conclusion**: even the largest, fixed-collapse
`rl` model and the Hungarian solver stay comfortably under any realistic decision-latency
budget — selection overhead is not what limits scalability at 128 hosts; §4.6 has already
established that the limit is `rl`'s load-balancing *quality*, not its selection *speed*.

### 4.5 Scalability trend: 16 → 32 hosts (heterogeneous)

Same construction as §4.2, doubled (20 primaries, 12 backup-eligible hosts, 4 degraded),
3 context seeds × 2 trials, 150 calls/primary, burst=8 (reduced from 5×3×200/10 to keep
the larger-scale run tractable on a single machine — see §5).

Degraded-host usage (out of 20 assignments per method per seed): `concentrated`,
`heuristic`, `optimal`, and `rl` again **never** route to a degraded host, in any of the 3
seeds. `round_robin` hits one in **4/20 times, every seed** — the same ~20% rate as at
16 hosts (2/10). Note this is *not* random variation: `round_robin_mapping()` takes no
resource context, so its output is entirely deterministic given the topology (identical
across every context seed at a fixed scale). Whether it happens to land on the degraded
range is purely a function of how Python's alphabetical string-sort of host names
(`"w1" < "w10" < "w11" < ... < "w2"`, not numeric order) happens to align with wherever
the degraded hosts sit in that sort — see §4.6, where this exact coincidence flips to a
**0/80** hit rate at 128 hosts, showing the "~20%" figure is not a stable property of
blind cycling in general.

| method | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| concentrated | 18000 | 210.35 | 4.28 | 1053.62 | 5538.62 |
| random | 18000 | 91.00 | 24.49 | 358.65 | 1050.90 |
| round_robin | 18000 | 98.97 | 28.78 | 359.29 | 363.41 |
| heuristic | 18000 | 36.49 | 26.49 | 58.41 | 105.63 |
| optimal | 18000 | 35.65 | 27.70 | 58.59 | 103.48 |
| rl | 18000 | 36.41 | 25.86 | 54.58 | 211.59 |

Key pairwise effect sizes (rank-biserial r):
- `round_robin` vs `heuristic`: r=-0.1422, **small**, same direction and similar magnitude
  as at 16 hosts (r=-0.1465) — the advantage persists at 2x scale, it does not vanish.
- `heuristic` vs `optimal` vs `rl`: r in [-0.07, +0.04] pairwise, all **small**, matching
  the 16-host pattern — the three feature-aware methods remain statistically close to
  each other at larger scale.
- `concentrated` vs everything: r in [0.71, 0.76], **large** — *stronger* than at 16 hosts
  (r was 0.47–0.56), because piling 19 of the 20 primaries onto one backup (vs. 9 of 10) pushes the
  single-threaded `SimpleXMLRPCServer` further into queue backlog (backup_load=19 bucket:
  mean 220ms, p99 6452ms, vs. the 16-host load=9 bucket's mean 172ms, p99 3374ms).

**Interpretation**: the central 16-host finding — feature-aware methods avoid degraded
hosts and materially outperform blind round-robin, while RL matches the heuristic and the
Hungarian-optimal bound — holds at 2x scale rather than being a small-topology artifact.
The one method whose relative penalty *grows* with scale is the naive worst case
(`concentrated`), which is expected: piling more primaries onto a single backup produces
proportionally worse queueing contention. This is a genuine (if modest) scalability
result, not just a repeat of §4.2 at a bigger number — but it is only two points (16, 32)
on a curve, not a sweep across an order of magnitude; see §4.6 for a third point at 128
hosts, which complicates the round-robin comparison specifically (see above) while
reinforcing the concentrated/heuristic/optimal findings.

### 4.6 Scalability trend continued: 128 hosts (heterogeneous)

> **The latency table in this section is withdrawn.** It was produced with 128 backup
> servers and the client on a single 8-core machine, so it measures CPU scheduling as much
> as backup placement: two independent executions of the identical gRPC configuration
> differ by 4.8× on `optimal` and 6.1× on `rl`. The structural results below (distinct
> backups used, peak load, degraded-host avoidance) are computed from the mapping and do
> not depend on timing, so they stand. **§4.11 reports this scenario on ten bare-metal
> nodes**, and the numbers there differ substantially — in particular, most of the
> concentration penalty seen here is DEFT's five-deep accept queue rather than load on the
> backup. The table is kept rather than deleted so the withdrawal is auditable.


Same construction again (80 primaries, 48 backup-eligible hosts, 16 degraded), matched to
the original 16-host trial count (5 context seeds × 3 trials, 200 calls/primary, burst=10)
per the request to keep this point statistically comparable rather than reduced like the
32-host point.

This run required a benchmark-harness fix first: the original 3-attempt retry budget in
`replicate_burst()` was written when the worst case was ~10 primaries sharing one backup;
at 79 primaries sharing one backup (the `concentrated` method), Python's
`SimpleXMLRPCServer` default accept backlog (5) was overwhelmed badly enough that calls
kept failing past 3 retries and crashed the whole experiment with an uncaught
`ConnectionResetError`. This was hardened to 8 attempts with jittered exponential backoff,
and a call that still fails after all retries is now recorded as a failure (latency =
10000ms sentinel) instead of crashing — the same robustness a real production 2PC client
would need. This changes the *harness's* failure handling, not the backup server being
measured (still the real, unmodified `Backup` class); the 16/32-host results above never
exhausted the original 3-attempt budget, so they are unaffected in practice, but the
retry-policy difference between runs is itself worth flagging as a methodology
inconsistency (see §5).

| method | n | mean (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| concentrated | 240000 | 772.41 | 4.05 | 3325.13 | 18971.94 |
| random | 240000 | 101.54 | 32.20 | 357.33 | 1087.47 |
| round_robin | 240000 | 48.13 | 30.95 | 69.15 | 1058.35 |
| heuristic | 240000 | 47.92 | 30.97 | 67.57 | 1058.17 |
| optimal | 240000 | 49.06 | 31.38 | 71.57 | 1057.86 |
| rl | 240000 | 457.86 | 26.65 | 232.01 | 7178.12 |

(`concentrated`'s tail is inherently unstable at this scale. In the reported run, a burst
of calls against the single backup serving 79 other primaries (context seed 2) exhausted
the 8-retry budget and were recorded at the 10,000ms sentinel; the other four seeds
completed every call. That is enough to swing the mean and p99 seed-to-seed. The
qualitative conclusion — a catastrophic outlier more than an order of magnitude above
every other method — holds regardless.)

**`concentrated` at this scale is not just "worse" — it sits at the edge of outright
failure.** In the reported run, a burst of calls against the single backup serving 79
other primaries (seed 2) failed even after 8 retries with exponential backoff, recorded at
the 10,000ms sentinel; the other four seeds completed every call. Either way, mean latency
(772.41ms) is more than an order of magnitude above every other method, with p99 at roughly
19 seconds of genuine queueing delay. That whether calls fail at all swings across
otherwise-identical seeds is itself part of the finding. This is a real, concrete finding
about DEFT's backup implementation: using Python's default `SimpleXMLRPCServer` accept
backlog, it cannot reliably serve ~79 concurrent primaries sharing one backup even with
client-side retry-with-backoff — a genuine scalability ceiling of the shipped
implementation, not a benchmark artifact.

**`round_robin`/`heuristic`/`optimal` are practically indistinguishable at this scale.**
`round_robin`'s mapping happened to hit **0/80** degraded-host assignments here (the
alphabetical-sort coincidence noted above), so at 128 hosts it isn't being penalized for
the mechanism this scenario is designed to test. Their means sit within about 2% of one
another (47.9, 48.1, and 49.1ms for `heuristic`, `round_robin`, and `optimal`), and since
these three methods' mappings are fully determined by the topology and context seed, that
residual spread is RPC-timing noise, not a quality difference. This is not a reversal of
the thesis's core claim — it is the clearest illustration of it: **round_robin's outcome is
uncontrolled**. It scored badly at 16/32 hosts and fine at 128 hosts purely by coincidence
of hostname ordering versus where the degraded hosts happened to sit. `heuristic`/`optimal`
avoid degraded hosts *by construction*, regardless of naming or indexing coincidences or
run-to-run noise — a predictable guarantee round_robin cannot offer, even in the scale
point where it happened to get lucky.

**RL: attempted fix, partial success.** The original model (`rl_model_scaled128_collapsed.zip`,
400,000 timesteps, SB3's default `ent_coef=0.0` and `net_arch=[64,64]`) converged to a
near-degenerate policy — across the 5 seeds it used only **1–5** distinct backups,
funneling nearly every one of the 80 primaries onto the same single host (`w101`),
collapsing to a `concentrated`-like mapping and triggering the accept-backlog failure
described above. Since SB3's PPO applies **zero entropy bonus by default**, nothing
discouraged the policy from locking onto one action early once it looked decent on average
— a plausible root cause, along with `[64,64]` possibly lacking the capacity to represent a
policy conditioned on which of 80 different one-hot primaries is active. Two escalating
retraining steps were taken:

1. `ent_coef=0.01`, `net_arch=[128,128]`, 600k timesteps: partial improvement only; not
   released as a checkpoint.
2. `ent_coef=0.02`, `net_arch=[256,256]`, `n_steps=4096`, `batch_size=256`, 1M timesteps
   (`rl_model_scaled128_v3.zip`, the model used for the reported `rl` results).

**This final fix repairs one of the policy's two failings completely and the other only in
part.** Measured directly from the reported run (`--context-seeds 1 2 3 4 5`):

| seed | distinct backups | max backup load | most-loaded host |
|---|---|---|---|
| 1 | 29 | 23 | `w53` (23 of 80) |
| 2 | 24 | 28 | `w53` (28 of 80) |
| 3 | 30 | 9 | `w39` (9 of 80) |
| 4 | 24 | 17 | `w39` (17 of 80) |
| 5 | 29 | 14 | `w10` (14 of 80) |
| **`heuristic`, seed 1 (for contrast)** | **74** | **2** | 68 backups carry exactly 1 |

- **Host quality: completely fixed.** `0/80 degraded-host assignments across all 5 seeds`
  (0 of 400), matching `heuristic`/`optimal` exactly.
- **Load balancing: improved, not fixed.** The policy now spreads across **24–30** distinct
  backups against the heuristic's **74–76** — a large jump from the collapsed 1–5 — but in
  the tighter seeds it still funnels **as many as 28 of the 80 primaries onto one favored
  host** (peak per-backup load ranges 9–28 across seeds). Those peak loads are independently
  visible as their own rows in the `backup_load` breakdown of the reported run.
- **The penalty is entirely in the tail.** `rl`'s median latency (26.65ms) is in fact the
  lowest of any method in the table above; only its mean and p99 pay for the concentration.

Measured under the real RPC benchmark, this costs: mean 457.86ms and p99 7178.12ms —
roughly 7-10x worse than `heuristic`/`optimal`/`round_robin` on the mean and tail, though
still far better than the original collapse (which crashed the harness entirely) or
`concentrated`'s p99 (18972ms).

**The network learned host-quality-awareness (avoiding degraded hosts) perfectly and
load-balancing precision only partially.** The reward's health/capacity terms label which
hosts are bad and the network learns that completely; nothing penalizes concentration
directly, so a few *labeled-good* hosts absorb a disproportionate share of the flow set
without the policy paying enough for it. This is a more precise, more useful
characterization of the limitation than "RL doesn't scale": the two properties are learned
at radically different rates as the action space grows, and the latter converged only part
of the way within this training budget.
Reward shaping that
penalizes load concentration more directly (rather than relying on the existing
per-assignment health/capacity decay to implicitly teach spreading) is the natural next
step, not attempted here.

### 4.7 Throughput and failure rate

Complements §4.1–4.6 (per-call latency) with the other two metrics DEFT's own base paper
(CNSM'23) reports: throughput and packet drop. `throughput_experiment.py` runs each
primary's burst workers for a fixed 3-second wall-clock window (rather than a fixed call
count) and measures successful calls/sec and the percentage that time out or fail even
after retry. **Naming note**: this testbed is RPC-level (TCP), not DEFT's own packet-level
input-buffer-overflow mechanism (which needs the Mininet/stamped-packet path this thesis's
RPC-isolation testbed deliberately bypasses — §3.1). "Failure rate" here means RPC calls
that never complete despite retries; it is evidence of the same underlying phenomenon
(an overloaded backup failing to keep up with demand), not a literal reproduction of
DEFT's packet-drop metric.

**Homogeneous scale** (5 seeds × 3 trials, 3s bursts, burst=10):

| method | n_runs | mean throughput (calls/s) | mean failure rate (%) | 95% CI (calls/s) |
|---|---|---|---|---|
| baseline | 15 | 1176.0 | 0.38 | [1144.8, 1206.1] |
| concentrated | 15 | 1158.1 | 0.70 | [1123.0, 1188.0] |
| random | 15 | 1180.5 | 0.26 | [1154.2, 1203.4] |
| round_robin | 15 | 1172.9 | 0.06 | [1159.5, 1186.9] |
| heuristic | 15 | 1197.3 | 0.14 | [1175.6, 1215.2] |
| optimal | 15 | 1159.2 | 0.07 | [1120.7, 1187.7] |
| rl | 15 | 1175.1 | 0.09 | [1160.0, 1189.9] |

Only 3 of the 21 pairwise comparisons reached significance in this run (all three
involving `heuristic`, p∈[0.03,0.05], small effects), and none corresponds to a stable
ordering. **Do not read that as equivalence, and do not read it as a stable result either
— at n=15 this measurement is dominated by background system load, not the mapping under
test.** Each method contributes only 15 aggregate measurements here (5 seeds × 3 trials),
against the tens of thousands of individual calls behind each latency test, so the test
has little power to resolve the few-percent differences in the table above, and the
verdict is not reproducible. The one consistent signal is failure rate: `concentrated`'s
0.70% is the highest of any method, while the feature-aware methods stay at 0.07–0.14%.
Treat the 16-host scenario below, where contention is severe enough to produce a large and
stable effect, as the only informative throughput result.

**Scaled + heterogeneous, 16 hosts** (5 seeds × 3 trials, 3s bursts, burst=10):

| method | n_runs | mean throughput (calls/s) | mean failure rate (%) | 95% CI (calls/s) |
|---|---|---|---|---|
| concentrated | 15 | 1191.4 | 2.57 | [1140.5, 1253.4] |
| random | 15 | 1297.6 | 0.62 | [1245.6, 1353.3] |
| round_robin | 15 | 1288.0 | 0.25 | [1260.9, 1324.2] |
| heuristic | 15 | 1310.0 | 0.29 | [1263.7, 1364.1] |
| optimal | 15 | 1358.8 | 0.16 | [1321.6, 1408.3] |
| rl | 15 | 1352.0 | 0.17 | [1317.3, 1397.9] |

**`concentrated` vs everything: r in [0.63, 0.78], large effect, all p < 0.004** — and its
failure rate (2.57%) is clearly elevated over every other method's (0.16–0.62%), a direct
throughput-side confirmation of the contention mechanism. Gross load concentration is
visible from the throughput side, not only from tail latency.

**The degraded-host comparison, however, resolves only partially from throughput.**
`optimal` and `rl` each significantly out-throughput `round_robin` (r = 0.63 and 0.65,
p < 0.004), consistent with `round_robin` hitting degraded hosts on 2/10 primaries every
seed (§4.2) — yet `heuristic` vs `round_robin` does **not** reach significance (p = 0.38),
despite `heuristic` avoiding degraded hosts exactly as `optimal` and `rl` do. The mechanism
explains the fragility: only 2 of round_robin's 10 primaries are actually paired with a
degraded backup, and their reduced individual throughput gets averaged into the other 8
primaries' normal throughput at the aggregate level, leaving a signal small enough that a
single method's run-to-run noise can decide the verdict. Tail latency (p95/p99) is
dominated by exactly the slow calls that throughput's mean-based aggregation smooths over,
and resolves the same effect consistently — which is why §4.1–4.6 use latency as the
primary metric and treat this throughput/failure-rate result as a corroborating, not a
replacing, measurement. **Reproducers: expect the `concentrated` comparisons to hold every
time and the finer degraded-host ones to be only partially resolved.**

### 4.8 Heuristic weight sensitivity and decay calibration (paper §V-H, Table XV)

The heuristic's scoring weights `(w_h, w_k, w_l, w_d) = (0.35, 0.35, 0.20, 0.10)` and its
load-decay mechanics (`cap -= 20`; `health *= 0.95`) are otherwise chosen by intuition. To
show they are principled rather than ad hoc, `weight_sensitivity.py` sweeps **287 weight
configurations** (286 points on a 0.1-step grid over the 4-simplex, plus the off-grid
published default) across the real 9-host DEFT topology and the synthetic 16/32/128-host
topologies, 5 context seeds each. Each configuration is scored **structurally** — no RPC
calls — via the real `backup_selector.select_backups`, so the sweep is exact and fast.

**Result 1 — degraded-host avoidance is essentially weight-independent.** All 287
configurations, at all four topologies (1,148 config-topology pairs), achieved 0%
degraded-host usage — including weighting 100% on hop-count alone. The degraded hosts
(health 0.05, capacity ratio ≈0.033) are so stark that even a zero-discrimination weight
vector avoids them on tie-break. The safety guarantee is robust to the weight choice.

**Result 2 — load-balancing is highly weight-sensitive, driven by the capacity term.** Max
per-backup load ranges from perfect (1 primary/backup, whenever the capacity weight is
nonzero) to severe (up to 79 of 80 primaries on one backup at 128 hosts, when capacity and
health weights are both zero). The capacity term — realised as the flat `-20` decrement per
assignment — is the dominant load-spreading lever; health (`*= 0.95`) is a weaker one;
latency/hops contribute nothing here (every host is two hops from every primary in these
topologies). The `-20` decay mechanic is therefore *the* mechanism, not an arbitrary knob.

**Result 3 — the published default is near-optimal (5-seed average):**

| Scale | Default max load | Best max load | Default backups used | Max possible |
|---|---|---|---|---|
| 9-host (real) | 1.40 | 1.00 | 4.60 | 5 |
| 16-host | 1.40 | 1.00 | 9.40 | 10 |
| 32-host | 2.00 | 1.00 | 18.00 | 20 |
| 128-host | 2.00 | 1.00 | 74.80 | 80 |

The default lands within one unit of the achievable max-load optimum and uses 90–94% of the
maximum distinct backups at every scale; sorted by (degraded-rate, max-load) it ranks
151st–187th of 287 only because so many configs tie at the max-load-of-1 ceiling. The honest
limitation this surfaces is that the latency/hops terms carry negligible practical weight in
these uniform-hop synthetic topologies — an artifact of the RPC-isolation testbed, not of the
scoring design. Full run saved in `findings/weight_sensitivity.txt`.

**Result 4 — the decay constants themselves are a characterized, monotone lever (paper
§V-H).** `decay_sensitivity.py` sweeps the load-decay mechanics directly — the
capacity decrement (`cap -= 20`) over {5, 10, 20, 40, 80} and the health-decay factor
(`health *= 0.95`) over {0.80, 0.90, 0.95, 0.99, 1.00} — with the scoring weights fixed, 5
seeds per topology (100 decrement×decay×topology cells). Degraded-host avoidance is
*decay-independent* (0 of 100 cells hit a degraded host), and load-balancing is a smooth,
monotone function of both constants: a bigger decrement or a stronger (smaller) health-decay
spreads better. Max per-backup load vs. the capacity decrement (health-decay fixed at the
published 0.95):

| Capacity decrement | 32-host max load | 128-host max load |
|---|---|---|
| 5 | 2.60 | 2.60 |
| 10 | 2.00 | 2.00 |
| 20 (published) | 2.00 | 2.00 |
| 40 | 1.00 | 1.00 |
| 80 | 1.00 | 1.00 |

The mechanism reaches a perfect max-load of 1 at a decrement of 40+. The published decrement
of 20 is a deliberately conservative point — within one unit of the perfect-spread optimum —
that avoids maximally aggressive spreading (which, on a pool with finer non-degraded quality
variation, could trade host quality for balance). So `(20, 0.95)` is a principled choice on a
monotone response surface, not an arbitrary constant. Full run saved in
`findings/decay_sensitivity.txt`.

### 4.9 Hash baselines, a fairer round-robin, and topology diversity (paper §V-I, Table XVI)

`baseline_comparison.py` broadens the baseline set — the NFV analogs of decentralized,
load-aware replica placement (CRUSH, Dynamo, consistent-hashing variants) plus a fairer
round-robin — and compares them structurally (degraded-host hits, max per-backup load; no RPC)
against the feature-aware methods, over both single-hub star and **multi-hub** topologies:

- **`hrw`** — plain rendezvous hashing: decentralized/deterministic, but health/capacity-unaware.
- **`weighted_hrw`** — rendezvous hashing weighted by `health × cap_ratio` (CRUSH/Dynamo-style
  node weights): decentralized *and* quality-aware, but coordination-free.
- **`weighted_chash`** — Dynamo-style weighted **consistent hashing** (hash ring with
  capacity/health-proportional virtual nodes): a second decentralized quality-weighted point.
- **`rr_random`** — round-robin with a random starting offset, so its degraded exposure
  reflects the degraded fraction rather than the fixed alphabetical-sort coincidence.

Heterogeneous star-scenario results (5-seed average; "deg." = primaries on a degraded host,
"load" = max per-backup load):

| Method | 16h deg. | 16h load | 32h deg. | 32h load | 128h deg. | 128h load |
|---|---|---|---|---|---|---|
| round_robin | 2.0 | 1.0 | 4.0 | 1.0 | 0.0 | 1.0 |
| rr_random | 1.2 | 1.0 | 2.6 | 1.0 | 10.8 | 1.0 |
| hrw | 2.0 | 3.0 | 0.0 | 3.0 | 9.0 | 2.0 |
| weighted_hrw | 0.0 | 3.0 | 0.0 | 3.6 | 0.0 | 2.2 |
| weighted_chash | 0.0 | 2.8 | 0.0 | 3.0 | 0.0 | 3.2 |
| heuristic | 0.0 | 1.4 | 0.0 | 2.0 | 0.0 | 2.0 |
| optimal | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |

Both weighted hash schemes (`weighted_hrw`, `weighted_chash`) avoid degraded hosts entirely —
so **quality-awareness, not centralization, buys degraded-host avoidance** — but being
coordination-free they spread only statistically (max-load ≈3), losing to the
lightly-coordinated `heuristic` (1.4/2.0/2.0), with the Hungarian `optimal` (1.0) tighter still.
The fair `rr_random` hits degraded hosts at **every** scale (1.2/2.6/10.8), confirming that
deterministic round-robin's lucky 0-hit at 128 hosts was a naming coincidence. On **multi-hub**
topologies (16-host/3-hub, 32-host/3-hub) the numbers are identical — the findings are driven
by host quality, not graph shape. Full run saved in `findings/baseline_comparison.txt`.

### 4.10 Cross-algorithm check: DQN vs PPO (paper §V-G, Table XIV)

§4.6's load-balancing gap is measured on PPO alone, which leaves open whether it is a PPO
quirk or a property of the problem formulation. DQN (`--algorithm dqn`) is trained and
evaluated through the same `BackupSelectionEnv`, reward, and inference path as PPO, so only
the learner differs: off-policy replay instead of on-policy rollouts, value-based instead of
actor-critic, epsilon-greedy decay instead of an entropy bonus.

Both are scored **structurally** — distinct backups used, peak per-backup load, degraded-host
assignments — rather than by RPC latency, for the same reason as §4.8. These are properties
of the produced mapping, not of timing, so they are hardware-independent and reproduce
exactly across machines; §4.3 separately establishes that backup load is what drives latency
under contention. Load-balancing precision is in any case a structural property, so this
measures it directly. Reproduce with `python3 rl_algorithm_comparison.py` (no RPC servers,
about a minute).

| Scale | Algorithm | Distinct backups | Peak load | Degraded |
|---|---|---|---|---|
| 9-host | PPO | 5-5 of 5 | 1-1 | 0/25 |
| 9-host | DQN | 5-5 of 5 | 1-1 | 0/25 |
| 16-host | PPO | 10-10 of 10 | 1-1 | 0/50 |
| 16-host | DQN | 9-10 of 10 | 1-2 | 0/50 |
| 32-host | PPO | 15-19 of 20 | 2-2 | 0/100 |
| 32-host | DQN | 12-16 of 20 | 2-3 | 1/100 |
| 128-host | PPO | 24-30 of 80 | 9-28 | 0/400 |
| 128-host | DQN | **1-3 of 80** | **45-80** | **172/400** |

(5 context seeds per scale; "Degraded" is out of 5 x |P| assignments.)

**Up to 32 hosts the two algorithms behave alike.** Both spread across nearly every available
backup at 9 and 16 hosts, and at 32 hosts DQN stays within a few backups of PPO at comparable
peak load. But DQN only reaches that after the same class of fix PPO needed at 128 hosts:
with SB3 defaults it collapsed to 2-4 distinct backups of 20, and widening exploration
(fraction 0.5→0.7, final eps 0.05→0.15), enlarging the replay buffer (50k→100k), and training
longer (100k→150k steps) recovered the spread reported above. That mitigated checkpoint is
`rl_model_dqn_scaled32_mitigated.zip`.

**At 128 hosts DQN concentrates far harder than PPO**: 1-3 distinct backups of 80, up to 80
primaries on a single host, and 172 of 400 assignments on degraded hosts. PPO in the same
comparison avoids degraded hosts entirely while still using only 24-30 backups, against the
heuristic's 74-76 at max load 2 (§4.8).

**Why this matters:** two architecturally different RL algorithms concentrate load the same
way as the action space grows, and both respond to the same widened-exploration remedy. That
points at the problem formulation — a reward that labels host quality explicitly but
discourages concentration only implicitly — rather than at any one algorithm. Full run in
`findings/rl_algorithm_comparison.txt`.

**Note on the reference DQN artifacts.** The `findings/dqn_*.txt` files are RPC-latency runs
of the same checkpoints from a separate machine, kept for reference. Their absolute latencies
are not comparable to §4.1-§4.7 (different hardware), which is precisely why the paper's
cross-algorithm claim rests on the structural table above instead. The structural outcome
matches across both machines: PPO 24-30 distinct backups at 128 hosts here vs 24-29 there,
DQN 1-3 vs 3.

### 4.11 Bare-metal validation at 128 hosts (paper §V-F, Tables XI–XIII)

§4.6's latency table was produced on one machine and is withdrawn (see the notice there).
This section is that scenario repeated on real hardware.

**Testbed.** Eleven bare-metal CloudLab nodes at the Utah cluster (`c6525-25g`: one 16-core
AMD EPYC 7302P, 128 GB, 25 GbE, Ubuntu 22.04). Ten carry the 128 backup servers, thirteen
per node except the last; the eleventh runs the client alone. Each server is launched under
`taskset` on a core of its own. Hyperthread siblings on this part pair as (0,16), (1,17),
…, so logical CPUs 2–14 are **fourteen distinct physical cores** and no two servers share
one core's two threads. `provision_remote.py --verify` reports the affinity each server
actually holds, so the "one core each" claim is checked rather than asserted. All
replication traffic crosses the 25 GbE experiment LAN, confirmed with `ethtool`.

**Three runs, each changing one variable.**

#### 4.11.1 DEFT as shipped: the accept queue bounds p99

| method | p95 (ms) | p99 (ms) |
|---|---|---|
| optimal | 64.94 | 1040.06 |
| heuristic | 66.79 | 1042.47 |
| round_robin | 66.67 | 1039.96 |
| random | 65.49 | 1051.67 |

Four methods whose p95 differs by under 2 ms have p99 within 12 ms of 1040 ms. A constant
surviving every change of method is an artifact, and one second is Linux's initial TCP
retransmission timeout. §3.1 and §4.6 already identified DEFT's five-deep accept backlog as
the cause; the testbed permits the direct test, by changing only that integer and reading
the kernel's own counters.

```
node1   TcpExtListenOverflows  103209      TcpRetransSegs  116
node5   TcpExtListenOverflows   24705      TcpRetransSegs  225
node10  TcpExtListenOverflows    1923      TcpRetransSegs  141
```

`ListenDrops` equals `ListenOverflows` exactly on every node: the connections were refused
at the server, not lost in the network. The gradient across nodes tracks how heavily each
node's backups were selected, so the artifact is *correlated with* the quantity being
measured — which is why it cannot simply be noted and ignored.

Raising the backlog to 512 (`provision_remote.py --backlog 512`, which sets
`SimpleXMLRPCServer.request_queue_size` before the server binds, leaving DEFT's own code
untouched) drove the overflow count to **zero** and p99 from 1040 ms to 79 ms.

#### 4.11.2 Homogeneous hosts: no method separates from random

5 context seeds × 3 trials, 200 calls/primary, burst=10, 128 hosts, 80 primaries, 0 degraded.

| method | mean (ms) | median | p95 | p99 | fail % |
|---|---|---|---|---|---|
| random | 41.55 | 40.72 | 65.47 | 79.24 | 0.00 |
| optimal | 41.73 | 40.97 | 65.71 | 78.81 | 0.00 |
| heuristic | 41.80 | 40.87 | 65.54 | 80.57 | 0.00 |
| round_robin | 41.97 | 41.10 | 66.05 | 80.09 | 0.00 |
| rl | 46.53 | 46.05 | 71.40 | 89.26 | 0.00 |
| concentrated | 56.27 | 56.28 | 83.43 | 114.69 | 0.00 |

Paired Wilcoxon, Holm-corrected: `optimal` vs `random` has a median per-run difference of
**0.002 ms** (p = 1.000); `optimal` vs `round_robin` and `optimal` vs `heuristic` are also
n.s. Only `concentrated` and `rl` separate. This is the expected result and is worth
stating plainly: where every candidate is equivalent, a selector that reads node features
has nothing to read.

The backup-load sweep tells the same story about the backlog:

| primaries sharing one backup | mean @ backlog 5 | mean @ backlog 512 |
|---|---|---|
| 1 | 50.3 | 41.9 |
| 16 | 219.0 | 42.4 |
| 45 | 342.6 | 49.7 |
| 79 | 378.2 | 56.4 |

**7.5× becomes 35%.** Most of the apparent penalty for sharing a backup was connection
setup being refused rather than queueing at the backup. This sharpens §4.6's claim that the
shipped transport inflates the measured placement effect: the inflation is attributable to
one configurable integer.

#### 4.11.3 Degraded hosts: physical degradation, and the result that matters

The deployment refuses a `slow_hosts` argument it cannot honour (`RemoteDeployment`
raises rather than sleep inside a server it did not start), so degradation here is imposed
by `tc netem`. `tc` attaches to an interface, but each node carries thirteen backups and
shaping the interface would degrade their healthy co-tenants — the controls the comparison
depends on. Each backup owns a distinct TCP port, so `apply_degradation.py` installs a
`prio` qdisc with a `u32` filter on source port, delaying exactly the degraded servers. The
fourth band of a four-band `prio` is unreachable through the default priomap, so healthy
flows never enter the netem queue at all rather than being delayed by zero.

Verified on adjacent ports of one interface:

```
 w1 on node1 port 19000:   101.6 ms/call   DEGRADED
 w3 on node1 port 19002:     0.8 ms/call   healthy
w27 on node3 port 19000:   101.3 ms/call   DEGRADED
w28 on node3 port 19001:     0.9 ms/call   healthy
```

The 101 ms is itself informative: netem delays every packet the server sends, so a
connection-per-call protocol pays the added latency twice — once on the SYN-ACK and once on
the response. The degraded arm is therefore not numerically comparable to the simulated
50 ms sleep of §4.2/§4.5; it is a more faithful model of a congested link, not the same one.

16 of 128 hosts degraded, spread along the selection ordering (see §4.11.4):

| method | mean (ms) | p99 (ms) | calls to a degraded backup |
|---|---|---|---|
| optimal | 41.76 | 79.66 | 0.00% |
| heuristic | 42.03 | 80.99 | 0.00% |
| rl | 45.04 | 83.26 | 0.00% |
| random | 51.25 | 155.07 | 11.25% |
| round_robin | 51.87 | 157.17 | 12.50% |
| concentrated | 260.11 | 1282.18 | 100.00% |

Paired Wilcoxon on per-run mean, Holm-corrected, all with **r = −1.000** (no run dissents):
`optimal` vs `round_robin` d = −10.39, `optimal` vs `random` d = −8.11, `heuristic` vs
`random` d = 7.96, `heuristic` vs `round_robin` d = 9.28, all p_holm = 0.0009. At p99 the
gap is roughly 2× (80 ms against 156 ms).

`optimal` vs `heuristic` is **not significant** on mean (p_holm = 0.909) or p99
(p_holm = 1.000). The multi-criteria greedy scorer attains the quality of the
capacity-constrained Hungarian assignment without solving one.

**PPO avoids degraded hosts but does not spread load.** It reaches a degraded backup 0.00%
of the time, matching the deterministic methods, and is still significantly slower than
both (p_holm = 0.0009 against `heuristic`, r = 1.000). The structural measurement explains
it: PPO used **44** distinct backups in this run where `heuristic` used 112 and `random`
119. Host-quality discrimination and load balancing are separable properties, and PPO
acquired the first without the second — the same split §4.6 reports structurally and §4.10
traces to the anti-concentration term.

#### 4.11.4 A contiguous degraded block makes the baselines look better than they are

An earlier execution of this arm degraded the first sixteen backup-eligible hosts,
`w81`–`w96`, and reported `round_robin` tied with `optimal` at 41.28 ms having selected a
degraded host **0%** of the time. That tie was an artifact of naming, and it is the same
"alphabetical-sort coincidence" §4.6 flags — but it needs stating as a methodology rule,
not a curiosity.

`round_robin_mapping()` cycles a **lexicographically** sorted host list, in which every name
beginning `w1`–`w7` precedes `w8`. With 80 primaries the cursor advances 80 positions and
never reaches `w81`–`w96`. The method scored zero by never arriving, not by avoiding.
Natural-order sorting would not fix it either: `w81`–`w96` still sit beyond position 80.

`--degraded-selection spread` samples the 16 degraded hosts evenly along that same ordering
(positions 0, 8, 16, …, 120), putting them inside every method's reach. After that change
`round_robin` meets them at the base rate of 12.50% and its mean rises from **41.28 to
51.87 ms**.

**Rule:** a load-spreading baseline evaluated against a contiguous impairment can appear to
avoid it. Any scenario that marks a contiguous block of hosts as degraded must check what
share of each method's calls actually reached one, or the comparison is uninterpretable.

#### 4.11.5 The same result at 16 and 32 hosts, and what the single machine cost

The degraded scenario was repeated at 16 and 32 hosts on the same nodes, 12.5% of hosts
degraded at each (`w1 w2` and `w1 w17 w24 w31` respectively, chosen by
`--degraded-selection spread`), two and four servers per node so no node carries more than
a handful.

| method | 16 hosts | degr. | 32 hosts | degr. | 128 hosts | degr. |
|---|---|---|---|---|---|---|
| optimal | 23.17 | 0.00% | 31.27 | 0.00% | 41.76 | 0.00% |
| heuristic | 23.82 | 0.00% | 31.59 | 0.00% | 42.03 | 0.00% |
| rl (PPO) | 23.49 | 0.00% | 32.37 | 1.00% | 45.04 | 0.00% |
| round_robin | 29.89 | 10.00% | 38.65 | 10.00% | 51.87 | 12.50% |
| random | 36.45 | 18.00% | 40.88 | 15.00% | 51.25 | 11.25% |
| concentrated | 101.71 | 100.00% | 98.35 | 95.00% | 260.11 | 100.00% |

`optimal` beats `round_robin` and `random` with r = −1.000 and −0.983 at 32 hosts
(p_Holm = 0.0009 both), and r = −1.000 and −0.967 at 16 hosts. `optimal` vs `heuristic` is
n.s. at every scale. The gap between the feature-aware and ordering-based groups widens
with the pool: 6.72 ms at 16, 7.38 at 32, 10.11 at 128.

**A power note.** The 32-host run was first executed at the original reduced parameters
(`--context-seeds 1 2 3 --trials 2`, n = 6). Every effect size was r = −1.000, but with 6
pairs the smallest attainable two-sided p is 0.0312, so Holm correction across 15
comparisons left everything at p = 0.4688 — non-significant purely from sample size, as
the harness's own `[power]` warning said. Re-run at 5 seeds × 3 trials (n = 15) the same
effects clear at p_Holm = 0.0009. Both files ship; the n = 6 one is a valid replication at
the original parameters and the two agree on every ordering.

#### 4.11.6 The accept backlog's bias does not have a single sign

The 16-host scenario was also run with DEFT's shipped backlog — same nodes, same `tc`
degradation, one variable changed.

| method | mean @ 5 | median @ 5 | p99 @ 5 | mean @ 512 | median @ 512 | p99 @ 512 |
|---|---|---|---|---|---|---|
| optimal | 28.70 | 14.64 | 1015.26 | 23.17 | 22.68 | 49.45 |
| heuristic | 32.04 | 12.08 | 1020.77 | 23.82 | 23.09 | 56.76 |
| rl (PPO) | 32.46 | 11.13 | 1022.93 | 23.49 | 22.87 | 49.84 |
| round_robin | 35.22 | 14.44 | 1013.35 | 29.89 | 22.11 | 120.62 |
| random | 51.54 | 12.04 | 858.46 | 36.45 | 23.62 | 95.39 |
| concentrated | 101.70 | 101.49 | 104.30 | 101.71 | 101.52 | 104.52 |

Three things fall out of this table.

**The one-second plateau is not specific to 128 hosts.** p99 sits at 1013–1023 ms for four
methods whose p95 spans 28 to 102 ms, on hardware where nothing else is contended. It has
been bounding p99 at every scale in this paper; it was only ever visible as an artifact
once the backlog could be varied.

**The bias runs the other way at this scale.** `round_robin` minus `heuristic` is
**3.18 ms as shipped** and **6.07 ms at backlog 512**. At 128 hosts the shipped queue
*inflated* the concentration penalty (§4.11.1); here it *understates* the placement gap.
Both follow from the same mechanism: overflow is driven by how many primaries share one
backup, so at 128 hosts with 80 primaries it falls hardest on exactly the placements that
are already worst and exaggerates the spread, while at 16 hosts with 10 primaries over 16
backups nothing is concentrated enough and the penalty falls roughly evenly, compressing
it. **The shipped backlog biases a measured placement effect with a sign that depends on
how concentrated the workload is** — which is a stronger reason to report it as a
controlled variable than if it merely scaled everything.

This narrows a claim made in an earlier draft of §4.11.1, which read as though the
inflation were general. It is not; it is specific to concentrated workloads.

**A five-deep queue is admission control.** It *lowers* the median — 22.68 → 14.64 ms for
`optimal`. Calls that are accepted contend with fewer others for a single-threaded server
and finish faster; calls that are refused wait a full second. DEFT as shipped trades a
better median for a catastrophic tail, which is a property of the system rather than of
our measurement.

#### 4.11.7 What this section does not claim

- `concentrated`'s 260 ms in the degraded arm mixes concentration with degradation, because
  the host it piles onto (`w1`) is itself degraded. The clean figure for concentration alone
  is the homogeneous column: 56.27 against 41.73, a 35% penalty.
- §4.2 and §4.5 (16 and 32 hosts) have since been re-run here (§4.11.5) and the ordering
  holds. Their single-machine *magnitudes* are still affected by the shipped backlog, in
  the direction §4.11.6 measures — which is not the direction an earlier draft assumed.
  §4.1 (9 hosts) remains a single-machine measurement.

---

### 4.12 Online selection under churn (paper §VI, Tables XX–XXV)

Every scenario above is one-shot: given the cluster as it is, choose a backup. This section
asks what *adaptation* is worth when hosts degrade, recover and fail over time and moving a
replica costs a state transfer. It needs no RPC and runs on a laptop in about 3.5 minutes
(`competitive_analysis.py`).

**Model.** Each round, every host's health, capacity and latency drift; a two-state Markov
chain degrades and recovers hosts, and a smaller rate kills them. A policy pays a service
cost per round for the backup it uses, and a migration cost `M` whenever it moves one. The
comparator is the **exact** offline optimum, computed by dynamic programming over the
time×host trellis in O(k) per round using the uniform-switching-cost trick, so it is a real
lower bound rather than another heuristic.

**Re-solving every round is unboundedly bad.** `snapshot` — the policy that plays a
cost-minimizing assignment at every round, which is exactly what §4.2's `optimal` would do
if deployed and re-run each round — is not ρ-competitive for any ρ (paper Theorem 1). Two
hosts alternate as the better choice by a vanishing margin ε; snapshot chases the margin and
pays `M` each time while the optimum stands still and pays ε per round. `test_online.py`
implements the proof's construction and measures the predicted 1/ε rate:

| ε | measured ratio |
|---|---|
| 0.1 | 5.9 |
| 0.01 | 59.4 |
| 0.001 | 594 |
| 0.0001 | 5940 |

The single-primary case reduces to a Metrical Task System on a uniform metric, so the
Borodin–Linial–Saks deterministic bound and the Karloff–Rabani–Ravid randomized bound
transfer by citation.

**The main empirical result is negative.** At the most volatile churn setting, with hard
failures enabled (excess over the offline optimum; 1.00 = optimal):

| policy | what it does | excess ratio |
|---|---|---|
| static | never moves | 42.25 |
| failover | moves only on hard failure | 4.64 |
| hysteresis (α=2) | full degradation-aware migration | 4.59 |
| snapshot | re-solves every round | unbounded |

Failover alone closes **91.2%** of the distance to the offline optimum. Adding full
degradation-aware hysteresis on top closes a further **0.13 percentage points**. Nearly all
the value of adapting comes from the simplest possible rule, and very little from
sophisticated degradation tracking — the same shape as §4.11.2, where the cheap policy
captures almost everything.

Note that `static` is only this bad because it sits on *dead* replicas forever; an earlier
draft compared against it alone and overstated the case for adaptation. `failover` is the
honest baseline and is reported as such.

**A certified migration budget (paper Proposition 2).** A hysteresis policy that migrates
only when accumulated regret exceeds α·M cannot spend more than 1/α of its service cost on
migration. This is a proof, and it holds at every threshold measured:

| α | excess ratio | budget used | bound (1/α) | migrations |
|---|---|---|---|---|
| 0.25 | 2.885 | 0.583 | 4.000 | 286.8 |
| 0.5 | 1.774 | 0.257 | 2.000 | 82.9 |
| 1.0 | 1.630 | 0.134 | 1.000 | 44.0 |
| 2.0 | 1.593 | 0.082 | 0.500 | 27.6 |
| 4.0 | 1.536 | 0.049 | 0.250 | 16.7 |
| 8.0 | 1.567 | 0.031 | 0.125 | 11.0 |

The tightest observation is at roughly half the permitted spend. The practical reading: an
operator can fix a migration budget in advance and the policy will respect it.

Migrations take effect at `t+1`, not within the round that triggers them. This is both the
physically correct semantics — a primary cannot retroactively have state on the new backup
— and necessary for the budget proposition: with same-round effect a policy can accumulate
regret on a host and then dodge paying for it, and the bound fails at α=4.

---

### 4.13 Migration cost of real network-function state (paper §VI-F, Tables XXVI–XXIX)

Every number in §4.12 depends on `M`, the cost of moving a replica in rounds of ordinary
replication. The model assumed `M = 3`. This section measures it
(`measure_migration_cost.py`).

**Why it could not be measured before.** DEFT's backup receives state and discards it — the
persistence code is commented out and `update_state` does nothing but print. There was no
table, so there was no size. We therefore built one: `flow_state.py` implements a
connection-tracking / NAT flow table keyed on the 5-tuple, with incremental delta
replication, TCP state tracking and idle eviction, driven by the flow-arrival workload in
`workload.py`.

**Workload validation.** Little's law predicts steady-state occupancy as arrival rate ×
(mean lifetime + idle timeout), independently of the implementation:

| flows/s | measured entries | Little's law | delta/round | delta share |
|---|---|---|---|---|
| 500 | 15,656 | 15,075 | 521 | 3.3% |
| 2,000 | 62,045 | 60,298 | 2,053 | 3.3% |
| 8,000 | 248,272 | 241,193 | 8,214 | 3.3% |

Agreement within 3.3%. This is a check on the generator, not a second result.

**Wire encoding.** The same table serialized three ways; `xmlrpc` is what DEFT puts on the
wire.

| encoding | bytes/entry | full table @ 2,000 flows/s | vs packed |
|---|---|---|---|
| packed | 44.0 | 2.7 MB | 1.00× |
| json | 74.8 | 4.7 MB | 1.70× |
| xmlrpc | 321.8 | 20.0 MB | 7.31× |

A correctness finding, not only a size one: XML-RPC's `<int>` is a **signed 32-bit**
integer, so an IPv4 address of 128.0.0.0 or above does not fit and raises `OverflowError`.
Any NAT mapping in the upper half of the address space cannot be sent as an integer at all
and must go as text. That, plus per-value markup, is where most of the 7.31× comes from.

**M itself**, at 2,000 flows/s:

| encoding | link | full migration | service/round | M (rounds) |
|---|---|---|---|---|
| packed | 1 Gb/s | 184.9 ms | 4.71 ms | 39.2 |
| packed | 25 Gb/s | 163.9 ms | 4.01 ms | 40.9 |
| json | 10 Gb/s | 342.5 ms | 9.15 ms | 37.4 |
| xmlrpc | 10 Gb/s | 2143.4 ms | 92.53 ms | 23.2 |

**M barely depends on the encoding or the link**, which looks wrong and is the most useful
thing here. A costlier encoding inflates the full transfer *and* the per-round delta by the
same factor, so the ratio is nearly unchanged. `M ≈ 1 / delta-share ≈ 1/3.3% ≈ 30` is a
property of how fast the traffic churns, not of serialization or NIC speed. M cannot be
engineered away by choosing a better wire format.

**What this corrected.** At the assumed `M = 3`, migrating away from a degraded backup
looked cheap and the degradation-aware policy looked worthwhile. At the measured
`M = 23–41` the round trip — leave, then return when the host recovers — costs `2M`, and
degradation must last a long time before that pays. We revised the conclusion rather than
the measurement. The ordering is stable at the top of the measured range; at the bottom
(`M = 7.7` under an earlier parameterization) static leads at four of six durations, is
level at one and behind by 0.01 at one, which is stated precisely rather than rounded in
our favour.

---

## 5. Threats to validity / limitations

- **RPC-isolation testbed, not the physical Mininet network.** The Mininet path
  (`test.py`) was attempted first and hit a sequence of environment failures on this
  machine (WSL2 lacks the `openvswitch` kernel module and the `htb` traffic-control qdisc;
  no SDN controller was ever included in the repo; a required `stamperScript.py` component
  referenced by `test.py` has no git history — it was never actually committed). After
  fixing what was fixable (switched to OVS userspace datapath and a built-in learning-switch
  controller, built the missing Python venv), the deciding factor to move to the RPC-only
  testbed was that `TCLink`'s bandwidth throttling never actually engaged on this kernel,
  so there was no real resource constraint for contention to manifest against over the
  physical network path — the RPC-isolation testbed, by directly exercising the *actual*
  backup RPC server under controlled concurrent load, gives a more direct and reproducible
  measurement of the mechanism in question, at the cost of not validating full end-to-end
  network behavior (switching, real link bandwidth, packet loss).
- **Single physical machine below 128 hosts; bare metal at 128.** The 9-, 16- and 32-host
  results are processes on one laptop; OS-level scheduling noise is a real and visible
  component of the tail latency (see the p99 clusters consistent with TCP retransmission
  backoff in raw data). Multi-seed aggregation and large sample sizes mitigate this but
  don't eliminate it. At 128 hosts it fails outright, which is why that table is withdrawn
  (§4.6). §4.11 removes the limitation at the scale where it binds: 128 backups on ten
  bare-metal nodes over a 25 GbE LAN, one physical core each with the affinity verified,
  and degradation imposed by `tc netem` rather than a sleep.
- **What the bare-metal run revised, and what the smaller scales inherit.** §4.11 changed
  more than it confirmed. (1) Most of the apparent cost of concentrating backups was DEFT's
  five-deep accept queue, not load on the backup: holding everything else fixed and raising
  the backlog took the 1→79 sweep from 7.5× to 35%. (2) With homogeneous hosts, no
  selection method separates from random. (3) A load-spreading baseline that appeared to
  avoid degraded hosts was missing them by lexicographic accident (§4.11.4). The claim that
  survived all three is the load-bearing one: feature-aware selection avoids degraded hosts
  and ordering-based selection does not. §4.2 and §4.5 (16 and 32 hosts) **have** since been
  re-run on bare metal (§4.11.5) and the ordering holds at both. Their single-machine
  magnitudes are still affected by the shipped backlog, but not in the direction an
  earlier draft assumed: §4.11.6 measures the bias at 16 hosts running the *opposite*
  way, understating the placement gap rather than inflating it, because overflow tracks
  concentration and nothing is concentrated enough at that scale. §4.1 (9 hosts) has not
  been re-run.
- **The degraded arm's mechanism changed with the hardware.** §4.2/§4.5 impose degradation
  as a 50 ms sleep inside the server. §4.11.3 imposes it with `tc netem` on the server's
  source port, which delays every packet the server sends; a connection-per-call protocol
  therefore pays it twice (101 ms measured, not 50). The two arms are not numerically
  comparable, and the physical one is the more faithful model of a congested link.
- **`M` was assumed before it was measured.** The online analysis (§4.12) was first run at
  `M = 3`. The measurement (§4.13) puts it at 23–41, which overturned an earlier conclusion
  of ours about when migrating away from a degraded backup pays. Any result in §4.12 should
  be read against the measured range, not the original assumption.
- **The online model is a model.** §4.12's churn process (two-state Markov chain, Beta
  severity, truncated-Pareto lifetimes) is parameterised plausibly rather than fitted to
  operator traces, and the workload in §4.13 likewise. The flow table is validated against
  Little's law, which checks the generator's internal consistency but not its realism.
  `workload.replay_trace()` accepts a pcap and would close this; no trace was available.
- **Synthetic resource values.** Health/capacity/CPU/storage are drawn from configured
  ranges, not measured from real workload telemetry.
- **RL trained per-topology, and needs scale-dependent hyperparameters.** The PPO policy's
  observation/action space size is fixed to a given host count; a differently-sized
  topology needs a freshly trained model (`rl_model.zip` for 9 hosts,
  `rl_model_scaled.zip` for 16, `rl_model_scaled32.zip` for 32, `rl_model_scaled128_v3.zip`
  for 128). At 128 hosts, SB3's raw PPO defaults (`ent_coef=0.0`, `net_arch=[64,64]`)
  produce a degenerate, collapsed policy; the code's own stronger defaults
  (`net_arch=[128,128]`, `ent_coef=0.01`) that suffice at 9/16/32 hosts only partially fix
  it, and a still-larger configuration (`ent_coef=0.02`, `net_arch=[256,256]`, larger
  rollout/batch, 1M timesteps) was needed for full degraded-host avoidance across every
  seed tested — but even that fixed policy still load-balances worse than
  `heuristic`/`optimal` under real contention (§4.6: mean 458ms vs ~48ms). **The
  unresolved limitation is precise load-balancing at 128-host scale, not host-quality
  discrimination** — a more specific claim than "RL doesn't scale," and one a reward
  function that directly penalizes load concentration (rather than relying on the
  existing health/capacity decay to implicitly teach it) would plausibly close, but that
  redesign was not attempted here.
- **RL results are tied to the released checkpoints and library versions.** The `rl` rows
  describe the specific trained models shipped in `code/*.zip`; retraining even at the same
  seed under a different `stable-baselines3`/PyTorch version, or at a different host count,
  yields a different policy. The released models were produced with the versions pinned in
  `requirements.txt` — reproduce against those for numbers comparable to the tables above.
- **Round-robin's degraded-host exposure is coincidental, not a stable property.**
  `round_robin_mapping()`'s cycling interacts with Python's alphabetical (not numeric)
  sort of host names in a way that happens to hit the degraded range at 16/32 hosts
  (§4.2/§4.5) but misses it entirely at 128 hosts (§4.6). Conclusions about round-robin's
  *relative* latency should be read as "uncontrolled/coincidental exposure to bad hosts,"
  not "consistently ~20% worse" — the consistent, load-bearing claim is that
  heuristic/optimal/RL avoid degraded hosts *by construction* in every scenario tested,
  which round-robin cannot guarantee in any of them.
- **Retry-policy inconsistency across scale points.** The 128-host run required hardening
  `replicate_burst()`'s retry budget (3 → 8 attempts, plus jittered backoff and a
  failure-instead-of-crash path) after the original budget crashed under 80-primary
  contention. The 9/16/32-host results were generated before this fix but never exhausted
  the original 3-attempt budget in practice, so they should be unaffected — but they were
  not re-run under the identical harness version, which is a methodology inconsistency a
  careful reviewer would flag.
- **Scale sweep is three points (16/32/128), not a continuous curve**, and trial counts
  differ across them (32 hosts used a reduced 3×2×150/burst=8 to stay tractable; 16 and
  128 hosts both used the full 5×3×200/burst=10). The qualitative finding (feature-aware
  methods avoid degraded hosts; concentrated degrades sharply with scale) holds at all
  three, but a continuous sweep with fully matched trial counts throughout would be needed
  for a rigorous asymptotic scalability claim.
- **Multi-seed generalization covers resources, not topology structure.** The
  `--context-seeds` sweep in §4.1/§4.2 varies the synthetic resource draw (health,
  capacity, CPU, storage) independently per seed, which is what drives the
  degraded-vs-healthy host distinction the results turn on. It does *not* vary the
  underlying graph shape: `generate_scaled_topology()`'s own `seed` parameter is currently
  unused (the star/hub structure is deterministic for a given `n_hosts`/`n_primaries`).
  The results therefore generalize across resource realizations on one topology shape,
  not across topology shapes themselves — a genuinely different graph (e.g. multiple
  hubs, uneven host-to-switch fan-out) is untested.

## 6. Reproducibility

```bash
cd code
source ../venv/bin/activate   # or use ../venv/bin/python3 directly

# Homogeneous scale, multi-seed
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl

# Scaled + heterogeneous, multi-seed
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 \
    --rl-model rl_model_scaled.zip

# Scaled + heterogeneous, 32 hosts (scalability trend point)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 \
    --rl-model rl_model_scaled32.zip

# Scaled + heterogeneous, 128 hosts (scalability trend point; see §4.6 for the rl story)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --rl-model rl_model_scaled128_v3.zip

# Selection-time overhead (all four topology sizes)
python3 selection_overhead.py

# Heuristic weight sensitivity + decay calibration (§4.8; no RPC, runs in seconds)
python3 weight_sensitivity.py
python3 decay_sensitivity.py

# Decentralized hash-based baselines: HRW comparison (§4.9; no RPC, runs in seconds)
python3 baseline_comparison.py

# Cross-algorithm check: DQN vs PPO (§4.10). Note the reported numbers come from a
# different machine, so expect different absolute latencies; the pattern is what reproduces.
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 \
    --methods heuristic optimal rl rl_dqn \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 \
    --rl-model rl_model_scaled32.zip --rl-dqn-model rl_model_dqn_scaled32_mitigated.zip
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 --trials 3 --calls 200 --burst 10 \
    --methods heuristic optimal rl rl_dqn \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --rl-model rl_model_scaled128_v3.zip --rl-dqn-model rl_model_dqn_scaled128.zip
# Fastest way to see the 128-host DQN collapse without any RPC run (seconds)
python3 rl_backup_selector.py --check-collapse --algorithm dqn \
    --model rl_model_dqn_scaled128.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0

# Throughput and failure rate (homogeneous + 16-host scenarios)
rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl
rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip

# Online selection under churn (§4.12). No RPC, no testbed; ~3.5 minutes.
python3 competitive_analysis.py
python3 test_online.py          # proofs + the adversarial construction behind Theorem 1

# Migration cost of real NF state (§4.13). No RPC; builds a flow table and measures it.
python3 measure_migration_cost.py
python3 test_flow_state.py      # flow table, delta replication, encodings, Little's law

# Bare-metal 128 hosts (§4.11). The only step that needs hardware: 11 machines with SSH
# between them. Check nproc first -- --per-node must not exceed physical cores minus the
# reserve, and on an SMT part only one thread per physical core may be used.
python3 provision_remote.py \
    --nodes node1 node2 node3 node4 node5 node6 node7 node8 node9 node10 \
    --hosts 128 --per-node 13 --reserve-cores 2 \
    --remote-dir /path/to/code --manifest ../findings/manifest.json --dry-run
python3 provision_remote.py ... --backlog 512 --manifest ../findings/m512.json
# DO NOT SKIP: reports the affinity each server actually holds. A warning here means the
# run would measure CPU scheduling rather than placement -- the §4.6 failure mode.
python3 provision_remote.py --manifest ../findings/m512.json --verify

# Homogeneous arm (§4.11.2)
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 0 --scale-seed 0 \
    --rl-model rl_model_scaled128_v3.zip --manifest ../findings/m512.json

# Degraded arm (§4.11.3). Slow 16 hosts by source port, then tell the experiment the
# slowness is already physical so it does not also ask for a simulated delay.
python3 apply_degradation.py --manifest ../findings/m512.json \
    --hosts w1 w2 w27 w34 w41 w49 w56 w63 w70 w78 w85 w92 w106 w113 w120 w128 \
    --delay-ms 50 --iface <10/25G NIC> --apply
python3 rpc_contention_experiment.py \
    --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --degraded-selection spread --physical-degradation \
    --rl-model rl_model_scaled128_v3.zip --manifest ../findings/m512.json
python3 apply_degradation.py --manifest ../findings/m512.json --iface <NIC> --clear
# --degraded-selection spread is required: with a contiguous block, round_robin can miss
# the degraded hosts entirely through lexicographic ordering (§4.11.4).

# Regenerate the paper tables from the findings files
python3 make_testbed_tables.py
python3 make_online_tables.py
python3 make_migration_tables.py

# Retrain RL models from scratch
python3 rl_backup_selector.py --train --timesteps 50000 --model rl_model.zip
python3 rl_backup_selector.py --train --timesteps 80000 --model rl_model_scaled.zip \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0
python3 rl_backup_selector.py --train --timesteps 100000 --model rl_model_scaled32.zip \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0

# 128 hosts: SB3's PPO defaults (ent_coef=0.0, net_arch=[64,64]) collapse to a degenerate
# policy at this scale -- included for reproducibility of that finding, not as a recipe:
python3 rl_backup_selector.py --train --check-collapse --timesteps 400000 \
    --model rl_model_scaled128_collapsed.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --ent-coef 0.0 --net-arch 64 64

# 128 hosts, the fix that was actually used for the rl_model_scaled128_v3.zip results in
# §4.6: higher entropy bonus + wider network + larger rollout/batch sizes. Fixes the
# degenerate collapse and the degraded-host avoidance, but still load-balances worse than
# heuristic/optimal under real contention -- see §4.6/§5 for the honest final verdict.
python3 rl_backup_selector.py --train --check-collapse --timesteps 1000000 \
    --model rl_model_scaled128_v3.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256

# Retrain the DQN models (§4.10). Add --algorithm dqn; DQN is tuned via exploration and
# replay-buffer flags rather than --ent-coef.
python3 rl_backup_selector.py --train --algorithm dqn --timesteps 50000 --model rl_model_dqn.zip
python3 rl_backup_selector.py --train --algorithm dqn --timesteps 80000 \
    --model rl_model_dqn_scaled.zip \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0

# 32 hosts: DQN collapses to 2-4 of 20 backups on SB3 defaults. This is the mitigated
# config that lifts it to 16-19 (the rl_model_dqn_scaled32_mitigated.zip used in §4.10).
python3 rl_backup_selector.py --train --algorithm dqn --check-collapse --timesteps 150000 \
    --model rl_model_dqn_scaled32_mitigated.zip \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 \
    --exploration-fraction 0.7 --exploration-final-eps 0.15 \
    --buffer-size 100000 --target-update-interval 1000

# 128 hosts: still collapses to 3 of 80 backups even with a wide network and 1M steps --
# this is the cross-algorithm evidence in §4.10, not a config to copy.
python3 rl_backup_selector.py --train --algorithm dqn --check-collapse --timesteps 1000000 \
    --model rl_model_dqn_scaled128.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --net-arch 256 256 --batch-size 256 --buffer-size 100000 --exploration-fraction 0.5
```

Raw per-call latency data: `results_rpc/rpc_contention_raw.csv` (columns: `method,
context_seed, trial, primary, backup, backup_load, latency_ms`). Raw throughput/
failure-rate data: `results_rpc/throughput_raw.csv` (columns: `method, context_seed,
trial, primary, backup, backup_load, successful_calls, failed_calls, duration_s,
throughput_cps, failure_rate_pct`). Each `rm -f` + run command overwrites the previous
scenario's CSV — copy it aside first if you want to keep more than one scenario's raw
data at once.
