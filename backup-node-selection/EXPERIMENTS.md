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
- **Single physical machine.** All "hosts" are processes on one laptop; OS-level scheduling
  noise is a real and visible component of the tail latency (see the p99 clusters
  consistent with TCP retransmission backoff in raw data). Multi-seed aggregation and
  large sample sizes mitigate this but don't eliminate it.
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
```

Raw per-call latency data: `results_rpc/rpc_contention_raw.csv` (columns: `method,
context_seed, trial, primary, backup, backup_load, latency_ms`). Raw throughput/
failure-rate data: `results_rpc/throughput_raw.csv` (columns: `method, context_seed,
trial, primary, backup, backup_load, successful_calls, failed_calls, duration_s,
throughput_cps, failure_rate_pct`). Each `rm -f` + run command overwrites the previous
scenario's CSV — copy it aside first if you want to keep more than one scenario's raw
data at once.
