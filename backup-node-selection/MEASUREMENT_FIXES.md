# Measurement and statistics corrections

Two defects in the original evaluation harness were found and fixed. Both changed
reported conclusions, so every RPC-latency number in the paper must be regenerated
from the re-run data in `findings/raw/`.

- **Code:** `code/analysis.py` (new), `code/rpc_contention_experiment.py`,
  `code/throughput_experiment.py`
- **Checks:** `code/test_analysis.py`, `code/test_connection_reuse.py`
- **Raw data:** `findings/raw/{9,16,32,128}hosts.csv.gz`

---

## Fix 1 — latency was fused with the client's own retry policy

`replicate_burst` started its stopwatch *outside* the retry loop, so a recorded
"latency" was the backup's service time plus every failed attempt plus the client's
own `time.sleep(backoff)`. Failed calls were additionally imputed into the latency
sample as a hardcoded `FAILED_CALL_PENALTY_MS = 10000.0` sentinel, which contaminates
every percentile computed from that sample.

Each call now records three separate timings:

| column | meaning |
| --- | --- |
| `service_ms` | the successful attempt alone: real backup service + queueing delay |
| `retry_wait_ms` | failed attempts plus client backoff sleep (a client property) |
| `wall_ms` | their sum: what the old harness called "latency" |

Failures are **censored** (`succeeded=False`, no latency value) and reported as a rate.
All comparisons use `service_ms`.

### What the fix actually showed

The retry-inflation hypothesis was **wrong**, and the instrumentation is what
disproved it. Client backoff contributed essentially nothing: 0.00% of calls retried
at 9, 16 and 32 hosts, except `concentrated` at 0.09%/0.15% (16/32 hosts), worth under
30 ms at p99.

The real mechanism is worse. Service time is sharply **bimodal** (9-host, n=105,000):

| band | share of calls |
| --- | --- |
| < 100 ms | 97.46% |
| 100-500 ms | 0.15% |
| **500-900 ms** | **0.00%  (exactly zero calls)** |
| 900-1100 ms | 2.00%  (clustered at 1045.9 +/- 24.1 ms) |
| 2000-3200 ms | 0.28% |

A hard empty band with a tight cluster beyond it is not queueing delay; queueing
produces a smooth tail. It is Linux's **TCP SYN-retransmission timer** (1 s initial
RTO, then +2 s), triggered by accept-backlog overflow.

The cause is a second incorrect claim. The paper's methodology states each burst
worker "keeps one persistent connection for all its calls." It does not:
`SimpleXMLRPCServer` uses `protocol_version = HTTP/1.0` and closes after every
response. `test_connection_reuse.py` measures **20 calls producing 20 TCP accepts**.
So every RPC performs a fresh handshake against a `listen(5)` backlog.

**Consequence: p99 is a threshold statistic, not a latency measurement.** It reads
~45 ms or ~1045 ms depending only on whether that run's SYN-drop rate crosses 1%:

| scale | method | % calls > 500 ms | % in 500-900 ms gap | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| 9 | heuristic | 1.907 | 0.000 | 35.09 | 1041.22 |
| 16 | heuristic | 1.637 | 0.000 | 56.37 | 1046.06 |
| 32 | heuristic | 1.585 | 0.000 | 73.05 | 1059.66 |
| 32 | round_robin | 1.448 | 0.017 | 358.23 | 1059.94 |

This explains the reviewer's unanswered question about "the p99 strong latency dip at
32 hosts in Fig. 1". The published 32-host run used a reduced protocol (burst=8, 150
calls); that lowered concurrency pushed the SYN-drop rate under 1% and p99 fell below
the timer quantum to 105.63 ms. Re-run at the matched protocol (burst=10, 200 calls)
the same cell reads **1059.97 ms**, in line with every other scale.

**Recommendation:** report `mean` and `p95`, drop `p99` claims until the transport is
replaced. `p95` discriminates well (round_robin 358 ms vs heuristic 56-73 ms at 16/32
hosts, the degraded-host penalty); `p99` carries no information about replication.

---

## Fix 2 — pseudo-replication in the statistics

The original analysis pooled every individual call and ran Mann-Whitney U with n in
the tens of thousands. Calls inside one burst share a TCP connection, a
single-threaded server process, an OS scheduler and one resource realization, so they
are nowhere near independent. The inflated n drove p-values to zero without adding
evidence, which is why the paper had to instruct readers to ignore its own p-values.

The independent unit is a **run**: one `(method, context_seed, trial)` execution.
`analysis.py` now:

1. reduces each run to per-run statistics over its own calls (n = 15, not 240,000);
2. compares methods with a **paired** Wilcoxon signed-rank test, matched by
   `(context_seed, trial)` -- every method runs under the same conditions, so pairing
   is valid and removes between-run variance (background load, seed difficulty);
3. reports the matched-pairs rank-biserial correlation, defined on run-level
   differences;
4. applies **Holm-Bonferroni** across the pairwise family (previously absent);
5. builds confidence intervals by **cluster bootstrap**, resampling runs not calls;
6. warns when n is small enough that the minimum attainable p approaches 0.05;
7. re-aggregates the latency-vs-load table to one observation per (run x backup).

`test_analysis.py` validates each piece. The decisive check builds two methods with
**no true difference** and correlated calls within runs: the old pooled test reports
**p = 1.08e-44**, the run-level paired test reports **p = 0.76**.

### Side effects found

- `record_raw` appended to `results_rpc/rpc_contention_raw.csv` indefinitely, and the
  load-breakdown table read the whole file back, so it silently pooled calls from
  earlier runs at other scales. Each invocation now truncates first (`reset_raw_csv`).
- `throughput_experiment.py` used an unpaired Mann-Whitney with no correction on data
  that was already run-level; it now uses the same paired machinery.
- The old pooled load table showed median latency *decreasing* with load
  (17.9 -> 10.2 -> 8.1 -> 6.1 ms), an artifact of pooling unequal call counts across
  methods. Run-level aggregation makes both mean and median rise monotonically
  (28.8 / 47.6 / 73.7 / 113.8 ms at loads 1/2/3/4).

---

## Conclusions that changed

Pairing makes the test **more** sensitive, not less, so the headline result improves
while a secondary claim is overturned.

### 9 hosts (homogeneous) -- claim overturned

The paper states the feature-aware methods show "no meaningful advantage over blind
round-robin" (old `heuristic` vs `round_robin`: r = -0.029, dismissed as negligible).
Corrected, the heuristic is **consistently worse**: d = +4.08 ms, r = +0.883,
p_holm = 0.0081, losing 14 of 15 paired runs. Same against `optimal` (+3.33 ms) and
`rl` (+3.20 ms), both p_holm = 0.0081.

Mechanism: the heuristic's max per-backup load is 1.40 here against round-robin's
1.00, and load 2 costs ~19 ms over load 1. The `-20` capacity decrement under-spreads
at small scale. This is the correct, defensible statement: a small but perfectly
consistent disadvantage where the scenario offers no heterogeneity to exploit.

### 16 and 32 hosts (heterogeneous) -- core claim strengthened

`round_robin` vs `heuristic` was reported as r = -0.1465 ("small-to-medium"), the
weakest-sounding number attached to the paper's most important claim. Corrected:

| scale | d_median | r | p_holm |
| --- | --- | --- | --- |
| 16 hosts | -66.4 ms | **-1.000** | 0.0009 |
| 32 hosts | -64.3 ms | **-1.000** | 0.0009 |

The heuristic wins **15 of 15 paired runs at both scales**.

The three feature-aware methods are now properly indistinguishable at 16 hosts
(optimal vs heuristic, optimal vs rl, rl vs heuristic all p_holm = 1.0), so the
"significant but ignore the p-value" framing can be dropped. At 32 hosts `optimal`
does beat `heuristic` significantly but slightly (-2.5 ms, r = -0.817,
p_holm = 0.013): the heuristic tracks the Hungarian bound closely but is measurably
behind it, which is more precise than "tracks it".

`random` is revealed as unstable rather than merely mediocre: its 16-host CI is
[72.9, 182.9] ms against the heuristic's [37.4, 42.1], and `random` vs `concentrated`
is **not significant** (r = -0.300). Pooling by call hid this entirely.

### Protocol change

The 32-host point was re-run at the full 5 seeds x 3 trials x 200 calls x burst 10
protocol instead of the published 3 x 2 x 150 x burst 8. At 6 runs the corrected test
cannot reach significance at all (minimum attainable p = 0.031, times 15 Holm
comparisons = 0.47). Matching the protocol also removes the paper's own
"reduced trial count" and "retry-policy inconsistency across scale points" threats to
validity, and it is what exposed the p99 dip as a configuration artifact.

---

## Fix 3 — the 128-host run deadlocks

`xmlrpc.client.ServerProxy` applies no socket timeout. On a retry the old proxy is
discarded, leaving a connection the server has accepted but from which no request will
ever arrive. DEFT's backup is a single-threaded `SimpleXMLRPCServer`, so it blocks
forever reading that dead connection, the accept loop stops, and every other client
queues behind one wedged socket indefinitely.

Observed at 128 hosts (80 primaries on one backup), sampled over 80 s with the process
alive and 24 minutes into the method:

```
t= 20s  server_cpu=8s  syn_sent=87  estab=20
t= 40s  server_cpu=8s  syn_sent=87  estab=20
t= 60s  server_cpu=8s  syn_sent=87  estab=20
t= 80s  server_cpu=8s  syn_sent=87  estab=20
```

Server CPU time, SYN-SENT count and established count all frozen: zero forward
progress. Not the degraded-host delay -- `concentrated` maps to `w2` in every seed,
never a degraded host.

**Implication for the paper:** the published 128-host `concentrated` figures (mean
772.41 ms, p99 18,971.94 ms) came from a harness that could hit this wedge, and the
claim that four of five seeds "completed every call" describes luck rather than a
property of the system. That row is not reliably reproducible as published.

**Fix:** a client-side socket timeout via a custom `xmlrpc` transport
(`CALL_TIMEOUT_S = 20 s`), above the largest genuine service time observed anywhere
(7.2 s), so it bounds the deadlock without censoring real slow calls. This changes the
benchmark client only; the DEFT backup under test is still unmodified.

---

## Fix 4 — results depend on what ran before them

Two independent executions of the identical 9-host configuration produced absolute
means ~17 ms apart on **every** method (optimal 29.70 -> 45.90, round_robin
29.25 -> 47.51). The cause is ephemeral-port exhaustion: the server is HTTP/1.0, so
every RPC opens a fresh connection, and TIME_WAIT accumulates across runs. Measured
mid-session: **14,038 sockets in TIME-WAIT against an ephemeral range of 28,231 ports**,
about half the port space.

Absolute latencies in this testbed are therefore not reproducible quantities; they
encode the machine's socket state at the time. The published absolute tables should be
reported as paired differences, or accompanied by this caveat.

The paired run-level design absorbs it, because the drift is common-mode:

| quantity | run 1 | run 2 | drift |
| --- | --- | --- | --- |
| absolute mean, optimal | 29.70 | 45.90 | **+16.20** |
| absolute mean, round_robin | 29.25 | 47.51 | **+18.26** |
| heuristic - round_robin | 4.39 | 3.59 | -0.80 |
| heuristic - optimal | 3.94 | 5.20 | +1.26 |
| heuristic - rl | 4.14 | 2.66 | -1.48 |

A 17 ms common-mode shift with differences stable to about 1 ms. The old pooled-by-call
analysis reported absolute distributions and would have read this drift as signal.

---

## Replication: which conclusions survive

Every scale was run twice, independently. The resolution floor of the testbed is set by
the ~17 ms common-mode drift above.

| effect magnitude | replicates? |
| --- | --- |
| >= ~40 ms (feature-aware vs blind or naive) | **yes -- r ~ 1.0, p ~ 0.001 in every run** |
| <= ~5 ms (heuristic vs optimal vs rl) | **no -- sign and significance both unstable** |

### Supported

Feature-aware selection beats blind load-spreading once hosts differ in quality.

| comparison | 16h run 1 | 16h run 2 | 32h run 1 | 32h run 2 |
| --- | --- | --- | --- | --- |
| round_robin vs heuristic | +66.4, r=1.000 | +48.0, r=0.983 | +64.3, r=1.000 | +43.7, r=1.000 |
| round_robin vs optimal | +68.8, r=1.000 | +55.9, r=1.000 | +65.7, r=1.000 | +43.1, r=1.000 |
| round_robin vs rl | +66.6, r=1.000 | +55.9, r=1.000 | +64.5, r=1.000 | +29.9, r=0.983 |

All at p_holm <= 0.0012. Against the published r = -0.1465 ("small-to-medium") for the
same claim, this is far stronger.

The greedy heuristic matches the Hungarian bound and the PPO policy: `optimal` vs
`heuristic`, `optimal` vs `rl` and `rl` vs `heuristic` are non-significant at 16 and 32
hosts in both runs. Correctly so -- the testbed cannot resolve differences that small.

`random` is unstable rather than merely mediocre: 16-host CI [72.9, 182.9] ms against
the heuristic's [37.4, 42.1], and `random` vs `concentrated` is non-significant in both
runs. Pooling by call hid this.

### Not supported (two claims withdrawn after replication)

- **9 hosts, heuristic worse than round_robin.** Run 1: d=+4.08, r=+0.883,
  p_holm=0.0081 (significant). Run 2: d=+2.08, r=+0.517, p_holm=0.42 (n.s.). The
  direction is consistent but the effect is not robustly significant. The paper's
  original "no meaningful advantage" conclusion stands at this scale, though because the
  scenario is underpowered for a ~3 ms effect, not for the reason the paper gives.
- **32 hosts, optimal better than heuristic.** Run 1: d=-2.5, r=-0.817, p_holm=0.013
  (significant). Run 2: d=+3.8, r=+0.100, p_holm=0.76 (n.s.), sign flipped. The
  heuristic is indistinguishable from the optimum here.

Both withdrawn claims were small effects read out of a single run. Two independent
replications per scale should be the standard for this testbed.

---

## 128 hosts: where the retry decomposition actually matters

At 9, 16 and 32 hosts client backoff contributes nothing (0.00% of calls retry). At 128
hosts it dominates the worst case:

| method | service p99 | retry p99 | wall p99 | wall - service | retried % |
| --- | --- | --- | --- | --- | --- |
| heuristic | 1203.12 | 0.03 | 1203.12 | 0.00 | 0.00 |
| optimal | 1263.54 | 0.03 | 1263.54 | 0.00 | 0.00 |
| rl | 7210.80 | 0.03 | 7281.93 | 71.13 | 0.27 |
| **concentrated** | **15485.05** | **24597.10** | **35081.66** | **19596.61** | **3.53** |

For `concentrated`, **19.6 s of the 35.1 s wall p99 is the client's own `time.sleep()`**,
56% of the figure. The paper describes its published p99 of 18,971.94 ms as "roughly
19 s of genuine queueing delay". That is wrong: genuine service time is 15.5 s and the
remainder is client backoff. Only the decomposition separates them.

With the socket timeout in place the run completes all 15 runs with **zero**
retry-exhaustion failures, where the original harness crashed outright.

### Results (run-level, n=15)

| method | mean (ms) | 95% CI | vs heuristic |
| --- | --- | --- | --- |
| round_robin | 156.64 | [147.22, 166.78] | n.s. |
| heuristic | 167.05 | [152.06, 189.09] | -- |
| optimal | 189.25 | [153.21, 252.63] | n.s. |
| random | 214.96 | [200.43, 232.83] | +46.6, r=1.000, p=0.0009 |
| rl (PPO) | 328.86 | [296.21, 368.31] | +145.3, r=1.000, p=0.0009 |
| concentrated | 536.87 | [503.12, 577.55] | +360.6, r=1.000, p=0.0009 |

- **PPO's collapse at 128 hosts is real and robust**: 2x the heuristic's mean, losing
  all 15 paired runs. The paper's central negative result stands.
- **round_robin is indistinguishable from the heuristic here** (p=0.66), as the paper
  states: its alphabetical sort coincidentally misses the degraded hosts at this scale.
- **heuristic matches the Hungarian bound** again (p=0.68).

---

## Summary of claims after re-measurement

| claim | status |
| --- | --- |
| Feature-aware selection beats blind load-spreading under heterogeneity | **Strongly supported** -- r ~ 1.0, p ~ 0.001, replicated at 16 and 32 hosts |
| Greedy heuristic matches the Hungarian optimum | **Supported** as a null at every scale, both runs |
| PPO degrades badly at 128 hosts | **Supported** -- r = 1.000, d = +145 ms |
| `random` is unstable, not merely mediocre | **Supported** -- n.s. vs `concentrated` in both runs |
| 9-host: heuristic worse than round-robin | **Withdrawn** -- did not replicate |
| 32-host: optimal better than heuristic | **Withdrawn** -- did not replicate, sign flipped |
| p99 as a latency metric | **Invalid** -- threshold statistic, zero calls in the 500-900 ms band |
| Absolute latency values | **Not reproducible** -- ~17 ms common-mode drift from TIME_WAIT |
