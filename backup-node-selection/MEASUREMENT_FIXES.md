# Measurement and statistics corrections

Eight defects in the evaluation harness were found and fixed, in the order they were
discovered. Every one of them changed a reported conclusion, and several invalidated
results that had already been written down. They are kept here in sequence rather than
folded away, because the sequence is the point: each fix was only visible once the
previous one stopped dominating the measurement.

Fixes 1–4 concern the statistics and the harness. Fix 5 makes the RPC transport a
controlled variable. Fixes 6–8 came from moving to real hardware, and are the ones that
revised the most: the shipped accept backlog accounted for most of the apparent cost of
concentrating backups, and a contiguous degraded block had made a load-spreading baseline
appear to avoid an impairment it merely never encountered.

- **Code:** `code/analysis.py`, `code/rpc_contention_experiment.py`,
  `code/throughput_experiment.py`, `code/grpc_transport.py`, `code/deployment.py`,
  `code/backup_server.py`, `code/provision_remote.py`, `code/apply_degradation.py`
- **Checks:** `code/test_analysis.py`, `code/test_connection_reuse.py`,
  `code/test_deployment.py`
- **Raw data:** `findings/raw/{9,16,32,128}hosts.csv.gz`, and the bare-metal runs in
  `findings/latency_128hosts_cloudlab*.txt`

The claim-status table partway down this document reflects the state after Fix 4;
**the authoritative one is at the end**, after bare-metal validation.

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

*State after Fix 4. Superseded by the table at the end of this document, which reflects
Fixes 5–8 and the bare-metal validation.*

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

---

## Fix 5 — the transport was never a controlled variable

All three reviewers questioned whether the measured contention was a property of
NFV replication or of Python's XML-RPC. It was largely the latter, and the only way
to show that is to vary the transport while holding everything else fixed.

`grpc_transport.py` adds a second arm: the same `update_state(state)` call over
gRPC/HTTP/2, one persistent channel per primary instead of a connection per call,
and a server thread pool instead of a single accept loop. Selection methods,
scoring, topologies, degraded-host injection, workload, retry policy and statistics
are identical across arms. Select with `--transport {xmlrpc,grpc}`; `--grpc-workers 1`
reproduces the XML-RPC backup's strict serialization over HTTP/2, separating the
transport change from the concurrency change.

### The pathology is entirely a transport artifact

16-host scenario, 180,000 XML-RPC calls vs 16,000 gRPC calls of the same workload:

| | XML-RPC | gRPC |
| --- | --- | --- |
| p50 | 49.95 ms | **11.58 ms** |
| p95 | 360.57 ms | **88.70 ms** |
| p99 | 1175.83 ms | **105.14 ms** |
| p99.9 | 7261.61 ms | **126.86 ms** |
| max | 22773.07 ms | **156.43 ms** |
| calls on the SYN-retransmit timer | 1.549% (n=2789, sd 24 ms) | **0** |
| calls > 1100 ms | 1.650% | **0** |

p99 is a usable metric on the gRPC arm.

### The findings hold on both transports

`round_robin` vs `heuristic`, median per-run difference (positive favours the heuristic):

| hosts | XML-RPC (n=15) | gRPC (n=25) |
| --- | --- | --- |
| 9 (homogeneous) | -2.08, r=-0.517, p=0.4163 | **-2.14**, r=-0.422, p=0.5010 |
| 16 | +48.05, r=+0.983, p=0.0012 | +11.43, r=+0.705, p=0.0104 |
| 32 | +43.69, r=+1.000, p=0.0009 | +9.75, r=+0.994, p<0.0001 |

The homogeneous null agrees to 0.06 ms across two different RPC stacks and
independent executions.

### The shipped transport inflates the effect ~4.3x

48.05/11.43 = 4.2x and 43.69/9.75 = 4.5x. A degraded host's modest per-call service
penalty pushes its five-deep accept queue into overflow, converting it into whole
seconds of kernel retransmission wait. In *proportional* terms the advantage is
preserved or larger on gRPC (31.8% at 16 hosts on both; 53.7% vs 25.7% at 32),
because baseline latencies fall by a similar factor.

Generalizable claim: evaluating placement policies over a connection-per-call RPC
stack inflates measured placement effects.

### 128 hosts is not measurable on one machine

80 primaries x burst 10 = 800 client threads, plus ~75 backup processes x 8 gRPC
workers = ~600 server threads, on 8 cores. Two independent executions of the
identical gRPC configuration:

| method | run 1 | run 2 | ratio |
| --- | --- | --- | --- |
| optimal | 60.97 | 12.65 | **4.8x** |
| rl (PPO) | 105.48 | 17.18 | **6.1x** |
| round_robin | 45.79 | 78.21 | 1.7x |
| heuristic | 21.95 | 24.36 | 1.1x |
| concentrated | 561.33 | 518.08 | 1.1x |

Only gross concentration survives (r=1.000 on both transports, both executions).
The paper reports structural results only at this scale. Real distributed hardware
is required to validate latency there.

**That hardware was obtained; see "128 hosts: now measured" below.** The measurement
confirmed the structural reading and revised two magnitudes substantially, including the
concentration penalty quoted in this section.

### Effects that did not survive replication

Four candidate effects reached significance in a single execution and vanished on
re-running: the 9-host heuristic disadvantage, the 32-host optimal-beats-heuristic
result, 9-host heuristic-vs-random on gRPC at n=15, and 16-host
round_robin-vs-heuristic at n=15 (which then *did* hold at n=25). The harness
resolution floor is about 10 ms; no claim below it is made. Two independent
replications per configuration is the working standard.

---

## Fix 6 — three harness defects a single machine could not expose

Moving to a testbed required separating three things the harness had treated as one:
a topology host's *identity*, the *process* serving it, and its *network address*. The
old harness started its own servers with `multiprocessing.Process` and addressed them at
`127.0.0.1:base_port+i`; `deployment.py` splits these into `Endpoint`, `LocalDeployment`
and `RemoteDeployment`, with the local path's port assignment byte-identical to the old
rule so every published single-machine result reproduces unchanged.

Three defects surfaced on first contact with real machines. All three were invisible
locally, because the local path never goes through SSH:

1. **SSH hung on the very first server.** `cd X && nohup Y &` backgrounds the *whole
   AND-list* in a subshell, and that subshell holds the SSH channel's stdout until the
   process it waits on exits — which, for a server, is never. The server started
   correctly every time; only the acknowledgement was stuck behind a `bash -c` wrapper,
   visible in the remote process table as a live parent of a healthy server. Now
   `cd X || exit 1; nohup Y &`, which keeps the directory check and backgrounds the
   server directly.
2. **`pkill -f "backup_server.py --host-id"` matched its own caller's command line.** The
   teardown path killed the invoking shell. Hardened to `backup_serve[r].py`.
3. **`throughput_experiment.py` built its own hard-coded `127.0.0.1` proxy** and
   referenced a `port` variable the refactor had removed — a runtime `NameError` waiting
   for the first remote run. Now uses `endpoint.url()`.

`test_deployment.py` covers the slot planner, manifest round-tripping, the remote delay
refusal, and `test_real_trial_through_the_remote_path()`, which drives a real 36-call
trial through a manifest against servers started outside the harness, treating localhost
as the "remote" node. That test is what made the remote path checkable without hardware.

**Verification that the pinning is real, not asserted.** `provision_remote.py --verify`
reports the CPU affinity each server actually holds. On the reported run: *all 128
endpoints answered, every server has its own core on its node*. Hyperthread siblings on
the testbed processor pair as (0,16), (1,17), …, so cores 2–14 are fourteen distinct
physical cores and no two servers share one core's two threads. Without this check, a
measurement at this scale cannot be distinguished from CPU scheduling — which is exactly
what the withdrawn single-machine table was.

---

## Fix 7 — the accept backlog was never a controlled variable

Fix 5 established that DEFT's five-deep accept queue causes the latency tail, inferred
from the distribution's shape and the contrast with gRPC. It was never varied directly,
because on one machine there was no way to separate it from everything else going wrong.

`backup_server.py --backlog N` sets `SimpleXMLRPCServer.request_queue_size` before
`Backup` constructs the server. Leaving it unset reproduces DEFT exactly, so the two
configurations are a controlled pair rather than a fix.

### The artifact, measured rather than inferred

Reproducing DEFT as shipped on bare metal gives a result that cannot be about placement:

| method | p95 (ms) | p99 (ms) |
| --- | --- | --- |
| optimal | 64.94 | 1040.06 |
| heuristic | 66.79 | 1042.47 |
| round_robin | 66.67 | 1039.96 |
| random | 65.49 | 1051.67 |

Four methods whose p95 differs by **1.85 ms** have p99 values within **11.67 ms** of
1040 ms. A constant surviving every change of method is an artifact, and one second is
Linux's initial retransmission timeout. The kernel counters confirm the mechanism
directly rather than merely being consistent with it:

```
node1   TcpExtListenOverflows  103209   TcpExtListenDrops  103209   TcpRetransSegs  116
node5   TcpExtListenOverflows   24705   TcpExtListenDrops   24705   TcpRetransSegs  225
node10  TcpExtListenOverflows    1923   TcpExtListenDrops    1923   TcpRetransSegs  141
```

`ListenDrops` equals `ListenOverflows` exactly: the connections were refused at the
server, not lost in the network. The gradient across nodes tracks how heavily each node's
backups were selected, so the artifact is **correlated with the quantity being measured**
— which is why it could not be noted and set aside.

### What it changes

Raising the backlog to 512, with every other variable fixed, took overflows to **zero**
and p99 from 1040 ms to 79 ms. The consequence for the placement claim is larger than the
tail alone suggests:

| primaries sharing one backup | mean @ backlog 5 | mean @ backlog 512 |
| --- | --- | --- |
| 1 | 50.32 | 41.86 |
| 16 | 218.99 | 42.39 |
| 45 | 342.64 | 49.66 |
| 79 | 378.19 | 56.41 |

**7.5x becomes 35%.** Most of the apparent penalty for sharing a backup was connection
setup being refused, not queueing at the backup. This sharpens Fix 5's ~4.3x inflation
estimate for the *concentration* penalty: it is attributable to one configurable integer.

### The bias does not have a single sign

An earlier version of this section generalised that to "the shipped backlog inflates the
measured placement effect". Repeating the 16-host scenario on the same nodes with the same
`tc` degradation, changing only the backlog, shows the opposite at that scale:

| method | mean @ 5 | median @ 5 | p99 @ 5 | mean @ 512 | median @ 512 | p99 @ 512 |
| --- | --- | --- | --- | --- | --- | --- |
| optimal | 28.70 | 14.64 | 1015.26 | 23.17 | 22.68 | 49.45 |
| heuristic | 32.04 | 12.08 | 1020.77 | 23.82 | 23.09 | 56.76 |
| round_robin | 35.22 | 14.44 | 1013.35 | 29.89 | 22.11 | 120.62 |
| random | 51.54 | 12.04 | 858.46 | 36.45 | 23.62 | 95.39 |

`round_robin` minus `heuristic` is **3.18 ms as shipped** and **6.07 ms at backlog 512**.
At 16 hosts the shipped queue *understates* the placement gap; at 128 hosts it inflated
the concentration penalty. Both follow from one mechanism: overflow is driven by how many
primaries share a backup, so at 80 primaries it falls hardest on exactly the worst
placements and exaggerates the spread, while at 10 primaries over 16 backups it falls
roughly evenly and compresses it.

**The correct statement is that the shipped backlog biases a measured placement effect,
with a sign that depends on how concentrated the workload is.** That is a stronger reason
to treat it as a controlled variable than if it merely scaled everything.

Two further observations from the same table. The one-second plateau appears at 16 hosts
too (1013-1023 ms across four methods whose p95 spans 28 to 102 ms), on hardware where
nothing else is contended: it has bounded p99 at **every** scale in this work, and was
only ever visible as an artifact once the backlog could be varied. And the five-deep queue
*lowers* the median, 22.68 to 14.64 ms — a short accept queue is admission control, and
DEFT as shipped trades a better median for a catastrophic tail.

Both configurations are reported. Neither replaces the other: the shipped one describes
DEFT as deployed, the raised one isolates placement.

---

## Fix 8 — a contiguous degraded block flatters ordering-based baselines

This one invalidated a result we had already written down, and it is the subtlest defect
in this document.

### Simulated degradation replaced with a physical mechanism

`RemoteDeployment` **refuses** a `slow_hosts` argument rather than ignoring it: it cannot
inject a per-call sleep into a server it did not start, and reporting a degradation that
was never applied would be worse than failing. Degradation on the testbed is therefore
imposed with `tc netem`.

`tc` attaches to an interface, but each node carries thirteen backups, and shaping the
interface would degrade the healthy co-tenants the comparison depends on as controls.
`apply_degradation.py` installs a four-band `prio` qdisc with a `u32` filter on **source
port**, so only the degraded servers' traffic is delayed. Band 4 is unreachable through
the default priomap, so healthy flows never enter the netem queue at all rather than
being delayed by zero. Measured on adjacent ports of one NIC:

| host | node | port | ms/call | |
| --- | --- | --- | --- | --- |
| w1 | node1 | 19000 | 101.6 | degraded |
| w3 | node1 | 19002 | 0.8 | healthy |
| w27 | node3 | 19000 | 101.3 | degraded |
| w28 | node3 | 19001 | 0.9 | healthy |

Note 101 ms, not 50: `netem` delays every packet the server sends, so a
connection-per-call protocol pays it twice — once on the SYN-ACK, once on the response.
The physical arm is therefore **not numerically comparable** to the simulated 50 ms sleep
used at 16 and 32 hosts. It is a more faithful model of a congested link, not the same
one.

### The defect

The first degraded run marked the first sixteen backup-eligible hosts, `w81`–`w96`, a
contiguous block. It reported:

| method | mean (ms) | calls to a degraded backup |
| --- | --- | --- |
| round_robin | **41.28** | **0.00%** |
| optimal | 41.55 | 0.00% |
| heuristic | 41.54 | 0.00% |
| random | 50.63 | 10.75% |

`round_robin` tying with `optimal` while touching no degraded host would have been
reported as "a cost-free load-spreading baseline matches feature-aware selection". It has
no access to the degradation features and cannot be avoiding anything.

`round_robin_mapping()` cycles a **lexicographically** sorted host list, in which every
name beginning `w1`–`w7` precedes `w8`. With 80 primaries the cursor advances 80
positions and never reaches `w81`–`w96`. It scored zero by **never arriving**, not by
avoiding. Natural-order sorting would not help: those hosts still sit beyond position 80.
`concentrated` at 0.00% was the same accident, piling onto `w1`–`w2`.

### The fix and what it changed

`--degraded-selection spread` samples the degraded hosts evenly along that same ordering
(cycle positions 0, 8, 16, …, 120), putting them inside every method's reach. `prefix`
remains the default so earlier runs reproduce.

| method | mean, contiguous | mean, spread | degraded share, spread |
| --- | --- | --- | --- |
| round_robin | 41.28 | **51.87** | 12.50% |
| random | 50.63 | 51.25 | 11.25% |
| optimal | 41.55 | 41.76 | 0.00% |
| heuristic | 41.54 | 42.03 | 0.00% |
| rl (PPO) | 42.54 | 45.04 | 0.00% |
| concentrated | 55.44 | 260.11 | 100.00% |

`round_robin`'s apparent parity with `optimal` disappears entirely. With the degraded
hosts reachable, the feature-aware methods beat both ordering-based ones with
**r = −1.000** over 15 paired runs — no run dissents — and roughly 2x at p99 (80 ms
against 156 ms).

**Methodological rule:** any scenario that marks a contiguous block of hosts as degraded
must report what share of each method's calls actually reached one. Without that column
the comparison is uninterpretable, and a load-spreading baseline can appear to avoid an
impairment it merely never encountered.

### A second thing the share column exposed

`rl` (PPO) reaches a degraded backup **0.00%** of the time, matching the deterministic
methods, and is still significantly slower than both (p_Holm = 0.0009 vs `heuristic`,
r = 1.000). It used **44** distinct backups where `heuristic` used 112 and `random` 119.
Host-quality discrimination and load balancing are separable properties, and PPO acquired
the first without the second. "PPO degrades at 128 hosts" was true but imprecise; this is
the sharper statement.

---

## 128 hosts: now measured, and what that changed

Fix 5 concluded that 128 hosts is not measurable on one machine and that real distributed
hardware would be required. That measurement now exists: ten bare-metal nodes carrying
128 backups at one physical core each, plus a client node, on a 25 GbE LAN.

It changed more than it confirmed:

| single-machine claim | status after bare metal |
| --- | --- |
| Concentration is catastrophic (7.5x across the load sweep) | **Revised** — 35% once the accept queue is sized; most of the 7.5x was refused connections |
| Feature-aware selection beats blind spreading under heterogeneity | **Confirmed**, r = −1.000, and now with the mechanism shown (0.00% vs 12.50% degraded) |
| Heuristic matches the Hungarian optimum | **Confirmed** as a null, p_Holm = 0.909 (mean) and 1.000 (p99) |
| PPO degrades at 128 hosts | **Confirmed and sharpened** — it avoids degraded hosts perfectly; the cost is load concentration alone |
| Selection matters regardless of host quality | **Withdrawn** — with homogeneous hosts, `optimal` vs `random` differs by 0.002 ms (p = 1.000) |

The last row is new and is the one most worth stating: where every candidate is
equivalent, a selector that reads node features has nothing to read, and the Hungarian
assignment buys nothing over choosing uniformly at random. The paper's claim is
specifically about **heterogeneous** pools, and that is now the only form in which it is
made.

### What the smaller scales inherit

The 9-, 16- and 32-host results have **not** been re-run on bare metal. They were
measured over the shipped five-deep backlog, so Fix 7 applies to them: their
concentration penalties should be read as upper bounds. The *ordering* of methods is
unaffected, since Fix 7 compresses all methods toward each other rather than reordering
them, but the magnitudes are inflated by an amount that is now known to be substantial.

---

## Summary of claims after bare-metal validation

| claim | status |
| --- | --- |
| Feature-aware selection beats blind load-spreading under heterogeneity | **Strongly supported** — r = −1.000 at 128 hosts on bare metal, with the mechanism measured |
| Greedy heuristic matches the Hungarian optimum | **Supported** as a null at every scale, including bare metal |
| PPO avoids degraded hosts but concentrates load | **Supported** — 0.00% degraded, 44 distinct backups vs 112 |
| Selection matters when hosts are homogeneous | **Withdrawn** — 0.002 ms difference, p = 1.000 |
| Concentration costs ~7.5x | **Revised to ~35%** — the remainder was the five-deep accept queue |
| `round_robin` matches feature-aware selection at 128 hosts | **Withdrawn** — an artifact of lexicographic ordering (Fix 8) |
| DEFT's shipped backlog imposes a 1 s p99 floor under concurrent replication | **Supported** — 103,209 ListenOverflows, zero at backlog 512 |
| p99 as a latency metric on XML-RPC | **Invalid** at backlog 5; **usable** at backlog 512 and on gRPC |
| The result holds at 16, 32 and 128 hosts | **Supported** — same ordering and mechanism at all three on bare metal; the gap widens with the pool (6.72 / 7.38 / 10.11 ms) |
| The shipped backlog inflates the measured placement effect | **Narrowed** — it *biases* it, with a sign that depends on workload concentration; it inflates at 128 hosts and understates at 16 |
| DEFT's five-deep queue is purely harmful | **Narrowed** — it lowers the median by shedding load, at the cost of a one-second tail |
| 9-host concentration magnitude | **Upper bound** — the only scale not re-run on bare metal |
