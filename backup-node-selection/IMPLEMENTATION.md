# Implementation Guide: Intelligent Backup-Node Selection for DEFT

This document is the code-level companion to [EXPERIMENTS.md](EXPERIMENTS.md) (methodology
and results) and [paper/paper.tex](paper/paper.tex) (the write-up). Where those two explain
*what was found*, this document explains *what was built, in which file, why, and in what
order to run it* — enough for a new reader to clone the repo and regenerate every number in
EXPERIMENTS.md from scratch.

All paths below are relative to [`code/`](code/) in this folder unless stated
otherwise; that directory is the working directory for every command in this guide.

## 1. TL;DR

```bash
cd code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train every RL model used in the final results (see §5 for exact timing per command)
python3 rl_backup_selector.py --train --timesteps 50000  --model rl_model.zip
python3 rl_backup_selector.py --train --timesteps 80000  --model rl_model_scaled.zip     --n-hosts 16  --n-primaries 10 --n-degraded 2  --scale-seed 0
python3 rl_backup_selector.py --train --timesteps 100000 --model rl_model_scaled32.zip   --n-hosts 32  --n-primaries 20 --n-degraded 4  --scale-seed 0
python3 rl_backup_selector.py --train --check-collapse --timesteps 1000000 --model rl_model_scaled128_v3.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256

# Run every experiment. IMPORTANT: the harnesses APPEND to results_rpc/*.csv, and the
# end-of-run backup_load breakdown reads the whole file -- so the raw CSV must be deleted
# (or copied aside) between scenarios, or results from different scenarios will mix.
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 --rl-model rl_model_scaled32.zip
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 --rl-model rl_model_scaled128_v3.zip

python3 selection_overhead.py

rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl
rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip
```

Each command's console output is exactly the numbers reported in the corresponding table
of EXPERIMENTS.md — §5 below maps every command to the section it produced.

## 2. What Problem This Solves

DEFT (`shahriyar2023deft`) replicates a primary NF's state to a backup NF for fault
tolerance, but its Failure Detection Unit assigns that backup with no stated selection
criterion — in the repo's own shipped `primary_to_secondary.json`, 3 of 5 primaries are
hardcoded onto a single backup (`h4`). Everything in this guide implements and evaluates
alternatives to that static mapping: a deterministic multi-criteria heuristic, an exactly
optimal assignment (Hungarian algorithm), and trained RL policies (PPO, with DQN as a
cross-algorithm check), evaluated against the static baseline, naive baselines (random,
round-robin, worst-case concentrated), and decentralized hash placement.

## 3. Repository Layout: New and Modified Files

```
code/
├── config.py                        (MODIFIED — one-line bugfix)
├── topology_features.py             (NEW — real-topology graph + synthetic resources)
├── backup_selector.py               (NEW — baseline/concentrated/random/round_robin/heuristic/optimal)
├── rl_backup_selector.py            (NEW — shared RL env; PPO + DQN training, inference)
├── rpc_contention_experiment.py     (NEW — main latency evaluation harness)
├── throughput_experiment.py         (NEW — throughput/failure-rate evaluation harness)
├── selection_overhead.py            (NEW — per-mapping compute-time benchmark)
├── requirements.txt                 (MODIFIED — added networkx, scipy, gymnasium, stable-baselines3)
├── exp_package/                     (DEFT's real, unmodified 2PC Primary/Backup classes)
├── primary_to_secondary.json        (DEFT's shipped static 9-host mapping)
├── rl_model.zip                     (trained model — 9-host homogeneous)
├── rl_model_scaled.zip              (trained model — 16-host scaled+heterogeneous)
├── rl_model_scaled32.zip            (trained model — 32-host scaled+heterogeneous)
├── rl_model_scaled128_collapsed.zip (trained model — 128-host degenerate baseline, ent 0 / [64,64])
├── rl_model_scaled128_v3.zip        (trained model — 128-host, FINAL fix, used in reported results)
├── rl_algorithm_comparison.py       (NEW — structural PPO-vs-DQN comparison, all four scales)
├── rl_model_dqn*.zip                (trained DQN models — cross-algorithm check, EXPERIMENTS.md §4.10)
│
│   # Multi-node deployment (EXPERIMENTS.md §4.11)
├── deployment.py                    (NEW — Endpoint + local/manifest-driven remote backends)
├── backup_server.py                 (NEW — one backup as a standalone process, --backlog)
├── provision_remote.py              (NEW — place servers on nodes, pin cores, write manifest)
├── apply_degradation.py             (NEW — tc netem per source port; physical degradation)
├── test_deployment.py               (NEW — 9 tests incl. a real trial through the remote path)
│
│   # Online selection under churn (EXPERIMENTS.md §4.12)
├── online_model.py                  (NEW — churn process, service/migration cost model)
├── online_selector.py               (NEW — policies + exact offline optimum by DP)
├── competitive_analysis.py          (NEW — the sweeps; writes findings/online_competitive.*)
├── test_online.py                   (NEW — 6 tests incl. Theorem 1's construction)
│
│   # Migration cost of real state (EXPERIMENTS.md §4.13)
├── flow_state.py                    (NEW — connection-tracking table, delta replication)
├── workload.py                      (NEW — flow arrivals; Little's-law validated)
├── measure_migration_cost.py        (NEW — occupancy, encodings, M, crossover)
├── test_flow_state.py               (NEW — 8 tests incl. lossless round-trip, Little's law)
│
│   # Table generators (findings -> paper/*.tex, no hand transcription)
├── make_testbed_tables.py           (NEW)
├── make_online_tables.py            (NEW)
├── make_migration_tables.py         (NEW)
└── results_rpc/                     (output — CSVs written by the experiments above)

findings/                             (saved output of every run, with the command in its header)
EXPERIMENTS.md                        (methodology + results narrative)
paper/                                (LaTeX source, figures, compiled paper.pdf)
IMPLEMENTATION.md                     (this file)
README.md                             (setup + reproduction commands)
```

Two 128-host PPO checkpoints ship: `rl_model_scaled128_collapsed.zip` (the degenerate
`ent_coef=0.0`/`[64,64]` baseline) and
`rl_model_scaled128_v3.zip` (the final fix, used in all reported 128-host results). The
intermediate `[128,128]`/600k attempt is described in EXPERIMENTS.md §4.6 but not released
as a checkpoint. The Mininet-path files (`test.py`, `perflow-packet-counter.py`,
`run_trials.sh`, `summarize_metrics.py`, §4.8) were part of the original development tree
but are not shipped, since they are not used for any reported result.

## 4. File-by-File: What Was Added/Changed and Why

### 4.1 `config.py` (modified)

Pre-existing bug fix: added a missing `import json` at the top of the file. Without it,
`json.load()` at the bottom of the file raised `NameError` on import — which blocked
*every* script that imports `config` (including the repo's own pre-existing `test.py`),
not just the new code added here.

### 4.2 `topology_features.py` (new)

Single source of truth for topology distances and synthetic per-host resource state,
shared by every method and both experiment harnesses.

- `build_topology_graph()` — builds a `networkx.Graph` directly from `config.SWITCH_HOST_TO_PORT`,
  so it can never silently drift from the real 9-host Mininet topology defined in
  `config.py`.
- `generate_scaled_topology(n_hosts, n_primaries, seed)` — generates a synthetic star
  topology (`w1..wN`, all connected through one hub) at an arbitrary size, used for the
  16/32/128-host scaled scenarios. **Note**: the `seed` parameter is currently unused —
  the graph *shape* is deterministic for a given `n_hosts`/`n_primaries`; only the
  resource values vary by seed (see EXPERIMENTS.md §5, "Multi-seed generalization covers
  resources, not topology structure").
- `compute_hop_and_latency(graph, hosts, latency_per_hop_ms)` — real hop-count/latency
  between every host pair via `networkx.single_source_shortest_path_length`.
- `HostResourceState` (dataclass) / `generate_synthetic_resources(hosts, seed, overrides)` —
  seeded synthetic health/capacity/CPU/storage per host; `overrides` lets a caller pin
  specific hosts to bad values (used to construct the "degraded hosts" in every scaled
  scenario).
- `FeatureContext` / `build_feature_context(...)` — bundles distances + resources into
  the one object every selection method and the RL environment consumes.

### 4.3 `backup_selector.py` (new)

All non-learned selection methods, plus the CLI used to regenerate
`primary_to_secondary.json` directly.

- `feasible_backups(primary, hosts)` — every host except the primary itself and
  infrastructure hosts (`client`, `stamper`); **note this includes other primaries** — a
  host that's a primary for one flow can still back up a different flow.
- `score_candidates(primary, candidates, ctx, weights)` — the shared scoring function
  (health, capacity ratio, normalized latency, normalized hop-count) used by both
  `heuristic` and `optimal`, and by the RL reward.
- `select_backups(...)` — the **heuristic**: greedy per-primary argmax of
  `score_candidates`, applying a load-decay penalty (`capacity -= 20`, `health *= 0.95`)
  to whichever host was just chosen, so load naturally spreads without any learning.
- `optimal_mapping(...)` — the **optimal** bound: duplicates each backup into
  capacity-constrained columns and solves the resulting assignment problem exactly via
  `scipy.optimize.linear_sum_assignment` (Hungarian algorithm).
- `get_baseline_mapping()` — DEFT's real shipped static mapping, kept verbatim for direct
  comparison.
- `concentrated_mapping(...)` — every primary onto the same single backup; a
  scale-independent worst case.
- `random_mapping(...)`, `round_robin_mapping(...)` — the two blind baselines. Note
  `round_robin_mapping` cycles the **alphabetically sorted** host list (Python's default
  string sort, not numeric) — this interacts with host naming in a way that matters at
  scale (see EXPERIMENTS.md §4.5/§4.6).
- `write_mapping(mapping, path)` / CLI (`--mode {heuristic,baseline,concentrated,random,round_robin,optimal}`) —
  drop-in replacement for however `primary_to_secondary.json` gets populated; no
  downstream changes needed anywhere else in the codebase.

### 4.4 `rl_backup_selector.py` (new)

The RL agents: shared environment, PPO and DQN training, and inference.

- `BackupSelectionEnv(gym.Env)` — one episode assigns every primary a backup in
  sequence, under one freshly-drawn random `FeatureContext` per episode (so the policy
  generalizes across resource realizations, not one memorized instance). Observation =
  per-host `[health, capacity_frac, cpu, storage]` concatenated with a one-hot encoding
  of the current primary. Action space = `Discrete(n_hosts)`. Reward = the same
  `score_candidates` value the heuristic uses for a feasible action, an overload penalty
  for re-using an already-loaded host, and a large fixed penalty (`-10`) for an
  infeasible action — chosen so invalid actions are discouraged **during training**,
  not just corrected after the fact.
- `train_ppo(hosts, primaries, ..., ent_coef=0.01, net_arch=None, n_steps=2048, batch_size=64)` —
  trains a PPO agent (`stable_baselines3`). **`ent_coef` defaults to 0.01, not SB3's own
  default of 0.0** — this was added specifically after the 128-host model trained with
  `ent_coef=0.0` collapsed to a degenerate constant policy (see §4.9 below); with no
  entropy bonus, nothing discourages the policy from locking onto one action early.
  `net_arch` defaults to `[128, 128]` (SB3's own default is `[64, 64]`) for the same
  reason — a bigger action/observation space needs more network capacity.
- `train_dqn(hosts, primaries, ..., buffer_size=50000, exploration_fraction=0.3, ...)` —
  trains a DQN agent on the **same** env, reward, and inference path, so only the learner
  differs: off-policy replay instead of on-policy rollouts, value-based instead of
  actor-critic, epsilon-greedy decay instead of an entropy bonus. Used for the
  cross-algorithm check in EXPERIMENTS.md §4.10. Its exploration flags play the role
  `ent_coef` plays for PPO: with SB3's defaults DQN collapses at 32 hosts, and widening
  exploration recovers it.
- `load_rl_model(path, algorithm)` — loads either a PPO or DQN checkpoint; both expose the
  same `predict(obs, deterministic=True)` interface, so nothing downstream is
  algorithm-specific.
- `predict_mapping(model, hosts, primaries, ctx)` — deterministic policy rollout,
  falling back to the best-scoring feasible candidate if the policy still picks an
  infeasible host at inference time (a true fallback now, not the only safety net).
  Shared by both algorithms, which is what keeps the comparison fair.
- CLI: `--train`, `--predict`, `--algorithm ppo|dqn`, `--timesteps`, `--model`, `--seed`,
  `--n-hosts`, `--n-primaries`, `--n-degraded`, `--scale-seed`; PPO knobs `--ent-coef`,
  `--net-arch` (space-separated layer sizes), `--n-steps`, `--batch-size`; DQN knobs
  `--buffer-size`, `--exploration-fraction`, `--exploration-final-eps`,
  `--target-update-interval`, `--learning-rate`; and `--check-collapse` — the last one prints
  how many distinct backups the trained/loaded policy uses across 5 context seeds, so you
  can catch a degenerate policy straight away instead of discovering it part-way through
  a full RPC experiment run.

### 4.5 `rpc_contention_experiment.py` (new)

The main evaluation harness — this is what actually produces the latency numbers in
EXPERIMENTS.md §4.1–4.6.

**Design**: rather than reconstruct the full Mininet network path (see §4.8 for why
that path was abandoned), this drives the **real, unmodified** `Backup`/`Primary` 2PC
classes from `exp_package/Two_phase_commit/` directly over XML-RPC, with each backup
host running as its own real OS process on localhost.

- `start_backup_servers(unique_backups, base_port, slow_hosts, slow_delay_s)` — one
  `multiprocessing.Process` per unique backup in a mapping (real OS-level isolation, not
  threads — threads would share a GIL and confound the very contention effect being
  measured). Hosts in `slow_hosts` get an artificial per-call delay simulating a
  genuinely overloaded machine.
- `replicate_burst(port, n_calls, burst_size)` — fires `n_calls` requests using
  `burst_size` concurrent in-flight connections (not one sequential stream), so a shared
  backend genuinely accumulates a request queue. Each burst worker keeps **one
  persistent connection** for all its calls (a fresh `ServerProxy` per call would force a
  new TCP handshake every time, which previously dominated measurements with
  connection-thrash noise). Retries on `ConnectionResetError`/`ConnectionRefusedError`/`OSError`
  with jittered exponential backoff, up to `MAX_RETRY_ATTEMPTS = 8`; a call that still
  fails after all retries is recorded as a failure with a `FAILED_CALL_PENALTY_MS = 10000`
  sentinel latency instead of raising and crashing the whole experiment. **This retry
  budget was originally 3 attempts with no failure-as-data path** — it crashed outright
  the first time the 128-host/80-primaries-on-one-backup scenario was run; it was
  hardened to the current form specifically because of that crash (see EXPERIMENTS.md
  §4.6 and §5, "Retry-policy inconsistency across scale points").
- `run_experiment(context_seeds, trials, n_calls, burst_size, methods, ...)` — the outer
  loop: for each context seed, builds a fresh `FeatureContext` and a fresh mapping per
  method (including re-seeding `random`'s mapping per context seed), then for each of
  `trials` trials runs every method **in randomized order** so no method is
  systematically favored by background system load at a particular point in a long run.
  Aggregates latencies across all (seed × trial) combinations and prints: a summary
  table (mean/median/p95/p99/bootstrap CI per method), pairwise Mann-Whitney U +
  rank-biserial effect size for every method pair, and a latency-vs-`backup_load`
  breakdown.
- CLI: `--context-seeds` (one or more), `--trials`, `--calls`, `--burst`, `--methods`,
  `--n-hosts`/`--n-primaries`/`--n-degraded`/`--scale-seed` (omit all four to use the
  real 9-host DEFT topology from `config.py`), `--rl-model`.

### 4.6 `throughput_experiment.py` (new)

Complements the latency harness with the other two metrics DEFT's own paper reports:
throughput and failure rate. Structurally similar to `rpc_contention_experiment.py`
(same server-process setup, same real `Backup` class) but each burst worker runs for a
**fixed wall-clock duration** (`--duration` seconds) instead of a fixed call count,
counting successes and failures (timeouts/connection errors after retry) rather than
per-call latency — a fixed-call-count design cannot measure throughput at all, since it
conflates "slower" with "fewer calls attempted." Sets `socket.setdefaulttimeout(2.0)` so
a call against an overloaded backup fails fast rather than hanging the measurement
window. CLI mirrors `rpc_contention_experiment.py`'s, with `--duration` replacing
`--calls`.

### 4.7 `selection_overhead.py` (new)

Answers a different question from the two harnesses above: not "how good is the
resulting mapping" but "how long does it take to *compute* the mapping." Times one full
mapping computation per method (30 repeats, first call excluded as JIT/cache warmup),
across all four topology sizes (9/16/32/128 hosts), loading each scale's own trained RL
model. No CLI arguments — the four scenarios are hardcoded in `main()`; run it directly.

### 4.8 Mininet path: `test.py`, `perflow-packet-counter.py`, `run_trials.sh`, `summarize_metrics.py` (modified/new, NOT used for final results)

These were the **first** attempt at an evaluation harness, driving the actual physical
(Mininet-emulated) network topology rather than RPC-isolation. They are left in the repo
and documented here for completeness, but **none of the numbers in EXPERIMENTS.md or the
paper come from this path** — it was abandoned after the development machine's kernel
turned out unable to provide a genuine bandwidth constraint for contention to manifest
against (see EXPERIMENTS.md §3.1 and §5 for the full rationale: missing `openvswitch`
kernel module, missing SDN controller, and decisively, `TCLink` bandwidth throttling
never actually engaging on the available kernel).

- `config.py`'s bugfix (§4.1) was required to even import this path.
- `test.py`: `n_h = len(config.PRIMARIES)` (was hardcoded to `1`, meaning the original
  test only ever exercised one primary); switched from `RemoteController` to Mininet's
  built-in `OVSController` (no SDN controller existed in the repo); switched to OVS
  userspace datapath (`OVSSwitch(datapath='user')`, since the WSL2 kernel lacks the
  `openvswitch` kernel module); added `parse_ping_output()` and `record_metrics()` to
  capture ping-based packet-loss/RTT metrics per run, gated by a `RUN_LABEL` environment
  variable (`sudo -E` required to propagate it through `sudo`) or a `run_label` parameter.
- `perflow-packet-counter.py`: added `record_replication_metrics()` to log
  throughput/latency to a CSV whenever `generate_statistics()` runs.
- `run_trials.sh` / `summarize_metrics.py`: a bash driver for running N trials of
  baseline vs. heuristic and an aggregation script for the resulting CSVs.

If you want to explore this path anyway: it additionally needs a real Mininet + Open
vSwitch install, the `hazelcast-4.2.2` distribution downloaded separately (not
committed), and root privileges (`sudo python3 test.py`). It is **not** part of the
reproduction sequence in §1/§5.

### 4.9 Trained RL model inventory

| File | Scenario | Command that produced it | Notes |
|---|---|---|---|
| `rl_model.zip` | 9 hosts (homogeneous) | `--timesteps 50000` | |
| `rl_model_scaled.zip` | 16 hosts | `--timesteps 80000 --n-hosts 16 --n-primaries 10 --n-degraded 2` | |
| `rl_model_scaled32.zip` | 32 hosts | `--timesteps 100000 --n-hosts 32 --n-primaries 20 --n-degraded 4` | |
| `rl_model_scaled128_collapsed.zip` | 128 hosts, **degenerate baseline** | `--timesteps 400000 --ent-coef 0.0 --net-arch 64 64` (SB3 raw defaults) | **Collapsed**: uses only 1–5 distinct backups, routing ~all 80 primaries to one host (`w101`). Shipped for reproducibility of the collapse finding, not usable in production. |
| *(intermediate attempt, not shipped)* | 128 hosts, **fix attempt 1** | `--timesteps 600000 --ent-coef 0.01 --net-arch 128 128` | Partial improvement only; no checkpoint released. |
| `rl_model_scaled128_v3.zip` | 128 hosts, **final** | `--timesteps 1000000 --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256` | Used in all reported 128-host `rl` results. Avoids degraded hosts correctly (0/400) and spreads across 24–30 distinct backups, but still load-balances worse than the heuristic (74–76) — see EXPERIMENTS.md §4.6. |

DQN checkpoints for the §4.10 cross-algorithm check (add `--algorithm dqn`):

| File | Scenario | Command that produced it | Notes |
|---|---|---|---|
| `rl_model_dqn.zip` | 9 hosts | `--algorithm dqn --timesteps 50000` | Spreads across all 5 backups, like PPO. |
| `rl_model_dqn_scaled.zip` | 16 hosts | `--algorithm dqn --timesteps 80000 --n-hosts 16 --n-primaries 10 --n-degraded 2` | 9–10 of 10 backups. |
| `rl_model_dqn_scaled32.zip` | 32 hosts, **pre-fix** | `--algorithm dqn --timesteps 100000 --n-hosts 32 ...` (SB3 exploration defaults) | **Collapsed**: 2–4 of 20 backups. Kept as the before case. |
| `rl_model_dqn_scaled32_mitigated.zip` | 32 hosts, **fixed** | `--algorithm dqn --timesteps 150000 --exploration-fraction 0.7 --exploration-final-eps 0.15 --buffer-size 100000 --target-update-interval 1000` | 16–19 of 20 backups. This is the checkpoint used in §4.10. |
| `rl_model_dqn_scaled128.zip` | 128 hosts | `--algorithm dqn --timesteps 1000000 --net-arch 256 256 --batch-size 256 --buffer-size 100000 --exploration-fraction 0.5` | **Still collapses**: 1–3 of 80 backups, 172/400 assignments on degraded hosts even with a wide network and 1M steps. This is the cross-algorithm evidence, not a config to copy. |

Use `--check-collapse` on any of these (or right after `--train`) to see this diagnosis
yourself quickly, instead of discovering it part-way through a full RPC experiment run.
`rl_algorithm_comparison.py` runs the same diagnosis for PPO and DQN across all four scales
in one pass.

### 4.10 `deployment.py`, `backup_server.py`, `provision_remote.py` (new — multi-node)

The single-machine harness started its own backup servers with
`multiprocessing.Process` and addressed them at `127.0.0.1:base_port+i`. That coupling —
**topology host identity, OS process, and network address all being the same thing** — is
what made a testbed run impossible, and it is what these three files separate.

- `deployment.py`
  - `Endpoint(host_id, ip, port)` — a topology host's address, with `.url()`/`.address()`.
    Everything downstream takes an `Endpoint` instead of an `int` port.
  - `LocalDeployment` — starts servers itself, `owns_servers = True`. Its port assignment
    is byte-identical to the old rule, so every published single-machine result reproduces
    unchanged.
  - `RemoteDeployment` — reads a manifest; the servers already exist on other machines and
    this only resolves addresses. **It raises on a `slow_hosts` argument** rather than
    silently ignoring it: it cannot inject a per-call sleep into a server it did not start,
    and reporting a degradation that was never applied would be worse than failing.
  - `make_deployment()` picks between them from `--manifest` or `IDEFT_MANIFEST`.
- `backup_server.py` — one backup as its own process, bound to a routable address. Prints
  `host_id`, pid and `os.sched_getaffinity(0)` to stderr so pinning is visible in the log.
  `--backlog N` sets `SimpleXMLRPCServer.request_queue_size` **before** `Backup` constructs
  the server (it binds and calls `listen()` in `__init__`, so patching afterwards is too
  late). Leaving it unset reproduces DEFT as shipped, which is the point of the comparison.
  `--delay` exists for parity with the local harness and warns that `tc`/cgroups are the
  right mechanism on a testbed.
- `provision_remote.py` — assigns each host id a (node, port, core) slot, launches over SSH
  under `taskset`, and writes the manifest. `--verify` reports the affinity each server
  **actually holds**, which is what lets a paper claim one core per backup rather than
  assert it. `--degrade HOST=KIND:ARG` records how a host was degraded so a run states the
  mechanism. `--teardown` stops everything.

**Three defects the single-machine harness could not expose**, all found on first contact
with a real testbed and all fixed here:

1. **SSH hung on the first server.** `cd X && nohup Y &` backgrounds the *whole AND-list*
   in a subshell; that subshell holds the SSH channel's stdout until the server it waits on
   exits, which is never. The server started correctly every time — only the acknowledgement
   was stuck behind a `bash -c` wrapper. Now `cd X || exit 1; nohup Y &`, which keeps the
   directory check and backgrounds the server directly.
2. **`pkill -f "backup_server.py --host-id"` matched its own caller's command line** and
   killed the invoking shell. Hardened to `backup_serve[r].py` in both the teardown path
   and the docs.
3. **`throughput_experiment.py` built its own hard-coded `127.0.0.1` proxy** and referenced
   a now-undefined `port`, which would have been a runtime `NameError` on the first remote
   run. It now uses `endpoint.url()`.

`test_deployment.py` covers the slot planner (no core assigned twice, oversubscription
refused), manifest round-tripping, the remote delay refusal, and — the decisive one —
`test_real_trial_through_the_remote_path()`, which runs a real 36-call trial through a
manifest against servers started outside the harness, treating localhost as the "remote"
node. That is what made the remote path testable without a testbed.

### 4.11 `apply_degradation.py` (new — physical degradation)

Imposes degradation with `tc netem` instead of a sleep. The complication is that `tc`
attaches to an **interface** while a node carries many backups: shaping the interface would
degrade the healthy co-tenants the comparison needs as controls.

Each backup owns a distinct TCP port, so the script installs a `prio` qdisc and a `u32`
filter matching **source port**, delaying exactly the degraded servers:

```
tc qdisc add dev <iface> root handle 1: prio bands 4
tc qdisc add dev <iface> parent 1:4 handle 40: netem delay 50ms
tc filter add dev <iface> protocol ip parent 1:0 prio 4 u32 \
    match ip sport <port> 0xffff flowid 1:4
```

Band 4 of a four-band `prio` is unreachable through the default priomap, so healthy flows
never enter the netem queue at all rather than being delayed by zero. Measured on adjacent
ports of one NIC: 101.4 ms for a shaped port against 1.0 ms for its neighbour.

Dry-run by default; `--apply` executes, `--show` prints the live qdisc and filter counters,
`--clear` removes the shaping. `tc qdisc del` failing because nothing is installed is
treated as the normal first-run case, not an error.

### 4.12 `online_model.py`, `online_selector.py`, `competitive_analysis.py` (new — online)

- `online_model.py` — the problem instance. A two-state Markov chain degrades and recovers
  hosts with within-regime drift, a smaller rate kills them, and `OnlineInstance` exposes
  `service_cost`, `round_cost` and `total_cost` with a service cache.
- `online_selector.py` — the policies (`static`, `failover`, `snapshot`, `periodic`,
  `hysteresis` at several thresholds) and `offline_optimal()`, which solves the
  time×host trellis by dynamic programming in **O(k) per round** using the uniform-switching
  -cost trick `dp[t][b] = cost(t,b) + min(dp[t-1][b], M + min_b' dp[t-1][b'])`. Coordinate
  descent handles the congestion coupling between primaries, and the decoupled relaxation
  is kept as a certified lower bound. `cost_floor()` and `excess_ratio()` isolate decision
  quality from the unavoidable floor, which otherwise compresses every ratio toward 1.
- `competitive_analysis.py` — the sweeps (churn, persistence, migration cost, pool size,
  α), with percentile bootstraps over seeds.

**Two things that had to be corrected during development**, both worth knowing:

- **The migration budget proposition failed at α=4** (0.282 against a 0.25 bound). The
  cause was that migrations took effect *within* the round that triggered them, so a policy
  could accumulate regret on a host and then dodge paying for it. Making migrations take
  effect at `t+1` fixed it — and is also the physically correct semantics, since a primary
  cannot retroactively have state on the new backup.
- **`static` was a strawman.** It looks catastrophic only because it sits on hard-failed
  replicas forever. `policy_static_failover` was added and became the honest comparator; it
  is why the headline is "failover closes 91.2%" rather than a flattering number against a
  policy nobody would deploy.

`test_online.py` checks trace reproducibility, DP against brute force, joint-optimum
feasibility, the migration budget, hysteresis boundedness, and Theorem 1 via
`adversarial_alternating(eps)` — the proof's own construction, which measures 5.9 → 59.4 →
594 → 5940 as ε shrinks, i.e. exactly 1/ε. An earlier version of that test perturbed a
feature-derived instance instead and could not have failed, because such an instance has a
cost floor the denominator never drops below.

### 4.13 `flow_state.py`, `workload.py`, `measure_migration_cost.py` (new — migration cost)

DEFT's backup discards the state it receives (`update_state` prints and returns `True`;
the persistence code is commented out), so the migration cost `M` the online model depends
on had no empirical basis. These supply one.

- `flow_state.py` — a connection-tracking / NAT flow table keyed on the 5-tuple, with
  incremental delta replication (`take_delta`/`apply_delta`), TCP state tracking and idle
  eviction. Three wire encodings: `packed` (fixed-width binary, 13-byte key + 31-byte
  value), `json`, and `xmlrpc` — the last being what DEFT actually sends.
- `workload.py` — Poisson flow arrivals with truncated-Pareto lifetimes;
  `expected_occupancy()` is the Little's-law prediction used to validate the generator.
  `replay_trace()` accepts a pcap for future work.
- `measure_migration_cost.py` — occupancy, serialized size per encoding, `M` in rounds, and
  the crossover (`2M/gap`) at which migrating pays.

**A genuine protocol limitation, not a bug in our code:** XML-RPC's `<int>` is a **signed
32-bit** integer, so `nat_ip = 0xc0a80001` raises `OverflowError`. An IPv4 address of
128.0.0.0 or above cannot be sent as an integer at all and must be stringified. That, plus
per-value markup, is most of the 7.31× size inflation over `packed`.

**The result that looks wrong and is the point:** `M` barely moves with encoding or link
speed, because a costlier encoding inflates the full transfer and the per-round delta by
the same factor. `M ≈ 1/delta-share ≈ 30` is a property of traffic churn. This measurement
(23–41 rounds against the assumed 3) **overturned our own earlier conclusion** about when
migrating away from a degraded backup pays.

### 4.14 `make_*_tables.py` (new — table generators)

`make_testbed_tables.py`, `make_online_tables.py` and `make_migration_tables.py` read the
`findings/*.txt` files — which carry in their header the exact command that produced them —
and emit the paper's `.tex` tables. Nothing in those tables is transcribed by hand, so a
rerun regenerates them and a table cannot drift from the run it claims to report.
`make_testbed_tables.py` also writes `findings/cloudlab_facts.json`, which the prose numbers
are checked against programmatically.

One gotcha worth recording: the findings files delimit blocks with `===== Title ... =====`
banners, and the title may be followed by more words on the same line. Splitting on the
title string alone returns the remainder of the banner, which parses as nothing and **fails
silently** — producing an empty table rather than an error. The parser is line-oriented for
that reason. (Same class of failure as the `grep -c` problem in §6.)

## 5. Full Reproduction Walkthrough, Mapped to EXPERIMENTS.md

| # | Command | Produces |
|---|---|---|
| 1 | `pip install -r requirements.txt` (in a fresh venv) | dependencies |
| 2 | `rl_backup_selector.py --train --timesteps 50000 --model rl_model.zip` | §4.1 `rl` column |
| 3 | `rl_backup_selector.py --train --timesteps 80000 --model rl_model_scaled.zip --n-hosts 16 --n-primaries 10 --n-degraded 2` | §4.2 `rl` column |
| 4 | `rl_backup_selector.py --train --timesteps 100000 --model rl_model_scaled32.zip --n-hosts 32 --n-primaries 20 --n-degraded 4` | §4.5 `rl` column |
| 5 | `rl_backup_selector.py --train --check-collapse --timesteps 1000000 --model rl_model_scaled128_v3.zip --n-hosts 128 --n-primaries 80 --n-degraded 16 --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256` | §4.6 `rl` column |
| 6 | `rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 --methods baseline concentrated random round_robin heuristic optimal rl` | §4.1 table + effect sizes, §4.3 homogeneous |
| 7 | Same, with `--n-hosts 16 --n-primaries 10 --n-degraded 2 --rl-model rl_model_scaled.zip` (drop `baseline`) | §4.2 table + effect sizes, §4.3 16-host |
| 8 | Same, with `--context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 --n-hosts 32 --n-primaries 20 --n-degraded 4 --rl-model rl_model_scaled32.zip` | §4.5 table + effect sizes |
| 9 | Same, with `--n-hosts 128 --n-primaries 80 --n-degraded 16 --rl-model rl_model_scaled128_v3.zip` | §4.6 table + effect sizes |
| 10 | `selection_overhead.py` | §4.4 table |
| 11 | `throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 --methods baseline concentrated random round_robin heuristic optimal rl` | §4.7 homogeneous table |
| 12 | Same, with `--n-hosts 16 --n-primaries 10 --n-degraded 2 --rl-model rl_model_scaled.zip` (drop `baseline`) | §4.7 16-host table |
| 13 | `competitive_analysis.py` (~3.5 min, no RPC) | §4.12 tables |
| 14 | `measure_migration_cost.py` (no RPC) | §4.13 tables |
| 15 | `provision_remote.py --nodes … --hosts 128 --per-node 13 --reserve-cores 2 --backlog 512 --manifest …` then `--verify` | §4.11 testbed, **needs 11 machines** |
| 16 | `apply_degradation.py --manifest … --hosts w1 w2 w27 … --delay-ms 50 --iface … --apply` | §4.11.3 physical degradation |
| 17 | Step 9's command plus `--degraded-selection spread --physical-degradation --manifest …` | §4.11.3 table |
| 18 | `make_testbed_tables.py`, `make_online_tables.py`, `make_migration_tables.py` | the paper's `.tex` tables |

Steps 6–9 each start by deleting `results_rpc/rpc_contention_raw.csv` (`rm -f`) before
running, since every run appends to that one file — copy it aside first if you want to
keep more than one scenario's raw per-call data at once. Same for
`results_rpc/throughput_raw.csv` between steps 11 and 12. The exact `rm -f` + command
pairs are spelled out in EXPERIMENTS.md §6 and in §1 above.

Steps 13–14 need no testbed and no RPC; they run on a laptop. Steps 15–17 are the only
ones requiring hardware. **Do not skip the `--verify` in step 15** — it reports the CPU
affinity each backup actually holds, and a warning there means the run would measure CPU
scheduling rather than placement, which is the failure that forced the §4.6 withdrawal.
Step 17's `--degraded-selection spread` is likewise not optional: with a contiguous block
of degraded hosts, `round_robin` can miss them entirely through lexicographic ordering and
appear to avoid them (EXPERIMENTS.md §4.11.4).

## 6. Known Gotchas (things that actually broke during development)

- **`config.py` import errors** — see §4.1; if you hit `NameError: name 'json' is not
  defined` anywhere, that fix is missing.
- **Background-shell working directory doesn't persist between separate command
  invocations.** If you're scripting this rather than typing commands interactively, use
  an explicit `cd path && command` on every line, or an absolute path — a `cd` in one
  shell invocation does not carry over to the next.
- **Root-owned `results/` directory** — if a prior run created `results/` as root (e.g.
  via `sudo python3 test.py` on the Mininet path), the RPC/throughput harnesses avoid the
  conflict entirely by writing to a separate `results_rpc/` directory — no permission fix
  needed for the reproduction sequence in §5.
- **`SimpleXMLRPCServer`'s default accept backlog is 5.** At high fan-in (e.g.
  `concentrated` with ~80 primaries on one backup) this is a real, load-bearing finding
  (EXPERIMENTS.md §4.6), not a bug to "fix" silently — raising it would change what DEFT's
  shipped implementation is being measured against.
  `rpc_contention_experiment.py`'s retry-with-backoff (§4.5) exists to survive it without
  crashing the harness, while recording the resulting failures as data.
  **Resolved as an ablation, not a fix:** `backup_server.py --backlog N` now makes it a
  controlled variable, and leaving it unset still reproduces DEFT exactly. The bare-metal
  run (EXPERIMENTS.md §4.11.1) changes only that integer and shows it accounts for most of
  the apparent cost of concentrating backups — 103,209 `ListenOverflows` on one node at
  backlog 5, zero at 512, and the 1→79 load sweep falling from 7.5× to 35%. Both
  configurations are reported; neither replaces the other.
- **RL policy collapse is a real risk whenever you retrain at a new scale**, not just at
  128 hosts. Always run with `--check-collapse` immediately after training, before
  committing to a full RPC experiment run against a possibly-degenerate model.
- **Trained model files are committed** (`rl_model*.zip`, ten of them, up to ~6 MB for
  the largest), so a fresh clone can reproduce every reported result without retraining.
- **`grep -c` on some shells prints nothing and exits non-zero even with no match**, which
  made "0 errors, 0 overfull boxes" build reports meaningless for a while — they were
  reporting the absence of output, not the absence of errors. An `awk` pass over the same
  log found nine real LaTeX errors, including a `tabular` declared with four columns
  holding five. `paper/checkbuild.py` exists so this cannot recur: it reports pages,
  errors, undefined and multiply-defined references and overfull boxes, and exits non-zero
  on a real problem. The same failure mode bit the findings-file parser in §4.14 — a
  silent empty result is worse than a loud failure, and both fixes are about converting
  one into the other.
- **The raw per-call CSVs in `results_rpc/` are gitignored** (each is ~128 MB). The
  `findings/*.txt` summaries carry every reported number plus the command that produced
  it, so results are reproducible from a clone; the raw records are not.

## 7. Where to Read More

- [EXPERIMENTS.md](EXPERIMENTS.md) — full methodology, every results table, statistical
  tests, and an honest threats-to-validity section.
- [paper/paper.tex](paper/paper.tex) / [paper/references.bib](paper/references.bib) — the
  paper draft built from these results plus the literature review.
