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
optimal assignment (Hungarian algorithm), and a trained RL (PPO) policy, evaluated against
the static baseline and three naive baselines (random, round-robin, worst-case
concentrated).

## 3. Repository Layout: New and Modified Files

```
code/
├── config.py                        (MODIFIED — one-line bugfix)
├── topology_features.py             (NEW — real-topology graph + synthetic resources)
├── backup_selector.py               (NEW — baseline/concentrated/random/round_robin/heuristic/optimal)
├── rl_backup_selector.py            (NEW — PPO environment, training, inference)
├── rpc_contention_experiment.py     (NEW — main latency evaluation harness)
├── throughput_experiment.py         (NEW — throughput/failure-rate evaluation harness)
├── selection_overhead.py            (NEW — per-mapping compute-time benchmark)
├── requirements.txt                 (MODIFIED — added networkx, scipy, gymnasium, stable-baselines3)
├── exp_package/                     (DEFT's real, unmodified 2PC Primary/Backup classes)
├── primary_to_secondary.json        (DEFT's shipped static 9-host mapping)
├── rl_model.zip                     (trained model — 9-host homogeneous)
├── rl_model_scaled.zip              (trained model — 16-host scaled+heterogeneous)
├── rl_model_scaled32.zip            (trained model — 32-host scaled+heterogeneous)
├── rl_model_scaled128_collapsed.zip (trained model — 128-host degenerate baseline, ent 0 / [64,64]; Figure 3 low bar)
├── rl_model_scaled128_v3.zip        (trained model — 128-host, FINAL fix, used in reported results)
└── results_rpc/                     (output — CSVs written by the experiments above)

EXPERIMENTS.md                        (methodology + results narrative)
paper/                                (LaTeX source, figures, compiled paper.pdf)
IMPLEMENTATION.md                     (this file)
README.md                             (setup + reproduction commands)
```

Two 128-host checkpoints ship: `rl_model_scaled128_collapsed.zip` (the degenerate
`ent_coef=0.0`/`[64,64]` baseline, the low bar of Figure 3) and
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

The RL agent: environment, training, and inference.

- `BackupSelectionEnv(gym.Env)` — one episode assigns every primary a backup in
  sequence, under one freshly-drawn random `FeatureContext` per episode (so the policy
  generalizes across resource realizations, not one memorized instance). Observation =
  per-host `[health, capacity_frac, cpu, storage]` concatenated with a one-hot encoding
  of the current primary. Action space = `Discrete(n_hosts)`. Reward = the same
  `score_candidates` value the heuristic uses for a feasible action, an overload penalty
  for re-using an already-loaded host, and a large fixed penalty (`-10`) for an
  infeasible action — chosen so invalid actions are discouraged **during training**,
  not just corrected after the fact.
- `train(hosts, primaries, ..., ent_coef=0.01, net_arch=None, n_steps=2048, batch_size=64)` —
  trains a PPO agent (`stable_baselines3`). **`ent_coef` defaults to 0.01, not SB3's own
  default of 0.0** — this was added specifically after the 128-host model trained with
  `ent_coef=0.0` collapsed to a degenerate constant policy (see §4.9 below); with no
  entropy bonus, nothing discourages the policy from locking onto one action early.
  `net_arch` defaults to `[128, 128]` (SB3's own default is `[64, 64]`) for the same
  reason — a bigger action/observation space needs more network capacity.
- `predict_mapping(model, hosts, primaries, ctx)` — deterministic policy rollout,
  falling back to the best-scoring feasible candidate if the policy still picks an
  infeasible host at inference time (a true fallback now, not the only safety net).
- CLI: `--train`, `--predict`, `--timesteps`, `--model`, `--seed`, `--n-hosts`,
  `--n-primaries`, `--n-degraded`, `--scale-seed`, `--ent-coef`, `--net-arch` (space-separated
  layer sizes), `--n-steps`, `--batch-size`, and `--check-collapse` — the last one prints
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
| `rl_model_scaled128_collapsed.zip` | 128 hosts, **degenerate baseline** | `--timesteps 400000 --ent-coef 0.0 --net-arch 64 64` (SB3 raw defaults) | **Collapsed**: uses only 1–5 distinct backups, routing ~all 80 primaries to one host (`w101`). Shipped for reproducibility of the collapse finding (Figure 3 low bar), not usable in production. |
| *(intermediate attempt, not shipped)* | 128 hosts, **fix attempt 1** | `--timesteps 600000 --ent-coef 0.01 --net-arch 128 128` | Partial improvement only; no checkpoint released. |
| `rl_model_scaled128_v3.zip` | 128 hosts, **final** | `--timesteps 1000000 --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256` | Used in all reported 128-host `rl` results. Avoids degraded hosts correctly (0/400) and spreads across 24–30 distinct backups, but still load-balances worse than the heuristic (74–76) — see EXPERIMENTS.md §4.6. |

Use `--check-collapse` on any of these (or right after `--train`) to see this diagnosis
yourself quickly, instead of discovering it part-way through a full RPC experiment run.

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

Steps 6–9 each start by deleting `results_rpc/rpc_contention_raw.csv` (`rm -f`) before
running, since every run appends to that one file — copy it aside first if you want to
keep more than one scenario's raw per-call data at once. Same for
`results_rpc/throughput_raw.csv` between steps 11 and 12. The exact `rm -f` + command
pairs are spelled out in EXPERIMENTS.md §6 and in §1 above.

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
  `concentrated` with ~80 primaries on one backup), this is a real, load-bearing finding
  (see EXPERIMENTS.md §4.6), not a bug to "fix" by raising the backlog — raising it would
  change what DEFT's actual shipped implementation is being measured against.
  `rpc_contention_experiment.py`'s retry-with-backoff (§4.5) exists specifically to
  survive this without crashing the harness, while still recording the resulting failures
  as data.
- **RL policy collapse is a real risk whenever you retrain at a new scale**, not just at
  128 hosts. Always run with `--check-collapse` immediately after training, before
  committing to a full RPC experiment run against a possibly-degenerate model.
- **Trained model files are currently untracked in git** (not yet committed). Decide
  whether to commit them (`rl_model*.zip`, up to ~6 MB for the largest) or add them to
  `.gitignore` and have readers regenerate them via §5 steps 2–5 — either is reasonable,
  but pick one explicitly so a fresh clone isn't missing files silently.

## 7. Where to Read More

- [EXPERIMENTS.md](EXPERIMENTS.md) — full methodology, every results table, statistical
  tests, and an honest threats-to-validity section.
- [paper/paper.tex](paper/paper.tex) / [paper/references.bib](paper/references.bib) — the
  paper draft built from these results plus the literature review.
