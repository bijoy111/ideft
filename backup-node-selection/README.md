# iDEFT: Intelligent Backup-Node Selection for Fault-Tolerant NF State Replication

Self-contained reproduction package for the paper/thesis work on backup-node selection
for [DEFT](https://github.com/MahirSez/DEFT) (CNSM'23), a fault-tolerant NF
state-management system. DEFT replicates each primary NF's state to a backup node but
never specifies *how that backup is chosen* — this package implements and evaluates six
selection strategies (static baseline, worst-case concentrated, random, round-robin, a
multi-criteria greedy heuristic, an exact Hungarian-optimal assignment) plus a trained
PPO reinforcement-learning policy, and measures them by driving DEFT's **real,
unmodified** two-phase-commit backup implementation under controlled concurrent load.

## Layout

| Path | What it is |
|---|---|
| [`code/`](code/) | All experiment code, pre-trained RL models, `requirements.txt` |
| [`paper/`](paper/) | The paper: LaTeX source, bibliography, figures, compiled [`paper.pdf`](paper/paper.pdf) |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Full methodology, every results table, statistical tests, threats to validity |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | File-by-file code walkthrough, design rationale, known gotchas |

All commands below are run from inside `code/`.

## Requirements

- Linux (tested on Ubuntu under WSL2), Python 3.10
- ~2 GB disk for the Python environment (PyTorch dominates)
- No root, no Mininet, no SDN controller — everything runs as ordinary localhost processes

## Setup

```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick check that it works

```bash
python3 backup_selector.py --mode heuristic --seed 42 --out /tmp/mapping.json
python3 rpc_contention_experiment.py --context-seeds 42 --trials 1 --calls 50 --burst 4 \
    --methods baseline heuristic
```

You should see per-method mapping printouts, then a summary table with latency
statistics and a Mann-Whitney comparison.

## Regenerating the reported results

Pre-trained RL models are included (`rl_model*.zip`), so you can run all experiments
without retraining. **Important:** the harnesses *append* to `results_rpc/*.csv` and the
end-of-run breakdown reads the whole file — delete the CSV between scenarios (as shown)
or results will mix.

### 1. Per-call latency (EXPERIMENTS.md §4.1, §4.2, §4.5, §4.6)

```bash
# 9 hosts, homogeneous (§4.1)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl

# 16 hosts, heterogeneous (§4.2)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip

# 32 hosts (§4.5)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 --rl-model rl_model_scaled32.zip

# 128 hosts (§4.6)
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 --rl-model rl_model_scaled128_v3.zip
```

### 2. Selection-time overhead (§4.4)

```bash
python3 selection_overhead.py
```

### 3. Throughput and failure rate (§4.7)

```bash
rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl

rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip
```

### 4. Retraining the RL models (optional — pre-trained models are included)

```bash
python3 rl_backup_selector.py --train --timesteps 50000  --model rl_model.zip
python3 rl_backup_selector.py --train --timesteps 80000  --model rl_model_scaled.zip \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0
python3 rl_backup_selector.py --train --timesteps 100000 --model rl_model_scaled32.zip \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0

# 128 hosts: stable-baselines3's PPO defaults (ent_coef=0.0, net_arch=[64,64]) collapse
# to a degenerate policy at this scale (EXPERIMENTS.md §4.6). To reproduce that finding:
python3 rl_backup_selector.py --train --check-collapse --timesteps 400000 \
    --model rl_model_scaled128_collapsed.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --ent-coef 0.0 --net-arch 64 64

# The fix actually used for the reported 128-host rl results:
python3 rl_backup_selector.py --train --check-collapse --timesteps 1000000 \
    --model rl_model_scaled128_v3.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --ent-coef 0.02 --net-arch 256 256 --n-steps 4096 --batch-size 256
```

Always pass `--check-collapse` after training at a new scale — it prints how many
distinct backups the policy actually uses across 5 seeds, catching a degenerate policy
straight away instead of part-way through an experiment run.

## What's in this folder

| File | Role |
|---|---|
| `config.py`, `primary_to_secondary.json` | DEFT's real 9-host topology definition and its shipped static backup mapping |
| `exp_package/Two_phase_commit/` | DEFT's **unmodified** 2PC `Backup`/`Primary` classes — the actual system under test |
| `topology_features.py` | Topology graph + seeded synthetic per-host resources (health/capacity/CPU/storage), scaled-topology generator |
| `backup_selector.py` | All non-learned methods: baseline, concentrated, random, round_robin, heuristic, optimal (Hungarian) |
| `rl_backup_selector.py` | PPO environment, training (with entropy/net-arch knobs), inference, collapse diagnostic |
| `rpc_contention_experiment.py` | Main latency harness: real OS process per backup, concurrent bursts, retry-with-backoff, Mann-Whitney + effect sizes + bootstrap CIs |
| `throughput_experiment.py` | Fixed-duration throughput / failure-rate harness |
| `selection_overhead.py` | Per-mapping compute-time benchmark across all four scales |
| `rl_model*.zip` | Pre-trained PPO models (9 / 16 / 32 / 128-host scenarios; the 128-host one is the fixed "v3") |
| `EXPERIMENTS.md` | Full methodology and results reference |

## Notes on expected variance

Latency numbers are timing measurements on shared hardware: expect run-to-run variation,
especially in tail percentiles (p99) and in `concentrated`'s extreme-contention numbers
(EXPERIMENTS.md §4.6 documents a case where the failure rate swung between ~1% and zero
across identical runs — that instability is itself one of the findings). The *qualitative*
results — degraded-host avoidance counts, method rankings by effect size, the monotonic
latency-vs-backup_load relationship — are what reproduce reliably; exact milliseconds
are not expected to match to the decimal.

**One known non-reproducible verdict:** the 9-host *throughput* comparison
(`throughput_experiment.py`, EXPERIMENTS.md §4.7) rests on only 15 aggregate
measurements per method and is dominated by background system load, not by the mapping
under test. In the committed run only 3 of the 21 pairwise comparisons reached p<0.05 (all
three involving `heuristic`, small effects), and none is a stable ordering. Whatever you
get here — significant or not, in either direction — is expected, and is not a failed
reproduction; the one consistent signal is failure rate (`concentrated` highest at 0.70%,
feature-aware methods 0.07–0.14%).

At **16 hosts**, throughput splits into a reproducible half and a fragile half. The
`concentrated` comparisons are the single most robust result in the package: r ∈ [0.63,
0.78] with every p < 0.004, and a failure rate (2.57%) clearly elevated above every other
method's (0.16–0.62%). Expect those every time. The `round_robin` vs
`heuristic`/`optimal`/`rl` comparisons, by contrast, only partially resolve: `optimal` and
`rl` significantly out-throughput `round_robin` (r ≈ 0.63–0.65, p < 0.004), but `heuristic`
vs `round_robin` does not reach significance (p = 0.38) — also expected, for the reason
explained in EXPERIMENTS.md §4.7.
