# iDEFT: Intelligent Backup-Node Selection for Fault-Tolerant NF State Replication

Self-contained reproduction package for the paper/thesis work on backup-node selection
for [DEFT](https://github.com/MahirSez/DEFT) (CNSM'23), a fault-tolerant NF
state-management system. DEFT replicates each primary NF's state to a backup node but
never specifies *how that backup is chosen* — this package implements and evaluates
selection strategies spanning naive baselines (static, worst-case concentrated, random,
round-robin and a fairer randomized-offset variant), decentralized hash placement
(rendezvous and weighted consistent hashing, the NFV analog of CRUSH/Dynamo), a
multi-criteria greedy heuristic, an exact Hungarian-optimal assignment, and trained
reinforcement-learning policies (PPO, with DQN as a cross-algorithm check). All are
measured by driving DEFT's **real, unmodified** two-phase-commit backup implementation
under controlled concurrent load.

Beyond the one-shot selection question, the package also covers: an **online**
formulation in which hosts degrade, recover and fail over time and moving a replica costs
a state transfer, with a competitive-ratio analysis against the exact offline optimum; a
**measurement of that migration cost** against a real connection-tracking flow table,
since DEFT's backup discards the state it receives and so could not supply one; and a
**bare-metal validation** of the largest scale on eleven networked machines, which is
where the single-machine 128-host latency numbers had to be withdrawn.

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
- No root, no Mininet, no SDN controller — everything below runs as ordinary localhost
  processes, with one exception
- **Exception:** the bare-metal 128-host run (§4.11) needs eleven networked machines with
  SSH between them, and root on those machines for `tc`. Everything else, including the
  online analysis and the migration-cost measurement, runs on a laptop.

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

# 128 hosts (§4.6). NOTE: the latency table from this command is WITHDRAWN -- one machine
# cannot host 128 backup servers and the client measuring them, and two identical runs
# differ several-fold. What stands here is structural (distinct backups used, peak load,
# degraded-host avoidance), which is computed from the mapping and does not depend on
# timing. For the latency measurement, see section 6 below.
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 --rl-model rl_model_scaled128_v3.zip
```

### 2. Selection-time overhead (§4.4)

```bash
python3 selection_overhead.py
```

### 3. Cross-algorithm check: PPO vs DQN (§4.10)

Is the RL load-balancing gap specific to PPO, or a property of the problem? DQN runs through
the same environment, reward, and inference path, with only the learner swapped. Both are
scored **structurally** (distinct backups used, peak per-backup load, degraded-host
assignments), which are properties of the mapping rather than of RPC timing, so unlike
latency they are hardware-independent. No RPC servers needed, about a minute:

```bash
python3 rl_algorithm_comparison.py
```

At 128 hosts DQN collapses to 1–3 distinct backups of 80 (vs PPO's 24–30) and puts 172/400
assignments on degraded hosts. DQN can also be driven through the full latency harness with
the `rl_dqn` method:

```bash
rm -f results_rpc/rpc_contention_raw.csv
python3 rpc_contention_experiment.py --context-seeds 1 2 3 --trials 2 --calls 150 --burst 8 \
    --methods heuristic optimal rl rl_dqn \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 \
    --rl-model rl_model_scaled32.zip --rl-dqn-model rl_model_dqn_scaled32_mitigated.zip
```

### 4. Throughput and failure rate (§4.7)

```bash
rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods baseline concentrated random round_robin heuristic optimal rl

rm -f results_rpc/throughput_raw.csv
python3 throughput_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --duration 3.0 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0 --rl-model rl_model_scaled.zip
```

### 5. Online selection under churn (§4.12)

Every scenario above is one-shot. This asks what *adaptation* is worth when hosts degrade,
recover and die over time and moving a replica costs a state transfer. No RPC, no testbed,
about 3.5 minutes:

```bash
python3 competitive_analysis.py     # writes findings/online_competitive.{txt,json}
python3 test_online.py              # proofs, plus the construction behind Theorem 1
```

Two results worth knowing before you run it. Re-solving the one-shot optimum every round —
which is exactly what the `optimal` method above would do if deployed — has **unbounded**
competitive ratio; `test_online.py` measures the predicted 1/ε rate up to 5940×. And the
main empirical finding is **negative**: failover alone (move only when a host dies) closes
91.2% of the distance to the offline optimum, and full degradation-aware hysteresis on top
adds a further 0.13 percentage points.

### 6. Migration cost of real NF state (§4.13)

The online analysis depends on `M`, the cost of moving a replica in rounds of ordinary
replication. The model assumed 3. DEFT's backup discards the state it receives, so there
was no table to measure — this builds one and measures it. No RPC:

```bash
python3 measure_migration_cost.py   # writes findings/migration_cost.{txt,json}
python3 test_flow_state.py          # flow table, deltas, encodings, Little's law
```

Measured `M = 23–41` rounds. It barely moves with encoding or link speed, because a
costlier encoding inflates the full transfer and the per-round delta by the same factor:
`M ≈ 1/delta-share` is a property of traffic churn, not of serialization. Along the way:
DEFT's XML-RPC encoding is 7.31× the size of a packed one, and XML-RPC's `<int>` is signed
32-bit, so an IPv4 address of 128.0.0.0 or above **cannot be sent as an integer at all**.

### 7. Bare-metal 128 hosts (§4.11) — the one step that needs hardware

Eleven machines with SSH between them: ten for backups, one for the client. We used
CloudLab (`c6525-25g`, 16-core EPYC, 25 GbE). Check `nproc` first — `--per-node` must not
exceed physical cores minus the reserve, and on an SMT processor only one thread per
physical core may be used (check `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`).

```bash
# read the plan before launching anything
python3 provision_remote.py \
    --nodes node1 node2 node3 node4 node5 node6 node7 node8 node9 node10 \
    --hosts 128 --per-node 13 --reserve-cores 2 \
    --remote-dir /path/to/code --manifest ../findings/m512.json \
    --backlog 512 --dry-run

# same without --dry-run, then prove each server owns a core
python3 provision_remote.py --manifest ../findings/m512.json --verify
```

**Do not skip `--verify`.** It reports the CPU affinity each backup *actually* holds. A
warning there means the run would measure CPU scheduling rather than backup placement —
the exact failure that forced the §4.6 withdrawal.

```bash
# homogeneous arm
python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 0 --scale-seed 0 \
    --rl-model rl_model_scaled128_v3.zip --manifest ../findings/m512.json

# degraded arm: slow 16 hosts with tc netem on the server's source port, so healthy
# backups on the same NIC are untouched, then tell the experiment it is already physical
python3 apply_degradation.py --manifest ../findings/m512.json \
    --hosts w1 w2 w27 w34 w41 w49 w56 w63 w70 w78 w85 w92 w106 w113 w120 w128 \
    --delay-ms 50 --iface <your 10/25G NIC> --apply

python3 rpc_contention_experiment.py --context-seeds 1 2 3 4 5 --trials 3 --calls 200 --burst 10 \
    --methods concentrated random round_robin heuristic optimal rl \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --degraded-selection spread --physical-degradation \
    --rl-model rl_model_scaled128_v3.zip --manifest ../findings/m512.json

python3 apply_degradation.py --manifest ../findings/m512.json --iface <NIC> --clear
python3 provision_remote.py --manifest ../findings/m512.json --teardown
```

`--degraded-selection spread` is **not optional**. With a contiguous block of degraded
hosts, `round_robin` cycles a lexicographically sorted list and can miss them entirely,
scoring 0% degraded by never arriving rather than by avoiding — and so tying with
`optimal` for the wrong reason (EXPERIMENTS.md §4.11.4).

`--backlog 512` is likewise deliberate. DEFT builds `SimpleXMLRPCServer` with no arguments,
so its accept queue holds five connections; leaving the flag off reproduces that exactly.
Both configurations are worth running — the difference between them is a result, not a
nuisance (EXPERIMENTS.md §4.11.1).

### 8. Regenerating the paper's tables

Every table in the paper is generated from the `findings/*.txt` files, which carry in their
header the exact command that produced them. Nothing is transcribed by hand:

```bash
python3 make_testbed_tables.py      # ../paper/tables_testbed.tex
python3 make_online_tables.py       # ../paper/tables_online.tex
python3 make_migration_tables.py    # ../paper/tables_migration.tex
```

### 9. Retraining the RL models (optional — pre-trained models are included)

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

# --- DQN (§4.10). Pass --algorithm dqn; DQN is tuned via exploration/replay flags
# rather than --ent-coef.
python3 rl_backup_selector.py --train --algorithm dqn --timesteps 50000 --model rl_model_dqn.zip
python3 rl_backup_selector.py --train --algorithm dqn --timesteps 80000 \
    --model rl_model_dqn_scaled.zip \
    --n-hosts 16 --n-primaries 10 --n-degraded 2 --scale-seed 0

# 32 hosts: DQN collapses to 2-4 of 20 backups on SB3 defaults. This mitigated config
# lifts it to 16-19 and is the checkpoint used in §4.10.
python3 rl_backup_selector.py --train --algorithm dqn --check-collapse --timesteps 150000 \
    --model rl_model_dqn_scaled32_mitigated.zip \
    --n-hosts 32 --n-primaries 20 --n-degraded 4 --scale-seed 0 \
    --exploration-fraction 0.7 --exploration-final-eps 0.15 \
    --buffer-size 100000 --target-update-interval 1000

# 128 hosts: still collapses to 3 of 80 even with a wide network and 1M steps. This is
# the cross-algorithm evidence in §4.10, not a config to copy.
python3 rl_backup_selector.py --train --algorithm dqn --check-collapse --timesteps 1000000 \
    --model rl_model_dqn_scaled128.zip \
    --n-hosts 128 --n-primaries 80 --n-degraded 16 --scale-seed 0 \
    --net-arch 256 256 --batch-size 256 --buffer-size 100000 --exploration-fraction 0.5
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
| `rl_backup_selector.py` | RL environment shared by PPO and DQN (`--algorithm ppo\|dqn`): training, inference, collapse diagnostic |
| `rl_algorithm_comparison.py` | Structural PPO-vs-DQN comparison across all four scales (distinct backups, peak load, degraded hits); no RPC needed |
| `rpc_contention_experiment.py` | Main latency harness: real OS process per backup, concurrent bursts, retry-with-backoff, Mann-Whitney + effect sizes + bootstrap CIs |
| `throughput_experiment.py` | Fixed-duration throughput / failure-rate harness |
| `selection_overhead.py` | Per-mapping compute-time benchmark across all four scales |
| `rl_model*.zip` | Pre-trained PPO models (9 / 16 / 32 / 128-host scenarios; the 128-host one is the fixed "v3") |
| `rl_model_dqn*.zip` | Pre-trained DQN models used for the §4.10 cross-algorithm check (the 32-host `_mitigated` one is the post-fix checkpoint; the 128-host one still collapses, by design) |
| `deployment.py` | `Endpoint` plus local and manifest-driven remote backends. The remote one **refuses** a simulated per-call delay rather than silently ignoring it |
| `backup_server.py` | One backup as a standalone process on a routable address; `--backlog` makes DEFT's five-deep accept queue a controlled variable |
| `provision_remote.py` | Places servers on nodes, pins each to a core with `taskset`, writes the manifest; `--verify` reports the affinity actually held |
| `apply_degradation.py` | `tc netem` on the server's source port, so a degraded backup is slowed by the kernel and its healthy co-tenants are not |
| `online_model.py`, `online_selector.py` | The churn process and cost model; the policies plus the exact offline optimum by dynamic programming |
| `competitive_analysis.py` | The online sweeps (churn, persistence, migration cost, pool size, hysteresis threshold) |
| `flow_state.py`, `workload.py` | A connection-tracking flow table with delta replication, and the flow-arrival workload validated against Little's law |
| `measure_migration_cost.py` | Table occupancy, serialized size per encoding, `M` in rounds, and the migration crossover |
| `make_*_tables.py` | Generate the paper's `.tex` tables from the findings files, so no number is transcribed by hand |
| `test_deployment.py`, `test_online.py`, `test_flow_state.py` | Tests for the three new subsystems, including a real trial through the remote path and Theorem 1's adversarial construction |
| `../findings/` | Saved output of every run, each file carrying in its header the command that produced it |
| `EXPERIMENTS.md` | Full methodology and results reference |

## Notes on expected variance

Latency numbers are timing measurements on shared hardware: expect run-to-run variation,
especially in tail percentiles (p99) and in `concentrated`'s extreme-contention numbers
(EXPERIMENTS.md §4.6 documents a case where the failure rate swung between ~1% and zero
across identical runs — that instability is itself one of the findings).

**The single-machine 128-host latency table is withdrawn** (§4.6). Two identical runs
differ several-fold there, because 128 backup servers and the client contend for one
machine's cores. Section 7 above reproduces that scale on eleven machines; the numbers
differ substantially, and in particular most of the apparent cost of concentrating backups
turns out to be DEFT's five-deep accept queue rather than load on the backup. The 9-, 16-
and 32-host results have **not** been re-run on bare metal, so their concentration
penalties should be read as upper bounds.

The online (§4.12) and migration-cost (§4.13) results are deterministic given a seed and
reproduce exactly; they involve no RPC and no timing of shared hardware. The migration
cost does depend on the machine's serialization throughput, so `M` may shift by a few
rounds — but the finding that `M` is 23–41 rather than 3, and that it is roughly
independent of encoding and link speed, is robust. The *qualitative*
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
