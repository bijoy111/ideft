# iDEFT: Intelligent Backup-Node Selection for Fault-Tolerant NF State Replication

Undergraduate thesis: every major NF state-replication system (OpenNF, Split/Merge,
StatelessNF, S6, Pico Replication, FTMB, REINFORCE, DEFT) replicates a primary network
function's state to a backup instance — yet none of them specify *how that backup is
chosen*. This work closes that gap for [DEFT](https://github.com/MahirSez/DEFT)
(CNSM'23): it designs and rigorously evaluates a multi-criteria greedy heuristic, an
exact Hungarian-optimal assignment, and a PPO reinforcement-learning policy against
DEFT's shipped static mapping and naive baselines, across four topology scales
(9–128 hosts), by driving DEFT's **real, unmodified** two-phase-commit backup
implementation under controlled concurrent load.

**📊 Project website (results, reproduction guide, code walkthrough):**
<https://bijoy111.github.io/ideft/>

## Where everything lives

| Path | What it is |
|---|---|
| [**`backup-node-selection/`**](backup-node-selection/) | **The complete artifact** — start here |
| [`backup-node-selection/code/`](backup-node-selection/code/) | Experiment code, trained RL models, requirements |
| [`backup-node-selection/findings/`](backup-node-selection/findings/) | Saved output of all 8 experiments — read the results without running anything |
| [`backup-node-selection/paper/`](backup-node-selection/paper/) | The paper (LaTeX source + compiled [PDF](backup-node-selection/paper/paper.pdf)) |
| [`backup-node-selection/EXPERIMENTS.md`](backup-node-selection/EXPERIMENTS.md) | Full methodology, all results tables, statistics, threats to validity |
| [`backup-node-selection/IMPLEMENTATION.md`](backup-node-selection/IMPLEMENTATION.md) | File-by-file code walkthrough and known gotchas |
| [`docs/`](docs/) | Source of the project website (GitHub Pages) |

## Quick start

```bash
cd backup-node-selection/code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# sanity check
python3 backup_selector.py --mode heuristic --seed 42 --out /tmp/mapping.json
python3 rpc_contention_experiment.py --context-seeds 42 --trials 1 --calls 50 --burst 4 \
    --methods baseline heuristic
```

Full reproduction commands for every reported table:
[`backup-node-selection/README.md`](backup-node-selection/README.md).

Requirements: Linux, Python 3.10, ~2 GB disk. No root, no Mininet, no SDN controller.
