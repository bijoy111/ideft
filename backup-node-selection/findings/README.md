# findings/

Saved output from every experiment, so you can read the results without running anything.
Each file is the full terminal output of one command (per-seed mappings, summary tables,
Mann-Whitney tests, and the `backup_load` breakdown), with a header recording the exact
command and the UTC time it was produced.

**Reproducing overwrites these files in place.** Re-running any command from
[`../README.md`](../README.md) (or the [Reproduce page](https://bijoy111.github.io/ideft/reproduce.html))
regenerates the matching file here with your own numbers. Timing values will differ from
run to run (see "Notes on expected variance" in the top-level README); the qualitative
results are what reproduce.

| File | Command | Paper artifact |
|---|---|---|
| `latency_9hosts.txt` | `rpc_contention_experiment.py` (9-host homogeneous) | Table I, Table III (Homog.) |
| `latency_16hosts.txt` | `rpc_contention_experiment.py --n-hosts 16 …` | Table II, Table III (16-host) |
| `latency_32hosts.txt` | `rpc_contention_experiment.py --n-hosts 32 …` | Table V |
| `latency_128hosts.txt` | `rpc_contention_experiment.py --n-hosts 128 …` | Table VI, §V-F |
| `selection_overhead.txt` | `selection_overhead.py` | Table IV, Figure 1 |
| `throughput_9hosts.txt` | `throughput_experiment.py` (9-host) | Table VII |
| `throughput_16hosts.txt` | `throughput_experiment.py --n-hosts 16 …` | Table VIII |
| `collapse_check_128hosts.txt` | `rl_backup_selector.py --check-collapse …` | Figure 3 |

The raw per-call latencies (for independent re-analysis) are written separately to
`../code/results_rpc/*.csv`; those are large and machine-specific, so they are not
committed.
