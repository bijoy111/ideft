"""Ablation of the RL reward's direct anti-concentration term.

The environment's reward subtracts `overload_penalty` for each primary already
assigned to the chosen backup (rl_backup_selector.BackupSelectionEnv.step). The
question this script answers is whether that term delivers the intended benefit:
does a learned policy spread load across the backup pool because of it, and does
its strength matter as the action space grows?

The measurement is structural: distinct backups used, maximum per-backup load, and
degraded-host assignments. These are properties of the produced mapping, so they
are exact and hardware-independent, unlike per-call RPC latency.

Usage:
    python3 rl_penalty_ablation.py --n-hosts 32 --n-primaries 20 --timesteps 150000
"""

import argparse
import json

import numpy as np

import degradation
import rl_backup_selector
import topology_features


def evaluate(model, hosts, primaries, graph, overrides, degraded, seeds):
    """Structural quality of the policy's mapping across independent resource draws."""
    distinct, peak, deg_hits = [], [], []
    for seed in seeds:
        ctx = topology_features.build_feature_context(
            hosts=hosts, seed=seed, graph=graph, overrides=overrides)
        mapping = rl_backup_selector.predict_mapping(model, hosts, primaries, ctx)
        counts = {}
        for backup in mapping.values():
            counts[backup] = counts.get(backup, 0) + 1
        distinct.append(len(counts))
        peak.append(max(counts.values()))
        deg_hits.append(sum(1 for b in mapping.values() if b in degraded))
    return {
        "distinct_mean": float(np.mean(distinct)),
        "distinct_range": [int(min(distinct)), int(max(distinct))],
        "peak_mean": float(np.mean(peak)),
        "peak_range": [int(min(peak)), int(max(peak))],
        "degraded_mean": float(np.mean(deg_hits)),
        "degraded_total": int(sum(deg_hits)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-hosts", type=int, default=32)
    ap.add_argument("--n-primaries", type=int, default=20)
    ap.add_argument("--n-degraded", type=int, default=4)
    ap.add_argument("--degradation-model", choices=["binary", "realistic"], default="binary")
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.5, 2.0, 5.0, 10.0])
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--net-arch", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1],
                    help="independent training runs per penalty; RL outcomes vary enough "
                         "between seeds that a single run per setting is not evidence")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hosts, primaries, graph = topology_features.generate_scaled_topology(
        n_hosts=args.n_hosts, n_primaries=args.n_primaries, seed=0)
    pool = [h for h in hosts if h not in primaries]
    deg = pool[:args.n_degraded]
    overrides, _, profiles = degradation.build(
        deg, model=args.degradation_model, seed=0,
        healthy_hosts=[h for h in pool if h not in deg])
    degraded = set(deg)

    print("RL anti-concentration penalty ablation")
    print("  topology: {} hosts, {} primaries, {} degraded ({})".format(
        args.n_hosts, len(primaries), len(deg), args.degradation_model))
    print("  training: {} timesteps, ent_coef={}, net_arch={}, seeds={}".format(
        args.timesteps, args.ent_coef, args.net_arch, args.train_seeds))
    print("  evaluated structurally over {} resource draws\n".format(len(args.eval_seeds)))
    print("  the heuristic's reference spread for this topology is {} distinct "
          "backups at peak load 1-2\n".format(len(primaries)))

    header = "{:>8}  {:>6}  {:>16}  {:>14}  {:>12}"
    print(header.format("penalty", "trainsd", "distinct backups", "peak load", "degraded"))
    results = []
    for penalty in args.penalties:
        for tseed in args.train_seeds:
            model = rl_backup_selector.train_ppo(
                hosts, primaries, graph=graph, overrides=overrides,
                total_timesteps=args.timesteps, seed=tseed,
                ent_coef=args.ent_coef, net_arch=list(args.net_arch),
                overload_penalty=penalty)
            r = evaluate(model, hosts, primaries, graph, overrides, degraded, args.eval_seeds)
            r.update(penalty=penalty, train_seed=tseed)
            results.append(r)
            print(header.format(
                "{:.1f}".format(penalty), tseed,
                "{:.1f}  [{}-{}]".format(r["distinct_mean"], *r["distinct_range"]),
                "{:.1f}  [{}-{}]".format(r["peak_mean"], *r["peak_range"]),
                "{:.1f}/{}".format(r["degraded_mean"], len(primaries))))

    print("\nAveraged over training seeds:")
    print("{:>8}  {:>16}  {:>12}  {:>12}".format("penalty", "distinct backups", "peak load", "degraded"))
    for penalty in args.penalties:
        rows = [r for r in results if r["penalty"] == penalty]
        print("{:>8}  {:>16.1f}  {:>12.1f}  {:>12.2f}".format(
            "{:.1f}".format(penalty),
            float(np.mean([r["distinct_mean"] for r in rows])),
            float(np.mean([r["peak_mean"] for r in rows])),
            float(np.mean([r["degraded_mean"] for r in rows]))))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
        print("\nwrote {}".format(args.out))


if __name__ == "__main__":
    main()
