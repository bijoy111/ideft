"""RL-based backup-node selector (PPO and DQN).

Design:
- Reward uses the real per-pair latency from topology_features (ctx.distances).
- Invalid actions (self, non-eligible hosts) are penalized during training via a large
  negative reward, not just corrected at inference, so the policy learns to avoid them.
- Reuses backup_selector.score_candidates() and topology_features.build_feature_context()
  directly, so the agent optimizes the same objective as the heuristic/optimal baselines,
  which keeps the comparison fair.
- Trains across many random FeatureContext realizations per episode (not one fixed
  scenario), so the learned policy generalizes instead of memorizing one instance.
- DQN is trained and evaluated through the same BackupSelectionEnv and predict_mapping()
  as PPO, so the two algorithms are compared on identical footing; only the learning
  algorithm and its own hyperparameters differ (see train_dqn()).

Usage:
    python rl_backup_selector.py --train --algorithm ppo --timesteps 50000 --model rl_model.zip
    python rl_backup_selector.py --train --algorithm dqn --timesteps 50000 --model rl_model_dqn.zip
    python rl_backup_selector.py --predict --algorithm ppo --model rl_model.zip --seed 42
    python rl_backup_selector.py --predict --algorithm dqn --model rl_model_dqn.zip --seed 42
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO

import backup_selector
import config
import topology_features

N_FEATURES_PER_HOST = 4  # health, capacity_frac, cpu_availability, storage_availability
INVALID_ACTION_PENALTY = -10.0
OVERLOAD_PENALTY_PER_EXTRA_PRIMARY = 2.0


class BackupSelectionEnv(gym.Env):
    """One episode = one full round of assigning every primary a backup, under one
    random FeatureContext realization. Action space is Discrete(n_hosts); the agent
    picks a host index for the "current" primary each step."""

    def __init__(self, hosts, primaries, graph=None, weights=None, overrides=None):
        super().__init__()
        self.hosts = hosts
        self.primaries = primaries
        self.graph = graph
        self.overrides = overrides
        self.weights = weights or backup_selector.DEFAULT_WEIGHTS
        self.n_hosts = len(hosts)
        self.host_index = {h: i for i, h in enumerate(hosts)}

        self.action_space = spaces.Discrete(self.n_hosts)
        obs_dim = self.n_hosts * N_FEATURES_PER_HOST + self.n_hosts
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.ctx = None
        self.current_primary_idx = 0
        self.load_count = None

    def _build_observation(self):
        features = np.zeros((self.n_hosts, N_FEATURES_PER_HOST), dtype=np.float32)
        for host, idx in self.host_index.items():
            state = self.ctx.resources.get(host)
            if state is not None:
                features[idx] = [
                    state.health,
                    state.capacity / state.max_capacity if state.max_capacity else 0.0,
                    state.cpu_availability,
                    state.storage_availability,
                ]

        primary_onehot = np.zeros(self.n_hosts, dtype=np.float32)
        if self.current_primary_idx < len(self.primaries):
            primary_name = self.primaries[self.current_primary_idx]
            primary_onehot[self.host_index[primary_name]] = 1.0

        return np.concatenate([features.flatten(), primary_onehot])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.ctx = topology_features.build_feature_context(
            hosts=self.hosts, seed=episode_seed, graph=self.graph, overrides=self.overrides
        )
        self.current_primary_idx = 0
        self.load_count = {h: 0 for h in self.hosts}
        return self._build_observation(), {}

    def step(self, action):
        primary = self.primaries[self.current_primary_idx]
        chosen_host = self.hosts[int(action)]

        feasible = backup_selector.feasible_backups(primary, self.hosts)
        if chosen_host not in feasible:
            reward = INVALID_ACTION_PENALTY
        else:
            scores = backup_selector.score_candidates(primary, feasible, self.ctx, self.weights)
            reward = scores[chosen_host]

            self.load_count[chosen_host] += 1
            if self.load_count[chosen_host] > 1:
                reward -= OVERLOAD_PENALTY_PER_EXTRA_PRIMARY * (self.load_count[chosen_host] - 1)

            backup_selector._apply_load_penalty(self.ctx.resources[chosen_host])

        self.current_primary_idx += 1
        done = self.current_primary_idx >= len(self.primaries)
        obs = self._build_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {}


def train_ppo(hosts, primaries, graph=None, overrides=None, total_timesteps=50000, seed=0, verbose=0,
              ent_coef=0.01, net_arch=None, n_steps=2048, batch_size=64):
    """ent_coef defaults to 0.01 (SB3's PPO default is 0.0). With no entropy bonus nothing
    stops the policy from collapsing onto a single action once it looks decent on average,
    which is what the 128-host model does at ent_coef=0.0 (see EXPERIMENTS.md Sec 4.6).
    net_arch defaults to [128, 128] (SB3's MlpPolicy default is [64, 64]), since larger
    action/observation spaces need more capacity to represent a policy conditioned on which
    of many one-hot primaries is active."""
    env = BackupSelectionEnv(hosts, primaries, graph=graph, overrides=overrides)
    policy_kwargs = dict(net_arch=net_arch or [128, 128])
    model = PPO(
        "MlpPolicy", env, verbose=verbose, seed=seed,
        ent_coef=ent_coef, policy_kwargs=policy_kwargs,
        n_steps=n_steps, batch_size=batch_size,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def train_dqn(hosts, primaries, graph=None, overrides=None, total_timesteps=50000, seed=0, verbose=0,
              learning_rate=1e-4, buffer_size=50000, batch_size=64, net_arch=None,
              exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=500):
    """DQN training with an off-policy replay buffer and epsilon-greedy exploration, the
    architecturally distinct counterpart to PPO's on-policy actor-critic design (off-policy
    vs. on-policy, value-based vs. actor-critic, epsilon-greedy vs. entropy-regularized
    exploration). Used to test whether PPO's policy-collapse behavior at large host counts
    is a PPO-specific artifact or a property of the problem itself.

    exploration_fraction/exploration_final_eps play the same role DQN's entropy coefficient
    would for PPO: without enough exploration, nothing stops the policy from locking onto a
    single action once it looks decent on average. net_arch defaults to [128, 128] for the
    same reason as PPO's."""
    env = BackupSelectionEnv(hosts, primaries, graph=graph, overrides=overrides)
    policy_kwargs = dict(net_arch=net_arch or [128, 128])
    model = DQN(
        "MlpPolicy", env, verbose=verbose, seed=seed,
        learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=1000,
        batch_size=batch_size, gamma=0.99, train_freq=4, gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def load_rl_model(model_path, algorithm="ppo"):
    """Load a trained RL model from disk. Both PPO and DQN expose the same
    model.predict(obs, deterministic=True) interface, so predict_mapping() below needs no
    algorithm-specific handling once the model is loaded."""
    if algorithm == "dqn":
        return DQN.load(model_path)
    return PPO.load(model_path)


def predict_mapping(model, hosts, primaries, ctx, weights=None):
    """Runs the trained policy deterministically over one FeatureContext to produce a full
    {primary: backup} mapping. Falls back to the best feasible candidate by score in the
    rare case the trained policy still picks an invalid host."""
    weights = weights or backup_selector.DEFAULT_WEIGHTS
    env = BackupSelectionEnv(hosts, primaries)
    env.ctx = ctx
    env.current_primary_idx = 0
    env.load_count = {h: 0 for h in hosts}

    mapping = {}
    for i, primary in enumerate(primaries):
        env.current_primary_idx = i
        obs = env._build_observation()
        action, _ = model.predict(obs, deterministic=True)
        chosen_host = hosts[int(action)]

        feasible = backup_selector.feasible_backups(primary, hosts)
        if chosen_host not in feasible:
            scores = backup_selector.score_candidates(primary, feasible, ctx, weights)
            chosen_host = max(feasible, key=lambda c: scores[c])

        mapping[primary] = chosen_host
        backup_selector._apply_load_penalty(ctx.resources[chosen_host])

    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo",
                         help="RL algorithm to train/load (default: ppo)")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--model", default="rl_model.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="primary_to_secondary.json")
    parser.add_argument("--n-hosts", type=int, default=None, help="train/predict on a scaled synthetic topology instead of config.py's 9 hosts")
    parser.add_argument("--n-primaries", type=int, default=None)
    parser.add_argument("--n-degraded", type=int, default=0)
    parser.add_argument("--scale-seed", type=int, default=0)
    # PPO-specific hyperparameters.
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    # DQN-specific hyperparameters.
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--check-collapse", action="store_true",
                         help="after training/loading, print how many distinct backups the policy uses across several context seeds")
    args = parser.parse_args()

    graph = None
    overrides = None
    if args.n_hosts is not None:
        hosts, primaries, graph = topology_features.generate_scaled_topology(
            n_hosts=args.n_hosts, n_primaries=args.n_primaries, seed=args.scale_seed
        )
        if args.n_degraded > 0:
            backup_pool = [h for h in hosts if h not in primaries]
            degraded = backup_pool[:args.n_degraded]
            overrides = {h: {"health": 0.05, "capacity": 2, "max_capacity": 60} for h in degraded}
    else:
        hosts, primaries = config.HOSTS, config.PRIMARIES

    if args.train:
        if args.algorithm == "dqn":
            model = train_dqn(
                hosts, primaries, graph=graph, overrides=overrides, total_timesteps=args.timesteps,
                seed=args.seed, verbose=1, learning_rate=args.learning_rate,
                buffer_size=args.buffer_size, batch_size=args.batch_size, net_arch=args.net_arch,
                exploration_fraction=args.exploration_fraction,
                exploration_final_eps=args.exploration_final_eps,
                target_update_interval=args.target_update_interval,
            )
        else:
            model = train_ppo(
                hosts, primaries, graph=graph, overrides=overrides, total_timesteps=args.timesteps,
                seed=args.seed, verbose=1, ent_coef=args.ent_coef, net_arch=args.net_arch,
                n_steps=args.n_steps, batch_size=args.batch_size,
            )
        model.save(args.model)
        print("Saved trained {} model to {}".format(args.algorithm.upper(), args.model))

    if args.check_collapse:
        model = model if args.train else load_rl_model(args.model, args.algorithm)
        header = "--- collapse check: distinct backups used per context seed ---"
        print("\n" + header)
        lines = [header]
        for check_seed in range(5):
            ctx = topology_features.build_feature_context(hosts=hosts, seed=check_seed, graph=graph, overrides=overrides)
            mapping = predict_mapping(model, hosts, primaries, ctx)
            distinct = len(set(mapping.values()))
            line = "  seed={}: {} distinct backups used out of {} primaries (mapping sample: {})".format(
                check_seed, distinct, len(primaries), dict(list(mapping.items())[:5]))
            print(line)
            lines.append(line)
        # Write the collapse-check summary to findings/ directly (not teed), so a combined
        # --train --check-collapse run does not dump its training logs into the file.
        import run_logging
        run_logging.write_findings(
            "collapse_check_{}hosts".format(args.n_hosts or 9), "\n".join(lines))

    if args.predict:
        model = load_rl_model(args.model, args.algorithm)
        ctx = topology_features.build_feature_context(hosts=hosts, seed=args.seed, graph=graph, overrides=overrides)
        mapping = predict_mapping(model, hosts, primaries, ctx)
        backup_selector.write_mapping(mapping, args.out)
        print("[rl] wrote {}: {}".format(args.out, mapping))


if __name__ == "__main__":
    main()
