"""Tier 0 of the algorithm ladder: uniform random over the candidate set.

This is the number the DQN has to beat. A reward curve that looks like progress but
ties this is a broken agent, and nothing else in the test suite catches that.
"""

import time

import numpy as np

from mol_optim import config, determinism, environment, graph_key, report, results, rewards


def rollout(cfg: config.Config) -> results.Run:
    determinism.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    episode_rewards: list[float] = []
    episode_molecules: list = []
    started = time.perf_counter()

    for _ in range(cfg.episodes):
        episode = environment.reset(cfg)
        while True:
            choice = int(rng.integers(len(episode.valid_actions)))
            result = environment.step(episode, choice, rewards.qed, cfg)
            if result.terminated:
                break
        episode_rewards.append(result.reward)
        episode_molecules.append(result.state)

    return results.Run(
        episode_rewards=tuple(episode_rewards),
        episode_molecules=tuple(episode_molecules),
        seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = rollout(config.Config(episodes=args.episodes, seed=args.seed))
    best_molecule, best_reward = run.best
    print(f"final_mean_reward {run.final_mean_reward:.4f}  in {run.seconds:.1f}s")
    print(
        f"best: {best_reward:.4f}  "
        f"{best_molecule.GetNumHeavyAtoms()} heavy atoms  "
        f"{graph_key.canonical_hash(best_molecule)}"
    )
