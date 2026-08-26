"""Tier 0 of the algorithm ladder: uniform random over the candidate set.

This is the number the DQN has to beat. A reward curve that looks like progress but
ties this is a broken agent, and nothing else in the test suite catches that.
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
from rdkit import Chem

from mol_optim import (
    bindingdb,
    config,
    determinism,
    environment,
    graph_key,
    oracle_gsk3b,
    results,
    reward_pic50,
    rewards,
    seeds,
)


def rollout(cfg: config.Config, reward_fn: Callable[[Chem.Mol], float]) -> results.Run:
    determinism.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    episode_rewards: list[float] = []
    episode_molecules: list = []
    started = time.perf_counter()

    for _ in range(cfg.episodes):
        episode = environment.reset(cfg)
        while True:
            choice = int(rng.integers(len(episode.valid_actions)))
            result = environment.step(episode, choice, reward_fn, cfg)
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
    parser.add_argument(
        "--reward",
        choices=("qed", "gsk3b", "pic50"),
        default="qed",
        help="qed is Step 1-2; gsk3b is the TDC oracle, Step 3; pic50 is Step 5",
    )
    parser.add_argument(
        "--seed-molecule",
        type=int,
        default=None,
        help="index into seeds.choose(); the molecule episodes start from",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.Config.max_steps_per_episode,
        help="edits per episode; 6 for pic50, where 40 leaves the applicability domain",
    )
    parser.add_argument(
        "--regressor", type=Path, default=Path("models/egfr_regressor.pt")
    )
    args = parser.parse_args()

    init_mol = None
    if args.reward == "qed":
        reward_fn = rewards.qed
    elif args.reward == "gsk3b":
        forest = oracle_gsk3b.load()
        reward_fn = lambda mol: oracle_gsk3b.score(forest, mol)  # noqa: E731
    else:
        # Divided by 10 to match train_dqn, so the two numbers compare directly.
        reward = reward_pic50.load(args.regressor)
        reward_fn = lambda mol: reward_pic50.score(reward, mol) / 10.0  # noqa: E731
    if args.seed_molecule is not None:
        init_mol = seeds.choose(bindingdb.load())[args.seed_molecule].mol

    run = rollout(
        config.Config(
            episodes=args.episodes,
            seed=args.seed,
            init_mol=init_mol,
            max_steps_per_episode=args.max_steps,
        ),
        reward_fn,
    )
    best_molecule, best_reward = run.best
    print(f"final_mean_reward {run.final_mean_reward:.4f}  in {run.seconds:.1f}s")
    print(
        f"best: {best_reward:.4f}  "
        f"{best_molecule.GetNumHeavyAtoms()} heavy atoms  "
        f"{graph_key.canonical_hash(best_molecule)}"
    )
