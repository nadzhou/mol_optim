"""Evolutionary search on the same action space — the search-strength baseline.

Same MDP, same candidate sets, same edit budget and the same number of reward
evaluations as dqn.py. The only thing that changes is what picks the next molecule: a
population under truncation selection instead of a Q network. A gap between this and the
DQN is therefore the value of learning with the action space held fixed, which is the
question docs/where_this_stands.md is left with after the halogen run took the ceiling up
and the catch down.

An individual is the *path* from the seed, not the molecule it ends on. That is what keeps
the search inside the DQN's reachable set: every individual is exactly
max_steps_per_episode edits from the seed, and a mutation re-rolls a suffix of the path
rather than editing the endpoint, which would wander further out every generation.
"""

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rdkit import Chem

from mol_optim import config, determinism
from mol_optim.chem import graph_key, seeds
from mol_optim.env import environment, rewards
from mol_optim.report import results

# The DQN spends cfg.episodes reward evaluations, one terminal molecule each. Splitting
# the same budget 50 wide and cfg.episodes // 50 deep is what makes the two comparable;
# it is not a tuned number and there is no config knob for it, because the moment it
# becomes one the budgets stop matching by construction.
POPULATION = 50


def _random_path(
    prefix: Sequence[Chem.Mol],
    reward_fn: Callable[[Chem.Mol], float],
    cfg: config.Config,
    rng: np.random.Generator,
) -> tuple[tuple[Chem.Mol, ...], float]:
    """Extends a prefix of a path with uniform random edits, to the full edit budget.

    Returns the whole path and the terminal reward. An empty prefix is a fresh individual;
    a prefix of length i is what a mutation at position i re-rolls from.
    """
    state = cfg.init_mol if not prefix else prefix[-1]
    episode = environment.Episode(
        state=state,
        num_steps_taken=len(prefix),
        valid_actions=environment.valid_actions(state, cfg),
    )
    path = list(prefix)
    reward = 0.0
    while True:
        choice = int(rng.integers(len(episode.valid_actions)))
        result = environment.step(episode, choice, reward_fn, cfg)
        path.append(result.state)
        reward = result.reward
        if result.terminated:
            return tuple(path), reward


def search(
    cfg: config.Config,
    reward_fn: Callable[[Chem.Mol], float],
    log_path: Path | None = None,
    report_every: int = 0,
) -> results.Run:
    determinism.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generations = cfg.episodes // POPULATION

    log_file = open(log_path, "w") if log_path is not None else None
    if log_file is not None:
        log_file.write("episode,reward,generation,graph_hash\n")

    evaluated_rewards: list[float] = []
    evaluated_molecules: list[Chem.Mol] = []
    started = time.perf_counter()

    population = [_random_path((), reward_fn, cfg, rng) for _ in range(POPULATION)]
    for generation in range(generations):
        for path, reward in population:
            evaluated_rewards.append(reward)
            evaluated_molecules.append(path[-1])
            if log_file is not None:
                log_file.write(
                    f"{len(evaluated_rewards) - 1},{reward:.6f},{generation},"
                    f"{graph_key.canonical_hash(path[-1])}\n"
                )
        if report_every and generation % report_every == 0:
            print(
                f"generation {generation:>4}  best {max(r for _, r in population):.4f}  "
                f"mean {sum(r for _, r in population) / POPULATION:.4f}",
                flush=True,
            )
        if generation + 1 == generations:
            break

        # Truncation selection: the better half survive, and each empty slot is one
        # survivor with a random suffix of its path re-rolled. Mutating at position 0
        # is a fresh random individual, which is the only immigration there is.
        population.sort(key=lambda individual: -individual[1])
        survivors = population[: POPULATION // 2]
        population = list(survivors)
        while len(population) < POPULATION:
            parent = survivors[int(rng.integers(len(survivors)))][0]
            cut = int(rng.integers(cfg.max_steps_per_episode))
            population.append(_random_path(parent[:cut], reward_fn, cfg, rng))

    if log_file is not None:
        log_file.close()
    return results.Run(
        episode_rewards=tuple(evaluated_rewards),
        episode_molecules=tuple(evaluated_molecules),
        seconds=time.perf_counter() - started,
    )


def run(settings: config.Settings, spec: config.AgentSpec) -> results.Run:
    reward = rewards.load(spec.regressor, settings.bindingdb.path)
    reward_fn = lambda mol: rewards.score(reward, mol) / rewards.PIC50_SCALE
    init_mol = seeds.molecule(settings.bindingdb.path, spec.seed_molecule)
    if init_mol is not None:
        print(
            f"starting from seed {spec.seed_molecule}: "
            f"{init_mol.GetNumHeavyAtoms()} heavy atoms, reward {reward_fn(init_mol):.4f}"
        )
    return search(
        replace(spec.cfg, init_mol=init_mol),
        reward_fn,
        log_path=settings.runs / f"{spec.name}.csv",
        report_every=spec.report_every,
    )
