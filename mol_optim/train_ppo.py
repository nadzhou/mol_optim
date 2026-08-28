"""PPO on the molecule MDP — rollout, advantage, clipped update, flat and in order.

The point of comparison. DQN and PPO run the same environment, the same reward and the
same encoder, so a difference between what they build is a property of the algorithm
and a similarity is a property of the reward surface. That is the question this file
exists to answer; a higher reward curve on its own is not evidence of anything, which
is what `results/README.md` is about.

    python -m mol_optim.train_ppo --updates 60 --reward pic50 --seed-molecule 0
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rdkit import Chem

from mol_optim import (
    bindingdb,
    config,
    determinism,
    environment,
    featurize,
    graph_key,
    ppo,
    pretrain,
    results,
    reward_pic50,
    rewards,
    seeds,
)


class Rollout:
    """One batch of episodes, flattened to steps. Plain lists; the update reads them."""

    def __init__(self) -> None:
        self.candidates: list[featurize.Graphs] = []  # the set offered at each step
        self.states: list[featurize.Graphs] = []  # the state the step was taken from
        self.choices: list[int] = []  # index into that step's candidate set
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.state_steps_remaining: list[float] = []
        self.candidate_steps_remaining: list[float] = []
        self.advantages = np.zeros(0, dtype=np.float32)
        self.returns = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.choices)


def collect(
    network: ppo.MolPPO,
    cfg: config.Config,
    ppo_cfg: config.PPOConfig,
    reward_fn: Callable[[Chem.Mol], float],
    rng: np.random.Generator,
) -> tuple[Rollout, list[float], list[Chem.Mol]]:
    """Run `rollout_episodes` episodes under the current policy, storing every step."""
    rollout = Rollout()
    episode_rewards: list[float] = []
    episode_molecules: list[Chem.Mol] = []

    for _ in range(ppo_cfg.rollout_episodes):
        episode = environment.reset(cfg)
        # Values are per episode so the GAE recursion below can walk one episode
        # backwards without stepping over the boundary into the previous one.
        first = len(rollout)
        candidates = featurize.graphs(episode.valid_actions)

        while True:
            steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken
            state_graphs = featurize.graphs((episode.state,))

            with torch.no_grad():
                logits = network.logits(
                    featurize.tensors(candidates, steps_remaining, cfg)
                )  # [num_candidates]
                log_probs = torch.log_softmax(logits, dim=0)
                choice = int(torch.multinomial(log_probs.exp(), 1))
                value = float(
                    network.values(
                        featurize.tensors(state_graphs, steps_remaining, cfg)
                    )[0]
                )

            result = environment.step(episode, choice, reward_fn, cfg)

            rollout.candidates.append(candidates)
            rollout.states.append(state_graphs)
            rollout.choices.append(choice)
            rollout.log_probs.append(float(log_probs[choice]))
            rollout.values.append(value)
            rollout.rewards.append(result.reward)
            rollout.dones.append(result.terminated)
            rollout.state_steps_remaining.append(float(steps_remaining))
            rollout.candidate_steps_remaining.append(float(steps_remaining))

            candidates = featurize.graphs(episode.valid_actions)
            if result.terminated:
                break

        episode_rewards.append(result.reward)
        episode_molecules.append(result.state)
        _finish_episode(rollout, first, cfg, ppo_cfg)

    return rollout, episode_rewards, episode_molecules


def _finish_episode(
    rollout: Rollout, first: int, cfg: config.Config, ppo_cfg: config.PPOConfig
) -> None:
    """GAE over the episode that just ended, appended to the rollout's arrays.

    The value of the state after the last step is zero: the episode is over, and the
    environment has already discounted the terminal reward by steps remaining.
    """
    values = np.array(rollout.values[first:], dtype=np.float32)
    step_rewards = np.array(rollout.rewards[first:], dtype=np.float32)
    next_values = np.append(values[1:], 0.0)

    deltas = step_rewards + cfg.gamma * next_values - values
    advantages = np.zeros_like(deltas)
    running = 0.0
    for t in range(len(deltas) - 1, -1, -1):
        running = deltas[t] + cfg.gamma * ppo_cfg.gae_lambda * running
        advantages[t] = running

    rollout.advantages = np.concatenate([rollout.advantages, advantages])
    rollout.returns = np.concatenate([rollout.returns, advantages + values])


def update(
    network: ppo.MolPPO,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    cfg: config.Config,
    ppo_cfg: config.PPOConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Clipped surrogate over the rollout. Returns (policy loss, value loss, entropy)."""
    old_log_probs = torch.tensor(rollout.log_probs, dtype=torch.float32)
    returns = torch.from_numpy(rollout.returns)
    advantages = torch.from_numpy(rollout.advantages)
    # Normalized per rollout: the pIC50 reward's scale moves as the agent climbs, and an
    # unnormalized advantage makes the step size move with it.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_losses, value_losses, entropies = [], [], []
    for _ in range(ppo_cfg.update_epochs):
        order = rng.permutation(len(rollout))
        for start in range(0, len(order), ppo_cfg.minibatch_steps):
            rows = order[start : start + ppo_cfg.minibatch_steps]
            if len(rows) < 2:
                continue

            sets = [rollout.candidates[i] for i in rows]
            set_sizes = np.array([s.num_graphs for s in sets])
            owner = torch.from_numpy(np.repeat(np.arange(len(rows)), set_sizes))
            logits = network.logits(
                featurize.tensors(
                    featurize.concatenate(sets),
                    np.repeat(
                        [rollout.candidate_steps_remaining[i] for i in rows], set_sizes
                    ),
                    cfg,
                )
            )  # [total_candidates]
            log_probs = ppo.segment_log_softmax(logits, owner, len(rows))
            # Where each step's set begins in the concatenated block, so the chosen
            # candidate's row is offset + choice.
            offsets = np.concatenate([[0], np.cumsum(set_sizes)[:-1]])
            taken = torch.from_numpy(
                offsets + np.array([rollout.choices[i] for i in rows])
            )
            chosen_log_probs = log_probs[taken]  # [minibatch]

            ratio = (chosen_log_probs - old_log_probs[rows]).exp()
            minibatch_advantages = advantages[rows]
            unclipped = ratio * minibatch_advantages
            clipped = (
                torch.clamp(ratio, 1 - ppo_cfg.clip_epsilon, 1 + ppo_cfg.clip_epsilon)
                * minibatch_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()

            predicted = network.values(
                featurize.tensors(
                    featurize.concatenate([rollout.states[i] for i in rows]),
                    np.array([rollout.state_steps_remaining[i] for i in rows]),
                    cfg,
                )
            )  # [minibatch]
            value_loss = ((predicted - returns[rows]) ** 2).mean()
            entropy = ppo.segment_entropy(log_probs, owner, len(rows)).mean()

            loss = (
                policy_loss
                + ppo_cfg.value_coefficient * value_loss
                - ppo_cfg.entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), ppo_cfg.grad_clip_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))

    return (
        sum(policy_losses) / len(policy_losses),
        sum(value_losses) / len(value_losses),
        sum(entropies) / len(entropies),
    )


def train(
    cfg: config.Config,
    ppo_cfg: config.PPOConfig,
    reward_fn: Callable[[Chem.Mol], float],
    num_updates: int,
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
    report_every: int = 0,
    pretrained_encoder: Path | None = None,
) -> results.Run:
    determinism.seed_everything(ppo_cfg.seed)
    rng = np.random.default_rng(ppo_cfg.seed)

    network = ppo.MolPPO(cfg)
    if pretrained_encoder is not None:
        network.encoder.load_state_dict(pretrain.load_encoder(pretrained_encoder, cfg))
    optimizer = torch.optim.Adam(network.parameters(), lr=ppo_cfg.learning_rate)

    log_file = open(log_path, "w") if log_path is not None else None
    if log_file is not None:
        # The same columns train_dqn writes, so plot_run can overlay the two runs.
        log_file.write("episode,reward,mean_loss,epsilon,graph_hash\n")

    all_rewards: list[float] = []
    all_molecules: list[Chem.Mol] = []
    started = time.perf_counter()

    for update_index in range(num_updates):
        rollout, episode_rewards, episode_molecules = collect(
            network, cfg, ppo_cfg, reward_fn, rng
        )
        policy_loss, value_loss, entropy = update(
            network, optimizer, rollout, cfg, ppo_cfg, rng
        )

        for reward, mol in zip(episode_rewards, episode_molecules):
            if log_file is not None:
                # epsilon has no meaning here; the column carries entropy instead, which
                # is what PPO explores with.
                log_file.write(
                    f"{len(all_rewards)},{reward:.6f},{value_loss:.6f},"
                    f"{entropy:.4f},{graph_key.canonical_hash(mol)}\n"
                )
            all_rewards.append(reward)
            all_molecules.append(mol)
        if log_file is not None:
            log_file.flush()

        if report_every and (update_index + 1) % report_every == 0:
            recent = all_rewards[-ppo_cfg.rollout_episodes * report_every :]
            elapsed = time.perf_counter() - started
            print(
                f"update {update_index + 1:4d}  episode {len(all_rewards):5d}  "
                f"mean reward {sum(recent) / len(recent):.4f}  "
                f"policy {policy_loss:+.4f}  value {value_loss:.5f}  "
                f"entropy {entropy:.2f}  {len(all_rewards) / elapsed:.2f} episodes/s",
                flush=True,
            )

    if log_file is not None:
        log_file.close()
    if checkpoint_path is not None:
        torch.save(
            {"network": network.state_dict(), "config": cfg, "ppo_config": ppo_cfg},
            checkpoint_path,
        )

    return results.Run(
        episode_rewards=tuple(all_rewards),
        episode_molecules=tuple(all_molecules),
        seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    import argparse

    from mol_optim import report

    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--reward", choices=("qed", "pic50"), default="pic50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=Path, default=None)
    parser.add_argument(
        "--seed-molecule",
        type=int,
        default=None,
        help="index into seeds.choose(); required for pic50",
    )
    parser.add_argument(
        "--max-steps", type=int, default=config.Config.max_steps_per_episode
    )
    parser.add_argument(
        "--regressor", type=Path, default=Path("models/egfr_regressor.pt")
    )
    parser.add_argument("--pretrained-encoder", type=Path, default=None)
    args = parser.parse_args()

    init_mol = None
    if args.reward == "qed":
        reward_fn = rewards.qed
    else:
        reward = reward_pic50.load(args.regressor)
        # Divided by 10 for the same reason train_dqn does it — the two runs have to be
        # on one scale or the comparison is meaningless.
        reward_fn = lambda mol: reward_pic50.score(reward, mol) / 10.0  # noqa: E731
        if args.seed_molecule is not None:
            init_mol = seeds.choose(bindingdb.load())[args.seed_molecule].mol
            print(
                f"starting from seed {args.seed_molecule}: "
                f"{init_mol.GetNumHeavyAtoms()} heavy atoms, "
                f"reward {reward_fn(init_mol):.4f}"
            )

    if init_mol is None:
        # The value head reads a state graph, and there is no graph before the first
        # atom exists. DQN scores candidates only, so it does not have this problem.
        raise SystemExit(
            "PPO needs a starting molecule: pass --seed-molecule, or --reward qed is "
            "not supported here because the empty state has no graph to value."
        )

    run = train(
        config.Config(
            seed=args.seed,
            init_mol=init_mol,
            max_steps_per_episode=args.max_steps,
        ),
        config.PPOConfig(seed=args.seed),
        reward_fn,
        num_updates=args.updates,
        log_path=args.log,
        checkpoint_path=args.checkpoint,
        report_every=args.report_every,
        pretrained_encoder=args.pretrained_encoder,
    )
    best_molecule, best_reward = run.best
    print(f"final_mean_reward {run.final_mean_reward:.4f}  in {run.seconds:.1f}s")
    print(
        f"best: {best_reward:.4f}  "
        f"{best_molecule.GetNumHeavyAtoms()} heavy atoms  "
        f"{graph_key.canonical_hash(best_molecule)}"
    )
    if args.top_k is not None:
        report.top_k(run, args.top_k)
        print(f"wrote {args.top_k}.png and {args.top_k}.sdf")
