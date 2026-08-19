"""DQN on the molecule MDP — the whole training step, flat and in order.

Reads top to bottom: enumerate candidates, score them, act, store, update. The
reference splits the update across a helper that loops over the batch in Python; here
it is one batched forward pass, inline, where the shapes are visible.
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from rdkit import Chem

from mol_optim import (
    config,
    determinism,
    dqn,
    environment,
    featurize,
    graph_key,
    oracle_gsk3b,
    pretrain,
    replay_buffer,
    report,
    results,
    rewards,
)


def epsilon_at_episode(episode_index: int, cfg: config.Config) -> float:
    """Piecewise linear: epsilon_start -> epsilon_mid at half the run -> epsilon_end.

    The published schedule (run_dqn.py PiecewiseSchedule). The PyTorch port instead
    multiplies epsilon by 0.99907 per episode, a schedule whose endpoint depends on how
    many episodes you happen to run.
    """
    halfway = cfg.episodes / 2
    if episode_index < halfway:
        return cfg.epsilon_start + (cfg.epsilon_mid - cfg.epsilon_start) * (
            episode_index / halfway
        )
    return cfg.epsilon_mid + (cfg.epsilon_end - cfg.epsilon_mid) * min(
        (episode_index - halfway) / halfway, 1.0
    )


def train(
    cfg: config.Config,
    reward_fn: Callable[[Chem.Mol], float],
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
    report_every: int = 0,
    pretrained_encoder: Path | None = None,
) -> results.Run:
    determinism.seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cpu")

    online_dqn = dqn.MolDQN(cfg).to(device)
    if pretrained_encoder is not None:
        # Only the encoder. The Q head scores a reward that did not exist during
        # pretraining, so it starts from its own initialization. load_encoder refuses a
        # checkpoint built on another featurization or another encoder shape — the
        # silent no-op this whole step is designed around (plan.md Step 3b).
        online_dqn.encoder.load_state_dict(pretrain.load_encoder(pretrained_encoder, cfg))
    target_dqn = dqn.MolDQN(cfg).to(device)
    target_dqn.load_state_dict(online_dqn.state_dict())
    for parameter in target_dqn.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(online_dqn.parameters(), lr=cfg.learning_rate)
    buffer = replay_buffer.ReplayBuffer(cfg.replay_buffer_size, rng)

    log_file = open(log_path, "w") if log_path is not None else None
    if log_file is not None:
        log_file.write("episode,reward,mean_loss,epsilon,graph_hash\n")

    episode_rewards: list[float] = []
    episode_molecules: list = []
    total_steps = 0
    started = time.perf_counter()

    for episode_index in range(cfg.episodes):
        epsilon = epsilon_at_episode(episode_index, cfg)
        episode = environment.reset(cfg)
        episode_losses: list[float] = []
        # Carried across the loop: this step's next-state candidates are the next
        # step's candidates, and featurizing them twice is a large share of a step.
        candidates = featurize.graphs(episode.valid_actions)  # num_candidates graphs

        while True:
            steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken

            if rng.random() < epsilon:
                choice = int(rng.integers(len(episode.valid_actions)))
            else:
                with torch.no_grad():
                    q_candidates = online_dqn(
                        featurize.tensors(candidates, steps_remaining, cfg)
                    )  # [num_candidates, 1]
                choice = int(torch.argmax(q_candidates))

            result = environment.step(episode, choice, reward_fn, cfg)
            next_steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken
            next_candidates = featurize.graphs(episode.valid_actions)
            buffer.push(
                state=featurize.graphs((result.state,)),
                state_steps_remaining=steps_remaining,
                reward=result.reward,
                next_candidates=next_candidates,
                next_steps_remaining=next_steps_remaining,
                done=result.terminated,
            )
            candidates = next_candidates
            total_steps += 1

            if total_steps % cfg.update_interval == 0 and len(buffer) >= cfg.batch_size:
                for _ in range(cfg.updates_per_interval):
                    batch = buffer.sample(cfg.batch_size)

                    q_taken = online_dqn(
                        featurize.tensors(
                            featurize.concatenate(batch.states),
                            batch.state_steps_remaining,
                            cfg,
                        )
                    ).squeeze(-1)  # [batch]

                    # The target is a max over each next state's *candidate set*, and
                    # those sets have different sizes. Stack them all into one forward
                    # pass, then segment-max back down to [batch].
                    set_sizes = np.array(
                        [
                            candidate_set.num_graphs
                            for candidate_set in batch.next_candidates
                        ]
                    )  # [batch]
                    owner = torch.from_numpy(
                        np.repeat(np.arange(cfg.batch_size), set_sizes)
                    ).to(device)  # [total_candidates]

                    with torch.no_grad():
                        q_next = target_dqn(
                            featurize.tensors(
                                featurize.concatenate(batch.next_candidates),
                                np.repeat(batch.next_steps_remaining, set_sizes),
                                cfg,
                            )
                        ).squeeze(-1)  # [total_candidates]
                        best_next = torch.zeros(
                            cfg.batch_size, device=device
                        ).scatter_reduce(
                            0, owner, q_next, reduce="amax", include_self=False
                        )  # [batch]
                        not_done = 1.0 - torch.from_numpy(batch.dones).to(device)
                        q_target = (
                            torch.from_numpy(batch.rewards).to(device)
                            + cfg.gamma * not_done * best_next
                        )  # [batch]

                    # Mean squared error. The reference uses Huber, which flattens
                    # the gradient once the TD error passes 1.0; with rewards in [0, 1]
                    # that clip almost never engages anyway, and gradient clipping below
                    # already bounds the step size.
                    loss = ((q_taken - q_target) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        online_dqn.parameters(), cfg.grad_clip_norm
                    )
                    optimizer.step()
                    episode_losses.append(float(loss.detach()))

                    with torch.no_grad():
                        for parameter, target_parameter in zip(
                            online_dqn.parameters(), target_dqn.parameters()
                        ):
                            target_parameter.mul_(cfg.polyak)
                            target_parameter.add_((1.0 - cfg.polyak) * parameter)

            if result.terminated:
                break

        episode_rewards.append(result.reward)
        episode_molecules.append(result.state)
        mean_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        if log_file is not None:
            log_file.write(
                f"{episode_index},{result.reward:.6f},{mean_loss:.6f},"
                f"{epsilon:.4f},{graph_key.canonical_hash(result.state)}\n"
            )
            log_file.flush()
        if report_every and (episode_index + 1) % report_every == 0:
            recent = episode_rewards[-report_every:]
            elapsed = time.perf_counter() - started
            print(
                f"episode {episode_index + 1:5d}  "
                f"mean reward {sum(recent) / len(recent):.4f}  "
                f"loss {mean_loss:.5f}  eps {epsilon:.3f}  "
                f"{total_steps / elapsed:.1f} steps/s",
                flush=True,
            )

    if log_file is not None:
        log_file.close()
    if checkpoint_path is not None:
        torch.save(
            {
                "config": cfg,
                "online_dqn": online_dqn.state_dict(),
                "target_dqn": target_dqn.state_dict(),
            },
            checkpoint_path,
        )

    return results.Run(
        episode_rewards=tuple(episode_rewards),
        episode_molecules=tuple(episode_molecules),
        seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument(
        "--reward",
        choices=("qed", "gsk3b"),
        default="qed",
        help="qed is Step 1-2; gsk3b is the TDC oracle, Step 3",
    )
    parser.add_argument(
        "--top-k", type=Path, default=None, help="stem for a top-k drawing and SDF"
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        help="a ZINC AttrMask checkpoint from mol_optim.pretrain, Step 3b",
    )
    args = parser.parse_args()

    if args.reward == "qed":
        reward_fn = rewards.qed
    else:
        forest = oracle_gsk3b.load()
        reward_fn = lambda mol: oracle_gsk3b.score(forest, mol)  # noqa: E731

    run = train(
        config.Config(episodes=args.episodes, seed=args.seed),
        reward_fn,
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
