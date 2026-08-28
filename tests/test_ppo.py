"""PPO's ragged-action-set primitive, and that a short run trains and repeats.

The candidate set changes size every step, so the policy normalizes within a segment of
a concatenated block rather than over a fixed action head. That segment softmax is the
one piece with no equivalent in the DQN path, so it is checked against torch's dense
log_softmax on each segment taken on its own.
"""

import numpy as np
import pytest
import torch

from mol_optim import config, environment, ppo, rewards, train_ppo

SET_SIZES = [1, 3, 7, 2]  # a one-candidate step is real: a single atom has few edits


def owner_of(sizes) -> torch.Tensor:
    return torch.from_numpy(np.repeat(np.arange(len(sizes)), sizes))


def test_segment_log_softmax_matches_dense_per_segment():
    torch.manual_seed(0)
    logits = torch.randn(sum(SET_SIZES))
    got = ppo.segment_log_softmax(logits, owner_of(SET_SIZES), len(SET_SIZES))

    start = 0
    for size in SET_SIZES:
        expected = torch.log_softmax(logits[start : start + size], dim=0)
        assert torch.allclose(got[start : start + size], expected, atol=1e-6)
        start += size


def test_each_segment_is_a_distribution():
    torch.manual_seed(0)
    # Large logits: without the per-segment max subtraction this overflows to nan.
    logits = torch.randn(sum(SET_SIZES)) * 100
    log_probs = ppo.segment_log_softmax(logits, owner_of(SET_SIZES), len(SET_SIZES))
    assert torch.isfinite(log_probs).all()

    start = 0
    for size in SET_SIZES:
        total = log_probs[start : start + size].exp().sum()
        assert total == pytest.approx(1.0, abs=1e-5)
        start += size


def test_entropy_is_highest_when_every_candidate_is_equal():
    flat = torch.zeros(sum(SET_SIZES))
    owner = owner_of(SET_SIZES)
    log_probs = ppo.segment_log_softmax(flat, owner, len(SET_SIZES))
    entropy = ppo.segment_entropy(log_probs, owner, len(SET_SIZES))
    # A uniform distribution over n candidates has entropy log(n).
    assert entropy.numpy() == pytest.approx(np.log(SET_SIZES), abs=1e-5)

    # One candidate dominating drives it to zero, which is the collapse being watched for.
    peaked = flat.clone()
    peaked[1] = 50.0
    collapsed = ppo.segment_entropy(
        ppo.segment_log_softmax(peaked, owner, len(SET_SIZES)), owner, len(SET_SIZES)
    )
    assert collapsed[1] == pytest.approx(0.0, abs=1e-5)


def tiny(seed: int = 0) -> tuple[config.Config, config.PPOConfig]:
    """A run small enough for the per-commit suite: 2 updates of 2 episodes, 3 edits."""
    start = environment.valid_actions(None, config.Config())[0]
    return (
        config.Config(seed=seed, init_mol=start, max_steps_per_episode=3),
        config.PPOConfig(
            seed=seed, rollout_episodes=2, update_epochs=1, minibatch_steps=4
        ),
    )


def test_a_short_run_produces_one_molecule_per_episode():
    cfg, ppo_cfg = tiny()
    run = train_ppo.train(cfg, ppo_cfg, rewards.qed, num_updates=2)

    assert len(run.episode_rewards) == 2 * ppo_cfg.rollout_episodes
    assert len(run.episode_molecules) == len(run.episode_rewards)
    assert all(0.0 <= reward <= 1.0 for reward in run.episode_rewards)


def test_the_same_seed_gives_the_same_run():
    # PPO samples its actions, so the seeding has to cover torch's generator as well as
    # numpy's. Without that this is the test that fails.
    first = train_ppo.train(*tiny(), rewards.qed, num_updates=2)
    second = train_ppo.train(*tiny(), rewards.qed, num_updates=2)
    assert first.episode_rewards == second.episode_rewards
