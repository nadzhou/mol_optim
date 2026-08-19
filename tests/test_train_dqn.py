"""The DQN update: exploration schedule, target network, and the batched max."""

import numpy as np
import pytest
import torch

from mol_optim import config, dqn, environment, featurize, rewards, train_dqn
from tests.molecules import NAMED

SMALL = config.Config(
    init_mol=NAMED["ethanol"],
    episodes=6,
    max_steps_per_episode=4,
    batch_size=4,
    update_interval=2,
    replay_buffer_size=100,
)


def test_epsilon_falls_from_start_through_mid_to_end_and_never_rises():
    cfg = config.Config(
        episodes=200, epsilon_start=1.0, epsilon_mid=0.1, epsilon_end=0.01
    )
    values = [train_dqn.epsilon_at_episode(i, cfg) for i in range(cfg.episodes)]
    assert values[0] == cfg.epsilon_start
    assert values[100] == pytest.approx(cfg.epsilon_mid)
    assert values[199] == pytest.approx(cfg.epsilon_end, abs=1e-3)
    assert train_dqn.epsilon_at_episode(500, cfg) == pytest.approx(cfg.epsilon_end)
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def load(path):
    return torch.load(path, weights_only=False)


def state_dicts_equal(first, second) -> bool:
    return all(torch.equal(first[key], second[key]) for key in first)


def test_target_network_lags_and_tracks_the_online_network(tmp_path):
    # Three short runs off the same seed, so all three start from identical weights.
    never_updates = tmp_path / "never.pt"
    frozen_target = tmp_path / "frozen.pt"
    copied_target = tmp_path / "copied.pt"

    # batch_size larger than the buffer will ever hold: the loop never updates, so
    # this checkpoint is the initialization.
    train_dqn.train(
        config.Config(**{**SMALL.__dict__, "batch_size": 10**6}),
        rewards.qed,
        checkpoint_path=never_updates,
    )
    train_dqn.train(
        config.Config(**{**SMALL.__dict__, "polyak": 1.0}),
        rewards.qed,
        checkpoint_path=frozen_target,
    )
    train_dqn.train(
        config.Config(**{**SMALL.__dict__, "polyak": 0.0}),
        rewards.qed,
        checkpoint_path=copied_target,
    )

    initial, frozen, copied = (
        load(never_updates),
        load(frozen_target),
        load(copied_target),
    )
    assert state_dicts_equal(initial["online_dqn"], initial["target_dqn"])
    # polyak=1.0 keeps the target exactly at initialization while the online net moves,
    # so gradients reach the online net only.
    assert state_dicts_equal(frozen["target_dqn"], initial["target_dqn"])
    assert not state_dicts_equal(frozen["online_dqn"], initial["online_dqn"])
    # polyak=0.0 copies the online net wholesale on every update.
    assert state_dicts_equal(copied["target_dqn"], copied["online_dqn"])


def test_segment_max_over_ragged_candidate_sets_matches_the_obvious_loop():
    # The target is max over each next state's candidate set, and those sets differ in
    # size. The training step stacks them into one forward pass and takes a segment
    # max; this pins that it agrees with scoring each set on its own.
    cfg = config.Config()
    torch.manual_seed(0)
    network = dqn.MolDQN(cfg)
    candidate_sets = [
        environment.valid_actions(NAMED[name], cfg)
        for name in ("methane", "benzene", "aspirin", "caffeine")
    ]  # ragged: 4 to a few hundred candidates
    steps_remaining = np.array([1.0, 7.0, 19.0, 40.0])

    set_sizes = np.array([len(candidates) for candidates in candidate_sets])
    owner = torch.from_numpy(np.repeat(np.arange(len(candidate_sets)), set_sizes))
    with torch.no_grad():
        stacked = network(
            featurize.tensors(
                featurize.concatenate([featurize.graphs(c) for c in candidate_sets]),
                np.repeat(steps_remaining, set_sizes),
                cfg,
            )
        ).squeeze(-1)  # [total_candidates]
        batched = torch.zeros(len(candidate_sets)).scatter_reduce(
            0, owner, stacked, reduce="amax", include_self=False
        )  # [batch]
        one_at_a_time = torch.tensor(
            [
                float(
                    network(
                        featurize.tensors(featurize.graphs(candidates), steps, cfg)
                    ).max()
                )
                for candidates, steps in zip(candidate_sets, steps_remaining)
            ]
        )
    assert torch.allclose(batched, one_at_a_time, atol=1e-5)


def test_negative_q_values_survive_the_segment_max():
    # scatter_reduce with include_self=True would fold the zero initialization into
    # the max and silently clamp every negative Q value to 0.
    owner = torch.tensor([0, 0, 1, 1])
    values = torch.tensor([-3.0, -1.0, -7.0, -5.0])
    best = torch.zeros(2).scatter_reduce(
        0, owner, values, reduce="amax", include_self=False
    )
    assert torch.equal(best, torch.tensor([-1.0, -5.0]))
