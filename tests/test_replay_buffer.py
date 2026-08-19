"""Our replay buffer, so it carries its own tests."""

import numpy as np
import pytest

from mol_optim import replay_buffer


def make_buffer(capacity: int) -> replay_buffer.ReplayBuffer:
    return replay_buffer.ReplayBuffer(capacity, np.random.default_rng(0))


def push_numbered(buffer: replay_buffer.ReplayBuffer, value: int) -> None:
    buffer.push(
        state=np.full(4, value, dtype=np.uint8),
        state_steps_remaining=value,
        reward=float(value),
        next_candidates=np.full((2, 4), value, dtype=np.uint8),
        next_steps_remaining=value - 1,
        done=False,
    )


def test_fifo_eviction_at_capacity():
    buffer = make_buffer(3)
    for value in [1, 2, 3, 4]:
        push_numbered(buffer, value)
    assert len(buffer) == 3
    assert set(buffer.sample(3).rewards) <= {2.0, 3.0, 4.0}  # 1 evicted, 4 took its slot


def test_sampling_more_than_stored_is_an_error():
    # Loud on purpose: a batch of 128 drawn from 3 transitions trains and learns
    # nothing, and there is no other symptom.
    buffer = make_buffer(10)
    push_numbered(buffer, 1)
    with pytest.raises(ValueError, match="need 4"):
        buffer.sample(4)


def test_sampled_shapes_and_dtypes():
    buffer = make_buffer(10)
    for value in range(5):
        push_numbered(buffer, value)
    batch = buffer.sample(4)
    assert batch.states.shape == (4, 4) and batch.states.dtype == np.uint8
    assert batch.rewards.shape == (4,) and batch.rewards.dtype == np.float32
    assert batch.dones.shape == (4,) and batch.dones.dtype == np.float32
    assert batch.state_steps_remaining.shape == (4,)
    assert batch.next_steps_remaining.shape == (4,)
    assert len(batch.next_candidates) == 4
    assert all(candidates.shape == (2, 4) for candidates in batch.next_candidates)


def test_steps_remaining_stays_paired_with_its_transition():
    # The stored state was scored with steps_remaining, its candidates with one fewer.
    # Swapping the two silently shifts every Q value by one step of discount.
    buffer = make_buffer(10)
    for value in range(1, 6):
        push_numbered(buffer, value)
    batch = buffer.sample(5)
    assert np.array_equal(batch.state_steps_remaining, batch.rewards)
    assert np.array_equal(batch.next_steps_remaining, batch.rewards - 1)


def test_stored_transitions_do_not_alias_the_caller_array():
    # Real bug: the training loop reuses its candidate array, and a buffer that stored
    # a view would silently rewrite its own history.
    buffer = make_buffer(10)
    state = np.ones(4, dtype=np.uint8)
    next_candidates = np.ones((2, 4), dtype=np.uint8)
    buffer.push(
        state=state,
        state_steps_remaining=5,
        reward=1.0,
        next_candidates=next_candidates,
        next_steps_remaining=4,
        done=False,
    )
    state[:] = 99
    next_candidates[:] = 99
    batch = buffer.sample(1)
    assert np.all(batch.states == 1)
    assert np.all(batch.next_candidates[0] == 1)


def test_sampling_is_deterministic_for_a_given_rng_seed():
    first, second = make_buffer(10), make_buffer(10)
    for value in range(8):
        push_numbered(first, value)
        push_numbered(second, value)
    assert np.array_equal(first.sample(4).rewards, second.sample(4).rewards)
