"""Our replay buffer, so it carries its own tests."""

import numpy as np

from mol_optim.chem import featurize
from mol_optim.env import replay_buffer
from tests.molecules import NAMED

ONE_GRAPH = featurize.graphs((NAMED["ethanol"],))
TWO_GRAPHS = featurize.graphs((NAMED["aspirin"], NAMED["methane"]))


def make_buffer(capacity: int) -> replay_buffer.ReplayBuffer:
    return replay_buffer.ReplayBuffer(capacity, np.random.default_rng(0))


def push_numbered(buffer: replay_buffer.ReplayBuffer, value: int) -> None:
    buffer.push(
        state=ONE_GRAPH,
        state_steps_remaining=value,
        reward=float(value),
        next_candidates=TWO_GRAPHS,
        next_steps_remaining=value - 1,
        done=False,
    )


def test_fifo_eviction_at_capacity():
    buffer = make_buffer(3)
    for value in [1, 2, 3, 4]:
        push_numbered(buffer, value)
    assert len(buffer) == 3
    assert set(buffer.sample(3).rewards) <= {2.0, 3.0, 4.0}  # 1 evicted, 4 took its slot


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
    # RDKit molecules are mutable and so are the code arrays taken off them. A buffer
    # holding a view would silently rewrite its own history.
    buffer = make_buffer(10)
    state = featurize.graphs((NAMED["ethanol"],))
    next_candidates = featurize.graphs((NAMED["aspirin"], NAMED["methane"]))
    buffer.push(
        state=state,
        state_steps_remaining=5,
        reward=1.0,
        next_candidates=next_candidates,
        next_steps_remaining=4,
        done=False,
    )
    state.atom_codes[:] = 99
    next_candidates.atom_codes[:] = 99
    batch = buffer.sample(1)
    assert np.array_equal(batch.states[0].atom_codes, ONE_GRAPH.atom_codes)
    assert np.array_equal(batch.next_candidates[0].atom_codes, TWO_GRAPHS.atom_codes)


def test_sampling_is_deterministic_for_a_given_rng_seed():
    first, second = make_buffer(10), make_buffer(10)
    for value in range(8):
        push_numbered(first, value)
        push_numbered(second, value)
    assert np.array_equal(first.sample(4).rewards, second.sample(4).rewards)
