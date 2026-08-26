"""The QED gate: DQN has to beat random, and hold the level it reached.

Slow — two 5000-episode runs, about two hours together. `pytest -m "not slow"` skips
them; the nightly run does not.

The random baseline is the real test here: a DQN that ties random is broken, and a
mediocre-but-rising reward curve looks like progress with nothing to compare it to.
"""

import pytest

from mol_optim import baseline_random, config, results, rewards, train_dqn

QED_RUN = config.Config(seed=0)  # the published setup: 5000 episodes


@pytest.fixture(scope="module")
def dqn_run() -> results.Run:
    return train_dqn.train(QED_RUN, rewards.qed)


@pytest.fixture(scope="module")
def random_run() -> results.Run:
    return baseline_random.rollout(QED_RUN, rewards.qed)


@pytest.mark.slow
def test_dqn_beats_random_on_qed(dqn_run, random_run):
    assert dqn_run.final_mean_reward > random_run.final_mean_reward + 0.1


@pytest.mark.slow
def test_dqn_holds_its_qed_level(dqn_run):
    # Golden regression. Measured 0.895 at seed 0 against a random floor of 0.145; the
    # threshold sits below that to survive refactors, not to leave room for a
    # regression. MolDQN publishes ~0.94, which uses double Q with 12 bootstrapped
    # heads (plan.md tier 1); this is the single-head, single-estimator version.
    assert dqn_run.final_mean_reward > 0.85
