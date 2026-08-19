"""The Step 1 and Step 3 gates: DQN has to beat random, and hold the level it reached.

Slow — four 5000-episode runs, about four hours together. `pytest -m "not slow"` skips
them; the nightly run does not.

The random baseline is the real test here: a DQN that ties random is broken, and a
mediocre-but-rising reward curve looks like progress with nothing to compare it to.
"""

import pytest

from mol_optim import baseline_random, config, oracle_gsk3b, results, rewards, train_dqn

QED_RUN = config.Config(seed=0)  # the published setup: 5000 episodes
GSK3B_RUN = config.Config(seed=0)  # same MDP, same knobs; only the reward differs
FOREST = oracle_gsk3b.load()


def gsk3b_reward(mol) -> float:
    return oracle_gsk3b.score(FOREST, mol)


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


@pytest.fixture(scope="module")
def dqn_gsk3b_run() -> results.Run:
    return train_dqn.train(GSK3B_RUN, gsk3b_reward)


@pytest.fixture(scope="module")
def random_gsk3b_run() -> results.Run:
    return baseline_random.rollout(GSK3B_RUN, gsk3b_reward)


@pytest.mark.slow
def test_dqn_beats_random_on_the_gsk3b_oracle(dqn_gsk3b_run, random_gsk3b_run):
    # Measured at seed 0: 0.610 against a random floor of 0.077. The margin asserted
    # here is half of that gap, so a run that merely drifts up still fails.
    assert (
        dqn_gsk3b_run.final_mean_reward > random_gsk3b_run.final_mean_reward + 0.25
    )


@pytest.mark.slow
def test_dqn_holds_its_gsk3b_level(dqn_gsk3b_run):
    # Golden regression, measured 0.610 at seed 0. The threshold sits below that to
    # survive refactors, not to leave room for a regression. For scale: 3000 ZINC
    # molecules score 0.029 on average and the best of them 0.51.
    assert dqn_gsk3b_run.final_mean_reward > 0.55
