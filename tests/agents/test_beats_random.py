"""The gate: DQN has to beat random on predicted pIC50, and hold the level it reached.

Slow — two 1000-episode runs against a five-model ensemble. `pytest -m "not slow"` skips
them; the nightly run does not. Both need the regressor checkpoint the pipeline's
`regressor` step writes, so they skip on a fresh checkout rather than erroring.

The random baseline is the real test here: a DQN that ties random is broken, and a
mediocre-but-rising reward curve looks like progress with nothing to compare it to.
"""

from dataclasses import replace

import pytest

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments, seeds
from mol_optim.env import rewards
from mol_optim.agents import dqn, random_walk
from mol_optim.report import results
from tests import conftest

LIBRARY = tuple(
    fragments.Fragment(mol=Chem.MolFromSmiles(s), smiles=s, count=1)
    for s in ("*C", "*OC", "*O", "*c1ccccc1", "*N(C)C")
)

CHECKPOINT = config.RegressorSpec().checkpoint or config.AgentSpec().regressor
# 6 edits, not 40: past that the run leaves the regressor's applicability domain.
RUN = config.Config(seed=0, episodes=1000, max_steps_per_episode=6)


@pytest.fixture(scope="module")
def pic50_run():
    conftest.require(CHECKPOINT, conftest.BUILD_IT)
    conftest.require(conftest.BINDINGDB_PATH, conftest.BUILD_IT)
    reward = rewards.load(CHECKPOINT, conftest.BINDINGDB_PATH)
    init_mol = seeds.molecule(conftest.BINDINGDB_PATH, 0)
    return replace(RUN, init_mol=init_mol), lambda mol: rewards.score(reward, mol) / 10.0


@pytest.fixture(scope="module")
def dqn_run(pic50_run) -> results.Run:
    return dqn.train(*pic50_run)


@pytest.fixture(scope="module")
def random_run(pic50_run) -> results.Run:
    return random_walk.rollout(*pic50_run)


@pytest.mark.slow
def test_dqn_beats_random_on_pic50(dqn_run, random_run):
    assert dqn_run.final_mean_reward > random_run.final_mean_reward + 0.1


@pytest.mark.slow
def test_dqn_holds_its_pic50_level(dqn_run):
    # Golden regression. Measured 0.859 at seed 0 against a random floor of 0.331; the
    # threshold sits below that to survive refactors, not to leave room for a regression.
    assert dqn_run.final_mean_reward > 0.80
