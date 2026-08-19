"""Step 3's reward. Every failure mode here returns a plausible number, not an error.

A wrong fingerprint radius, a walk that stops one level short of a leaf, a leaf whose
children were not rewritten to point at itself — each of those still hands the training
loop a float in [0, 1], and the run just learns something else. So the gate is exact
agreement with PyTDC's own oracle on molecules spanning its whole output range.
"""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from mol_optim import config, environment, graph_key, molio, oracle_gsk3b, rewards
from tests.molecules import NAMED

FOREST = oracle_gsk3b.load()
# Written by PyTDC 1.0.0 (scikit-learn 1.2.2) in a throwaway venv: 14 ZINC molecules
# picked to spread across the oracle's range, 24 molecules from random rollouts of this
# environment, and the drug fixtures. Each record carries TDC's score in `gsk3b`.
REFERENCE = molio.read_named(
    Path(__file__).parent / "fixtures" / "gsk3b_reference.sdf"
)


def test_matches_tdc_on_the_reference_set():
    for name, mol in REFERENCE.items():
        expected = float(mol.GetProp("gsk3b"))
        assert oracle_gsk3b.score(FOREST, mol) == pytest.approx(expected, abs=1e-6), name


def test_the_reference_set_spans_the_oracles_range():
    # Guards the test above from passing on a forest that returns 0.0 for everything.
    scores = [float(mol.GetProp("gsk3b")) for mol in REFERENCE.values()]
    assert min(scores) == 0.0
    assert max(scores) > 0.4
    assert sum(score > 0 for score in scores) > 25


def test_score_is_zero_without_a_molecule():
    assert oracle_gsk3b.score(FOREST, None) == 0.0
    assert oracle_gsk3b.score(FOREST, Chem.RWMol().GetMol()) == 0.0


def test_score_does_not_depend_on_atom_numbering():
    mol = NAMED["caffeine"]
    shuffled = list(range(mol.GetNumAtoms()))
    np.random.default_rng(0).shuffle(shuffled)
    renumbered = Chem.RenumberAtoms(mol, shuffled)
    assert oracle_gsk3b.score(FOREST, renumbered) == oracle_gsk3b.score(FOREST, mol)


def test_every_walk_reaches_a_leaf():
    # `depth` is how many rounds score() walks. One short and it reads an internal
    # node's class ratio, which is a number in [0, 1] and is not the model's answer.
    rng = np.random.default_rng(0)
    for _ in range(20):
        bits = rng.random(2048) < 0.02  # a fingerprint's density, near enough
        node = FOREST.roots
        for _ in range(FOREST.depth):
            node = np.where(bits[FOREST.bit[node]], FOREST.right[node], FOREST.left[node])
        assert np.array_equal(FOREST.left[node], node)  # a leaf points at itself


def test_a_molecule_the_environment_built_survives_the_trip_to_disk(tmp_path):
    # The reference fixture reaches the test through an SDF, but training scores the
    # RWMol the environment just edited. If perception differed between the two, the
    # fixture would pin the wrong numbers and nothing else would say so.
    cfg = config.Config(init_mol=NAMED["aspirin"], max_steps_per_episode=6)
    episode = environment.reset(cfg)
    built = []
    for step_index in range(6):
        built.append(environment.step(episode, step_index, rewards.qed, cfg).state)

    path = tmp_path / "built.sdf"
    molio.write(path, tuple(built), {"i": list(range(len(built)))})
    from_disk = [mol for mol in Chem.SDMolSupplier(str(path))]

    for live, loaded in zip(built, from_disk):
        assert graph_key.canonical_hash(live) == graph_key.canonical_hash(loaded)
        assert oracle_gsk3b.score(FOREST, live) == oracle_gsk3b.score(FOREST, loaded)


def test_a_missing_model_file_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="mol_optim.fetch_gsk3b"):
        oracle_gsk3b.load(tmp_path / "absent.npz")
