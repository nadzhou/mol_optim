"""The scaffold split and the leakage it exists to prevent.

BindingDB duplicates compounds across assays and fills up with series of close analogs
around one frame. Both leak through a random split, and both produce a test score that
is beautiful and describes nothing. These run on the real dataset, on every rebuild of
it, because that is where the leak would appear.
"""

import pytest

from mol_optim import config
from mol_optim.chem import splits


@pytest.fixture(scope="module")
def split(compounds):
    return splits.scaffold_split(compounds, config.RegressorConfig().test_fraction)


def keys(group) -> set[str]:
    return {compound.mol.GetProp("_Name") for compound in group}


def scaffolds(group) -> set[str]:
    return {compound.scaffold for compound in group}


def test_no_duplicate_compounds_across_splits(split):
    train, test = split
    assert not (keys(train) & keys(test))


def test_scaffold_disjoint(split):
    train, test = split
    assert not (scaffolds(train) & scaffolds(test))


def test_seed_scaffolds_are_held_out_of_training(compounds):
    # The agent starts from seed molecules, and the regressor must not already
    # know their series. Held out here by scaffold, not by molecule: an analog of a seed
    # in the training set is the same leak one bond removed.
    seeds = tuple(compounds[i] for i in (0, 7, 30, 100, 500))
    held_out = frozenset(scaffolds(seeds))
    train, test = splits.scaffold_split(compounds, 0.2, held_out_scaffolds=held_out)
    assert not (scaffolds(train) & held_out)
    assert keys(seeds) <= keys(test)


def test_the_split_does_not_move_between_runs(compounds):
    # Two groups of equal size must not swap sides because a dict iterated differently.
    # If they do, every number in the report describes a split nobody can reproduce.
    first = splits.scaffold_split(compounds, 0.2)
    second = splits.scaffold_split(compounds, 0.2)
    assert keys(first[0]) == keys(second[0])