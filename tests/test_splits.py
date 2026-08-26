"""The scaffold split and the leakage it exists to prevent.

BindingDB duplicates compounds across assays and fills up with series of close analogs
around one frame. Both leak through a random split, and both produce a test score that
is beautiful and describes nothing. These run on the real dataset, on every rebuild of
it, because that is where the leak would appear.
"""

import numpy as np
import pytest

from mol_optim import config, graph_key, splits


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
    # Step 5 starts the agent from seed molecules, and the regressor must not already
    # know their series. Held out here by scaffold, not by molecule: an analog of a seed
    # in the training set is the same leak one bond removed.
    seeds = tuple(compounds[i] for i in (0, 7, 30, 100, 500))
    held_out = frozenset(scaffolds(seeds))
    train, test = splits.scaffold_split(compounds, 0.2, held_out_scaffolds=held_out)
    assert not (scaffolds(train) & held_out)
    assert keys(seeds) <= keys(test)


def test_the_split_is_the_size_it_was_asked_for(split, compounds):
    train, test = split
    assert len(train) + len(test) == len(compounds)
    assert 0.75 < len(train) / len(compounds) < 0.85


def test_the_split_does_not_move_between_runs(compounds):
    # Two groups of equal size must not swap sides because a dict iterated differently.
    # If they do, every number in the report describes a split nobody can reproduce.
    first = splits.scaffold_split(compounds, 0.2)
    second = splits.scaffold_split(compounds, 0.2)
    assert keys(first[0]) == keys(second[0])


def test_both_sides_carry_the_range_of_potencies(split):
    # A split that put every potent compound on one side would report a fine MAE and be
    # useless. Not guaranteed by construction, so it is checked.
    train, test = split
    train_labels = np.array([compound.pic50 for compound in train])
    test_labels = np.array([compound.pic50 for compound in test])
    assert abs(train_labels.mean() - test_labels.mean()) < 0.5
    assert test_labels.max() > 9.0
    assert test_labels.min() < 5.0


def test_a_scaffold_group_holds_molecules_that_share_a_frame(compounds):
    groups = splits.by_scaffold(compounds)
    largest = next(iter(groups.values()))
    assert len(largest) > 100
    assert len({compound.scaffold for compound in largest}) == 1
    # And the carried key is the one graph_key computes, not a stale copy of it.
    assert largest[0].scaffold == graph_key.scaffold_hash(largest[0].mol)
