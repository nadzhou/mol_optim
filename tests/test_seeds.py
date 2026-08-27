"""The molecules the RL run starts from."""

from rdkit import DataStructs

from mol_optim import seeds, splits


def test_seeds_are_distinct_chemotypes(compounds):
    # Five starting points that are five analogs of one quinazoline would give five runs
    # around the same corner. Distinct scaffolds is the weaker half of that claim, and
    # the fingerprint distance is the half a chemist would recognize.
    chosen = seeds.choose(compounds)
    assert len({seed.scaffold for seed in chosen}) == len(chosen)
    fingerprints = [seeds.MORGAN.GetFingerprint(seed.mol) for seed in chosen]
    for index, first in enumerate(fingerprints):
        for second in fingerprints[index + 1 :]:
            assert DataStructs.TanimotoSimilarity(first, second) <= seeds.MAX_SEED_SIMILARITY


def test_the_choice_does_not_move_between_runs(compounds):
    first = [seed.mol.GetProp("_Name") for seed in seeds.choose(compounds)]
    second = [seed.mol.GetProp("_Name") for seed in seeds.choose(compounds)]
    assert first == second


def test_the_regressor_never_trains_on_a_seed_series(compounds):
    # The whole point of choosing seeds here rather than in the RL run: the split has to
    # know about them. A regressor trained on a seed's series already knows the answer
    # for the molecules the agent starts from, and reports a reward it has not earned.
    chosen = seeds.choose(compounds)
    held_out = seeds.held_out_scaffolds(chosen)
    train, test = splits.scaffold_split(compounds, 0.2, held_out)

    assert held_out == {seed.scaffold for seed in chosen}
    assert not ({compound.scaffold for compound in train} & held_out)
    assert {seed.mol.GetProp("_Name") for seed in chosen} <= {
        compound.mol.GetProp("_Name") for compound in test
    }
