"""The molecules the RL run starts from. plan.md, "Target selection"."""

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


def test_seeds_are_potent_and_their_labels_are_settled(compounds):
    # The final report quotes each seed's own pIC50 as the number to beat. A label with
    # a two-log spread across its measurements is not a number to beat.
    for seed in seeds.choose(compounds):
        assert seed.pic50 >= 8.0
        assert seed.num_measurements > 1
        assert seed.pic50_spread <= 1.0


def test_seeds_come_from_series_a_chemist_has_walked_around(compounds):
    # Largest clusters first: a scaffold with one measured active is a hit, not a lead
    # series, and derivatizing it has nothing to compare against.
    actives = [compound for compound in compounds if compound.pic50 >= 8.0]
    clusters = splits.by_scaffold(actives)
    for seed in seeds.choose(compounds):
        assert len(clusters[seed.scaffold]) >= 20


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
