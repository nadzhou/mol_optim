"""The EGFR dataset: unit conversion, aggregation, and what came out of the ingest.

plan.md Step 4. The conversion tests are hand-computed on purpose — nM against uM
against M is endemic in binding data, and getting it wrong shifts every label by a
constant, which is invisible in a test-set MAE and wrong in every comparison against a
real number.
"""

import pytest
from rdkit import Chem

from mol_optim import bindingdb, graph_key


@pytest.mark.parametrize(
    "ic50_nm,expected", [(1.0, 9.0), (1000.0, 6.0), (0.1, 10.0), (1_000_000.0, 3.0)]
)
def test_pic50_conversion(ic50_nm, expected):
    assert bindingdb.to_pic50(ic50_nm) == pytest.approx(expected)


def test_pic50_refuses_a_measurement_that_has_no_logarithm():
    # A zero in the IC50 column is a missing value wearing a number's clothes.
    for value in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            bindingdb.to_pic50(value)


def test_pic50_is_monotone_the_right_way_round():
    # Higher pIC50 is the more potent compound. A sign error here inverts the reward and
    # the agent optimizes for inactivity.
    assert bindingdb.to_pic50(1.0) > bindingdb.to_pic50(100.0)


def test_median_takes_the_middle_and_ignores_one_wild_row():
    assert bindingdb.median([3.0, 1.0, 2.0]) == 2.0
    assert bindingdb.median([1.0, 2.0, 3.0, 4.0]) == 2.5
    # Why median and not mean: one lab's outlier moves the label by nothing.
    assert bindingdb.median([7.0, 7.1, 7.2, 200.0]) == pytest.approx(7.15)


def test_the_dataset_is_the_size_and_shape_the_ingest_reported(compounds):
    assert len(compounds) > 10_000
    assert all(1.0 < compound.pic50 < 12.0 for compound in compounds)
    assert all(compound.num_measurements >= 1 for compound in compounds)
    # A spread is only defined where there is more than one measurement.
    assert all(
        compound.pic50_spread == 0.0
        for compound in compounds
        if compound.num_measurements == 1
    )


def test_every_compound_appears_once(compounds):
    names = [compound.mol.GetProp("_Name") for compound in compounds]
    assert len(set(names)) == len(names)


def test_a_sample_of_the_dataset_still_answers_to_its_own_name(compounds):
    # The ingest drops compounds whose key moves in transit, so this holds for every
    # record in the file; the slow test below checks all of them.
    for compound in compounds[:500]:
        assert graph_key.stereo_hash(compound.mol) == compound.mol.GetProp("_Name")


def test_the_labels_have_not_been_silently_shifted(compounds):
    # Known EGFR chemistry, not a statistic: the potent end of this set must reach
    # single-digit nanomolar (pIC50 8-11) and the weak end must be micromolar or worse.
    # A unit error moves the whole distribution by three logs and this notices.
    labels = sorted(compound.pic50 for compound in compounds)
    assert labels[int(0.99 * len(labels))] > 9.0
    assert labels[int(0.01 * len(labels))] < 5.0
    assert 6.0 < labels[len(labels) // 2] < 8.0


def test_salts_and_counter_ions_were_stripped(compounds):
    # "compound + HCl" and "compound" are one measurement of one compound. If the ingest
    # kept the salt, they are two graphs and two rows.
    assert all(len(Chem.GetMolFrags(compound.mol)) == 1 for compound in compounds)


@pytest.mark.slow
def test_every_compound_in_the_file_answers_to_its_own_name(compounds):
    renamed = [
        compound.mol.GetProp("_Name")
        for compound in compounds
        if graph_key.stereo_hash(compound.mol) != compound.mol.GetProp("_Name")
    ]
    assert renamed == []
