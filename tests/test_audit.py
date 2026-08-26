"""Substructure auditing of a run's molecules.

`tests/fixtures/audit_motifs.sdf` was generated once, the same way molecules.sdf was,
from: hydrazine NN, phenylhydrazine NNc1ccccc1, tetrazane_chain NNNNc1ccccc1,
pyrazole c1cc[nH]n1, hemiaminal OC(N)c1ccccc1, aminal c1ccccc1C(N)N,
gem_diol OC(O)c1ccccc1, n_hydroxylamine ONc1ccccc1.
"""

from pathlib import Path

import pytest

from mol_optim import audit, molio
from tests.molecules import NAMED

MOTIF_MOLECULES = molio.read_named(
    Path(__file__).parent / "fixtures" / "audit_motifs.sdf"
)


@pytest.mark.parametrize(
    "name,motif",
    [
        ("hemiaminal", "hemiaminal"),
        ("aminal", "aminal"),
        ("gem_diol", "gem-diol"),
        ("n_hydroxylamine", "N-hydroxyl"),
        ("phenylhydrazine", "N-N"),
        ("tetrazane_chain", "N-N-N"),
    ],
)
def test_each_motif_fires_on_its_own_molecule(name, motif):
    assert audit.audit(MOTIF_MOLECULES[name]).motif_counts[motif] >= 1


def test_an_aromatic_nitrogen_nitrogen_bond_is_not_a_hydrazine():
    # The distinction the N-N motif exists to make: a pyrazole is ordinary chemistry and
    # appears throughout the vocabulary, a hydrazine is what the pIC50 run built.
    pyrazole = audit.audit(MOTIF_MOLECULES["pyrazole"])
    assert pyrazole.motif_counts["N-N"] == 0
    # ...but the bond is still counted, because that audit came back clean the
    # first time by looking only at the SMARTS.
    assert pyrazole.num_nitrogen_nitrogen_bonds == 1


def test_a_drug_carries_none_of_them():
    counts = audit.audit(NAMED["aspirin"]).motif_counts
    assert set(counts.values()) == {0}


def test_chain_length_is_counted_not_just_presence():
    chain = audit.audit(MOTIF_MOLECULES["tetrazane_chain"])
    assert chain.num_nitrogen_nitrogen_bonds == 3
    assert chain.motif_counts["N-N"] == 3
    assert chain.motif_counts["N-N-N"] == 2


def test_atoms_and_nitrogens_are_counted():
    row = audit.audit(NAMED["caffeine"])
    assert row.num_heavy_atoms == NAMED["caffeine"].GetNumHeavyAtoms()
    assert row.num_nitrogens == 4


def test_a_molecule_contains_its_own_scaffold():
    paracetamol = NAMED["paracetamol"]
    assert audit.audit(paracetamol, audit.scaffold_of(paracetamol)).scaffold_intact


def test_a_different_scaffold_is_not_intact():
    assert not audit.audit(
        NAMED["ethanol"], audit.scaffold_of(NAMED["caffeine"])
    ).scaffold_intact


def test_no_scaffold_given_means_no_claim():
    assert audit.audit(NAMED["benzene"]).scaffold_intact is None


def test_every_motif_is_a_valid_smarts():
    # A typo in a SMARTS gives None, and None matches nothing — a motif that silently
    # never fires is exactly the bug this module exists to prevent.
    for name, pattern in audit.MOTIFS.items():
        assert pattern is not None, name
