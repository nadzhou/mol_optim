from pathlib import Path

import pytest

from mol_optim.chem import molio
from mol_optim.report import audit
from tests.molecules import NAMED

MOTIF_MOLECULES = molio.read_named(
    Path(__file__).parents[1] / "fixtures" / "audit_motifs.sdf"
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


def test_a_drug_carries_none_of_them():
    counts = audit.audit(NAMED["aspirin"]).motif_counts
    assert set(counts.values()) == {0}


def test_a_molecule_contains_its_own_scaffold():
    paracetamol = NAMED["paracetamol"]
    assert audit.audit(paracetamol, audit.scaffold_of(paracetamol)).scaffold_intact


def test_every_motif_is_a_valid_smarts():
    # A typo in a SMARTS gives None, and None matches nothing — a motif that silently
    # never fires is exactly the bug this module exists to prevent.
    for name, pattern in audit.MOTIFS.items():
        assert pattern is not None, name
