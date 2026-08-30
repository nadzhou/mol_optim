"""The measured-pIC50 reward: the positive control's honest oracle.

Nothing is fitted here, so the only ways it can be wrong are the lookup key and the
missing-molecule case — a reward that quietly paid out for unmeasured molecules would
make the control agree with the fitted reward for the wrong reason.
"""

from rdkit import Chem

from mol_optim.chem import graph_key, seeds
from mol_optim.env import measured


def test_unmeasured_molecules_score_zero(measured_table):
    table = measured_table

    assert measured.score(table, Chem.MolFromSmiles("c1ccccc1")) == 0.0
    assert measured.score(table, None) == 0.0
    assert measured.score(table, Chem.MolFromSmiles("")) == 0.0


def test_a_measured_compound_scores_its_own_pic50(compounds, measured_table):
    table = measured_table
    seed = seeds.choose(compounds)[0]

    assert measured.score(table, seed.mol) == seed.pic50


def test_lookup_is_by_graph_not_by_smiles_text(compounds, measured_table):
    """The agent builds an RWMol and has no name or SMILES string for it."""
    table = measured_table
    seed = seeds.choose(compounds)[0]
    # A round trip through text, which the agent never does, must not change the score.
    rebuilt = Chem.MolFromSmiles(Chem.MolToSmiles(seed.mol))

    assert graph_key.canonical_hash(rebuilt) == graph_key.canonical_hash(seed.mol)
    assert measured.score(table, rebuilt) == seed.pic50


def test_every_analog_recovery_counts_is_one_the_reward_pays_for(compounds, measured_table):
    """The control is only a control if the reward and the metric agree on what a hit is."""
    from mol_optim.report import recovery

    table = measured_table
    seed = seeds.choose(compounds)[0]
    analogs = recovery.held_out_analogs(compounds, seed)

    assert all(key in table for key in analogs)
    assert all(table[key] == compound.pic50 for key, compound in analogs.items())
