"""The Step 2 state encoder's input: codes in the buffer, one-hot at the network."""

import numpy as np
import pytest
from rdkit import Chem

from mol_optim import config, featurize, graph_key
from tests.molecules import NAMED, START_MOLECULES

CFG = config.Config()
ASPIRIN = NAMED["aspirin"]


def test_edges_reproduce_the_rdkit_adjacency_matrix():
    # The bond table is where a featurization silently loses the molecule: an off-by-one
    # in the atom offsets gives a well-shaped graph of the wrong compound.
    for mol in START_MOLECULES:
        graphs = featurize.graphs([mol])
        adjacency = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int64)
        adjacency[graphs.edge_index[0], graphs.edge_index[1]] = 1
        assert np.array_equal(adjacency, Chem.GetAdjacencyMatrix(mol))


def test_every_one_hot_block_has_exactly_one_bit_set():
    # An unknown atom type must land in the trailing "other" bucket. An all-zero row
    # reaches the network as "no atom" and trains perfectly well.
    exotic = Chem.MolFromSmiles("[Se]1CCCC1")  # selenium is not in ATOM_TYPES
    Chem.SanitizeMol(exotic)
    batch = featurize.tensors(featurize.graphs([ASPIRIN, exotic]), 3, CFG)
    for features, blocks in [
        (batch.atom_features, featurize.ATOM_BLOCKS),
        (batch.bond_features, featurize.BOND_BLOCKS),
    ]:
        offsets = np.cumsum((0,) + blocks[:-1])
        for offset, width in zip(offsets, blocks):
            block = features[:, offset : offset + width]
            assert np.array_equal(block.sum(dim=1).numpy(), np.ones(len(features)))


def test_feature_widths_are_the_same_for_every_molecule():
    for mol in START_MOLECULES:
        batch = featurize.tensors(featurize.graphs([mol]), 1, CFG)
        assert batch.atom_features.shape == (
            mol.GetNumAtoms(),
            featurize.ATOM_FEATURE_LENGTH,
        )
        assert batch.bond_features.shape == (
            2 * mol.GetNumBonds(),
            featurize.BOND_FEATURE_LENGTH,
        )
        assert batch.graph_features.shape == (1, featurize.NUM_GRAPH_FEATURES)


def test_a_molecule_with_no_bonds_still_featurizes():
    # Methane is the first state of an episode that starts from the empty molecule.
    batch = featurize.tensors(featurize.graphs([NAMED["methane"]]), 40, CFG)
    assert batch.atom_features.shape == (1, featurize.ATOM_FEATURE_LENGTH)
    assert batch.bond_features.shape == (0, featurize.BOND_FEATURE_LENGTH)
    assert batch.edge_index.shape == (2, 0)


def test_the_empty_molecule_is_refused():
    for mol in [None, Chem.RWMol().GetMol()]:
        with pytest.raises(ValueError, match="empty molecule"):
            featurize.graphs([mol])


def test_graph_features_carry_steps_remaining_and_atom_count():
    # Steps remaining is the non-stationarity fix; the atom count is what mean pooling
    # throws away. Both are normalized by max_steps_per_episode.
    mols = (ASPIRIN, NAMED["benzene"])
    batch = featurize.tensors(featurize.graphs(mols), 8, CFG)
    expected = np.array(
        [[8, mol.GetNumAtoms()] for mol in mols], dtype=np.float32
    ) / CFG.max_steps_per_episode
    assert np.allclose(batch.graph_features.numpy(), expected)


def test_steps_remaining_accepts_one_value_per_graph():
    # A replay batch mixes transitions from different points in their episodes.
    batch = featurize.tensors(
        featurize.graphs((ASPIRIN, NAMED["ethanol"])), np.array([3.0, 11.0]), CFG
    )
    assert np.allclose(
        batch.graph_features[:, 0].numpy(),
        np.array([3.0, 11.0]) / CFG.max_steps_per_episode,
    )


def test_concatenated_sets_hold_the_same_graphs_as_the_sets_alone():
    # Candidate sets go into the buffer one at a time and come out concatenated into a
    # batch. If the offsets are wrong here, every target Q value is read off the wrong
    # molecule, and nothing raises.
    sets = [
        featurize.graphs((ASPIRIN, NAMED["methane"])),
        featurize.graphs((NAMED["caffeine"],)),
        featurize.graphs((NAMED["benzene"], NAMED["toluene"], NAMED["ethanol"])),
    ]
    in_order = (
        ASPIRIN,
        NAMED["methane"],
        NAMED["caffeine"],
        NAMED["benzene"],
        NAMED["toluene"],
        NAMED["ethanol"],
    )
    joined = featurize.concatenate(sets)
    solo = featurize.graphs(in_order)

    assert joined.num_graphs == len(in_order)
    assert np.array_equal(
        joined.graph_index,
        np.repeat(np.arange(len(in_order)), [m.GetNumAtoms() for m in in_order]),
    )
    assert np.array_equal(joined.atom_codes, solo.atom_codes)
    assert np.array_equal(joined.edge_index, solo.edge_index)
    assert np.array_equal(joined.bond_codes, solo.bond_codes)


def test_the_same_molecule_encodes_identically_wherever_it_appears():
    solo = featurize.graphs((ASPIRIN,))
    in_a_set = featurize.graphs((NAMED["ethanol"], ASPIRIN, NAMED["benzene"]))
    aspirin_rows = in_a_set.graph_index == 1
    assert np.array_equal(solo.atom_codes, in_a_set.atom_codes[aspirin_rows])


def test_a_kekulized_copy_encodes_the_same_as_the_aromatic_one():
    # Atom and bond features read aromaticity, so a candidate stored in kekule form
    # would reach the network as a different molecule. graph_key.normalize is what
    # prevents that, and this is the assertion that says so.
    kekulized = Chem.RWMol(NAMED["benzene"])
    Chem.Kekulize(kekulized, clearAromaticFlags=True)
    assert np.array_equal(
        featurize.graphs((graph_key.normalize(kekulized.GetMol()),)).atom_codes,
        featurize.graphs((NAMED["benzene"],)).atom_codes,
    )
