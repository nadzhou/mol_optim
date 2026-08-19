"""The Step 1 state encoder. Packed in the buffer, dense only at the network."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from mol_optim import config, featurize
from tests.molecules import NAMED

CFG = config.Config()
ASPIRIN = NAMED["aspirin"]


def test_packed_fingerprint_unpacks_to_the_rdkit_bit_vector():
    reference = rdFingerprintGenerator.GetMorganGenerator(
        radius=CFG.fingerprint_radius, fpSize=CFG.fingerprint_length
    ).GetFingerprintAsNumPy(ASPIRIN)
    unpacked = np.unpackbits(featurize.packed_fingerprint(ASPIRIN, CFG))
    assert np.array_equal(unpacked, reference)
    assert unpacked.sum() > 0  # a fingerprint of all zeros would pass everything else


def test_the_empty_molecule_encodes_as_zeros():
    for mol in [None, Chem.RWMol().GetMol()]:
        packed = featurize.packed_fingerprint(mol, CFG)
        assert packed.shape == (CFG.fingerprint_length // 8,)
        assert not packed.any()


def test_observations_carry_steps_remaining_in_the_last_column():
    packed = featurize.packed_candidates(
        (ASPIRIN, NAMED["ethanol"], NAMED["benzene"]), CFG
    )
    observations = featurize.observations(packed, 7, CFG)
    assert observations.shape == (3, CFG.fingerprint_length + 1)
    assert observations.dtype == np.float32
    assert np.all(observations[:, -1] == 7.0)
    assert set(np.unique(observations[:, :-1])) <= {0.0, 1.0}


def test_observations_accept_one_steps_value_per_row():
    # A replay batch mixes transitions from different points in their episodes.
    packed = featurize.packed_candidates((ASPIRIN, NAMED["ethanol"]), CFG)
    observations = featurize.observations(packed, np.array([3.0, 11.0]), CFG)
    assert np.array_equal(observations[:, -1], np.array([3.0, 11.0], dtype=np.float32))


def test_the_same_molecule_encodes_identically_wherever_it_appears():
    solo = featurize.observations(featurize.packed_fingerprint(ASPIRIN, CFG), 4, CFG)
    in_a_set = featurize.observations(
        featurize.packed_candidates(
            (NAMED["ethanol"], ASPIRIN, NAMED["benzene"]), CFG
        ),
        4,
        CFG,
    )
    assert np.array_equal(solo[0], in_a_set[1])


def test_a_kekulized_copy_encodes_the_same_as_the_aromatic_one():
    # Morgan fingerprints read aromaticity, so a candidate stored in kekule form would
    # reach the network as a different molecule. graph_key.normalize is what prevents
    # that, and this is the assertion that says so.
    kekulized = Chem.RWMol(NAMED["benzene"])
    Chem.Kekulize(kekulized, clearAromaticFlags=True)
    from mol_optim import graph_key

    assert np.array_equal(
        featurize.packed_fingerprint(graph_key.normalize(kekulized.GetMol()), CFG),
        featurize.packed_fingerprint(NAMED["benzene"], CFG),
    )
