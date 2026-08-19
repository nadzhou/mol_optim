"""State encoding for Step 1: Morgan fingerprint + steps remaining, taken off the graph.

Fingerprints stay *packed* (numpy uint8, one bit per bit) everywhere except the moment
they enter the network. The replay buffer holds a candidate set per transition — tens of
thousands of sets — and at float32 per bit a 5000-episode run needs ~20 GB of RAM.
Packed it needs a few hundred MB.

This whole module is what Step 2 replaces with a GNN encoder, which reads the same
graphs directly instead of hashing them into bits.
"""

import functools

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from mol_optim import config


@functools.lru_cache(maxsize=8)
def _morgan_generator(radius: int, length: int):
    """Cached because building the generator costs more than using it."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=length)


def packed_fingerprint(mol: Chem.Mol | None, cfg: config.Config) -> np.ndarray:
    """[fingerprint_length // 8] uint8. All-zeros for the empty molecule."""
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros(cfg.fingerprint_length // 8, dtype=np.uint8)
    generator = _morgan_generator(cfg.fingerprint_radius, cfg.fingerprint_length)
    return np.packbits(generator.GetFingerprintAsNumPy(mol))


def packed_candidates(mols: tuple[Chem.Mol, ...], cfg: config.Config) -> np.ndarray:
    """[num_candidates, fingerprint_length // 8] uint8."""
    return np.stack([packed_fingerprint(mol, cfg) for mol in mols])


def observations(
    packed: np.ndarray, steps_remaining: float | np.ndarray, cfg: config.Config
) -> np.ndarray:
    """Network input: [num_candidates, fingerprint_length + 1] float32.

    The trailing column is steps remaining — one value for a whole candidate set, or one
    per row. Without it the discount makes the MDP non-stationary and the same molecule
    carries two different Q values. Raw, not normalized, to match the MolDQN baseline
    being reproduced.
    """
    bits = np.unpackbits(np.atleast_2d(packed), axis=1).astype(
        np.float32
    )  # [num_candidates, fingerprint_length]
    steps_column = np.broadcast_to(
        np.asarray(steps_remaining, dtype=np.float32).reshape(-1, 1), (len(bits), 1)
    )
    return np.concatenate([bits, steps_column], axis=1)
