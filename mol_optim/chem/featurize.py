"""Graph featurization: an RDKit molecule to the tensors the encoder reads.

Molecules are stored as int8 codes and expanded to one-hot float32 only where a batch
enters the network — 13 bytes an atom against 48, which is what makes a replay buffer of
tens of thousands of candidate sets fit. Every categorical ends in an "other" bucket, so
an atom the tables do not name lands in a real column rather than an all-zero row.
"""

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem

from mol_optim import config

# Featurization alphabet, not the action alphabet (cfg.atom_types sizes that).
ATOM_TYPES = ("C", "N", "O", "F", "S", "Cl", "Br")
HYBRIDIZATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
CHIRAL_TAGS = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)
RING_SIZES = (3, 4, 5, 6, 7)

# Width of each atom field's one-hot block, in the order _atom_code emits them.
ATOM_BLOCKS = (
    (
        len(ATOM_TYPES) + 1,  # element
        6,  # heavy-atom degree, 0-5
        5,  # formal charge, -2..+2
        len(HYBRIDIZATIONS) + 1,
        2,  # aromatic
        5,  # total hydrogens, 0-4
        2,  # in a ring
    )
    + (2,) * len(RING_SIZES)
    + (len(CHIRAL_TAGS) + 1,)
)
BOND_BLOCKS = (
    len(BOND_TYPES) + 1,
    2,  # conjugated
    2,  # in a ring
    len(BOND_STEREO) + 1,
)
ATOM_FEATURE_LENGTH = sum(ATOM_BLOCKS)
BOND_FEATURE_LENGTH = sum(BOND_BLOCKS)
# Steps remaining and heavy-atom count, concatenated to the pooled embedding.
NUM_GRAPH_FEATURES = 2


def signature() -> str:
    """A short hash of the featurization alphabet, recorded in every checkpoint.

    An encoder pretrained on one featurization and then loaded against another reads
    every input column as something it was not — a silent, total waste of the
    pretraining. Widths alone do not catch it: reordering ATOM_TYPES
    changes what every element column means and changes no dimension. So this hashes
    the tables themselves, and pretrain.load_encoder refuses a checkpoint whose hash
    is not this one.
    """
    tables = (
        ATOM_TYPES,
        HYBRIDIZATIONS,
        CHIRAL_TAGS,
        BOND_TYPES,
        BOND_STEREO,
        RING_SIZES,
        ATOM_BLOCKS,
        BOND_BLOCKS,
        NUM_GRAPH_FEATURES,
    )
    return hashlib.sha256(repr(tables).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Graphs:
    """One or more molecular graphs, concatenated into flat code arrays.

    The unit of storage and of batching both: a candidate set is a Graphs, and a
    replay batch is those Graphs concatenated again (see `concatenate`). Bonds appear
    in both directions, so a message passes each way independently.
    """

    atom_codes: np.ndarray  # [total_atoms, len(ATOM_BLOCKS)] int8
    bond_codes: np.ndarray  # [total_edges, len(BOND_BLOCKS)] int8
    edge_index: np.ndarray  # [2, total_edges] int32 — source row, target row
    graph_index: np.ndarray  # [total_atoms] int32 — which graph each atom belongs to
    num_graphs: int


@dataclass(frozen=True)
class Batch:
    """What the network reads: one-hot features and the index arrays that group them."""

    atom_features: torch.Tensor  # [total_atoms, ATOM_FEATURE_LENGTH] float32
    bond_features: torch.Tensor  # [total_edges, BOND_FEATURE_LENGTH] float32
    edge_index: torch.Tensor  # [2, total_edges] int64
    graph_index: torch.Tensor  # [total_atoms] int64
    graph_features: torch.Tensor  # [num_graphs, NUM_GRAPH_FEATURES] float32
    num_graphs: int


def graphs(mols: Sequence[Chem.Mol]) -> Graphs:
    atom_codes: list[tuple] = []
    bond_codes: list[tuple] = []
    edges: list[tuple[int, int]] = []
    graph_index: list[int] = []
    atom_offset = 0

    for index, mol in enumerate(mols):
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Cannot featurize an empty molecule")
        ring_info = mol.GetRingInfo()
        for atom in mol.GetAtoms():
            atom_codes.append(_atom_code(atom, ring_info))
            graph_index.append(index)
        for bond in mol.GetBonds():
            code = _bond_code(bond)
            begin = bond.GetBeginAtomIdx() + atom_offset
            end = bond.GetEndAtomIdx() + atom_offset
            edges.append((begin, end))
            edges.append((end, begin))
            bond_codes.append(code)
            bond_codes.append(code)
        atom_offset += mol.GetNumAtoms()

    return Graphs(
        atom_codes=np.array(atom_codes, dtype=np.int8).reshape(-1, len(ATOM_BLOCKS)),
        bond_codes=np.array(bond_codes, dtype=np.int8).reshape(-1, len(BOND_BLOCKS)),
        edge_index=np.array(edges, dtype=np.int32).reshape(-1, 2).T,
        graph_index=np.array(graph_index, dtype=np.int32),
        num_graphs=len(mols),
    )


def _atom_code(atom: Chem.Atom, ring_info) -> tuple[int, ...]:
    # One small integer per field, in ATOM_BLOCKS order.
    return (
        _index_of(atom.GetSymbol(), ATOM_TYPES),
        min(atom.GetDegree(), 5),
        min(max(atom.GetFormalCharge(), -2), 2) + 2,
        _index_of(atom.GetHybridization(), HYBRIDIZATIONS),
        int(atom.GetIsAromatic()),
        min(atom.GetTotalNumHs(), 4),
        int(atom.IsInRing()),
        *(int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), size)) for size in RING_SIZES),
        _index_of(atom.GetChiralTag(), CHIRAL_TAGS),
    )


def _bond_code(bond: Chem.Bond) -> tuple[int, ...]:
    # One small integer per field, in BOND_BLOCKS order.
    return (
        _index_of(bond.GetBondType(), BOND_TYPES),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _index_of(bond.GetStereo(), BOND_STEREO),
    )


def _index_of(value, known: tuple) -> int:
    return known.index(value) if value in known else len(known)


def concatenate(sets: Sequence[Graphs]) -> Graphs:
    """Join candidate sets into one block, shifting atom rows and graph numbers."""
    atom_offsets = np.cumsum([0] + [len(s.atom_codes) for s in sets[:-1]])
    graph_offsets = np.cumsum([0] + [s.num_graphs for s in sets[:-1]])
    return Graphs(
        atom_codes=np.concatenate([s.atom_codes for s in sets]),
        bond_codes=np.concatenate([s.bond_codes for s in sets]),
        edge_index=np.concatenate(
            [s.edge_index + offset for s, offset in zip(sets, atom_offsets)], axis=1
        ).astype(np.int32),
        graph_index=np.concatenate(
            [s.graph_index + offset for s, offset in zip(sets, graph_offsets)]
        ).astype(np.int32),
        num_graphs=sum(s.num_graphs for s in sets),
    )


def tensors(
    graph_set: Graphs, steps_remaining: float | np.ndarray, cfg: config.Config
) -> Batch:
    """Expand codes to one-hot and attach the graph-level features.

    `steps_remaining` is one value for the whole block or one per graph. It is required:
    the environment discounts by steps remaining, so without it the same molecule
    carries two different Q values and the MDP is non-stationary. Both graph features
    are divided by max_steps_per_episode — an episode adds at most one atom per step,
    so that keeps the heavy-atom count near 1.0 as well.
    """
    atom_counts = np.bincount(
        graph_set.graph_index, minlength=graph_set.num_graphs
    ).astype(np.float32)  # [num_graphs]
    steps_column = np.broadcast_to(
        np.asarray(steps_remaining, dtype=np.float32).reshape(-1),
        (graph_set.num_graphs,),
    )
    graph_features = (
        np.stack([steps_column, atom_counts], axis=1) / cfg.max_steps_per_episode
    )  # [num_graphs, NUM_GRAPH_FEATURES]

    return Batch(
        atom_features=torch.from_numpy(_one_hot(graph_set.atom_codes, ATOM_BLOCKS)),
        bond_features=torch.from_numpy(_one_hot(graph_set.bond_codes, BOND_BLOCKS)),
        edge_index=torch.from_numpy(graph_set.edge_index.astype(np.int64)),
        graph_index=torch.from_numpy(graph_set.graph_index.astype(np.int64)),
        graph_features=torch.from_numpy(graph_features),
        num_graphs=graph_set.num_graphs,
    )


def _one_hot(codes: np.ndarray, blocks: tuple[int, ...]) -> np.ndarray:
    """[rows, len(blocks)] of codes -> [rows, sum(blocks)] float32, one 1 per block."""
    offsets = np.cumsum((0,) + blocks[:-1])
    dense = np.zeros((len(codes), sum(blocks)), dtype=np.float32)
    dense[np.arange(len(codes))[:, None], codes + offsets] = 1.0
    return dense
