"""Substituent-level edits. The median held-out target is 6 atom edits from the seed but
differs by one connected group, so the substituent is the action, not the atom.

The library is mined from the measured set, so every action attaches something real.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem

MAX_FRAGMENT_ATOMS = 12


@dataclass(frozen=True)
class Fragment:
    mol: Chem.Mol
    smiles: str
    count: int


def _split_out_substituent(mol: Chem.Mol, bond: Chem.Bond) -> Chem.Mol | None:
    fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
    pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
    if len(pieces) != 2:
        return None
    smaller = min(pieces, key=lambda piece: piece.GetNumHeavyAtoms())
    if smaller.GetNumHeavyAtoms() - 1 > MAX_FRAGMENT_ATOMS:  # the dummy counts as heavy
        return None
    for atom in smaller.GetAtoms():
        atom.SetIsotope(0)
    if Chem.SanitizeMol(smaller, catchErrors=True):
        return None
    return smaller


def library(
    molecules: Sequence[Chem.Mol], min_count: int = 10, max_size: int = 40
) -> tuple[Fragment, ...]:
    """The most common substituents, counted once per compound that wears one."""
    counts: Counter[str] = Counter()
    examples: dict[str, Chem.Mol] = {}
    for mol in molecules:
        seen: set[str] = set()
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            piece = _split_out_substituent(mol, bond)
            if piece is None:
                continue
            smiles = Chem.MolToSmiles(piece)
            seen.add(smiles)
            examples.setdefault(smiles, piece)
        counts.update(seen)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(
        Fragment(mol=examples[smiles], smiles=smiles, count=count)
        for smiles, count in ranked[:max_size]
        if count >= min_count
    )


def attach(core: Chem.Mol, atom_idx: int, fragment: Fragment) -> Chem.Mol | None:
    combined = Chem.RWMol(Chem.CombineMols(core, fragment.mol))
    offset = core.GetNumAtoms()
    dummy_idx = next(
        atom.GetIdx()
        for atom in combined.GetAtoms()
        if atom.GetIdx() >= offset and atom.GetAtomicNum() == 0
    )
    anchor_idx = combined.GetAtomWithIdx(dummy_idx).GetNeighbors()[0].GetIdx()
    combined.AddBond(atom_idx, anchor_idx, Chem.BondType.SINGLE)
    combined.RemoveAtom(dummy_idx)  # last: it renumbers everything above it
    if Chem.SanitizeMol(combined, catchErrors=True):
        return None
    return combined.GetMol()


def detach(mol: Chem.Mol, bond: Chem.Bond) -> Chem.Mol | None:
    fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
    pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
    if len(pieces) != 2:
        return None
    larger = max(pieces, key=lambda piece: piece.GetNumHeavyAtoms())
    if Chem.SanitizeMol(larger, catchErrors=True):
        return None
    return larger


def _reattachment_site(
    state: Chem.Mol, bond: Chem.Bond, trimmed: Chem.Mol
) -> int | None:
    fragmented = Chem.FragmentOnBonds(state, [bond.GetIdx()], addDummies=False)
    groups = Chem.GetMolFrags(fragmented)
    if len(groups) != 2:
        return None
    kept = max(groups, key=len)
    for end in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
        if end in kept:
            return kept.index(end)
    return None


def substitutions(
    state: Chem.Mol, fragment_library: tuple[Fragment, ...]
) -> list[Chem.Mol]:
    """Every molecule one substituent edit away: attach, detach, or swap.

    Not deduplicated; environment._deduplicated does that for both action spaces.
    """
    candidates: list[Chem.Mol] = []

    for atom in state.GetAtoms():
        if atom.GetNumImplicitHs() < 1:
            continue
        for fragment in fragment_library:
            grown = attach(state, atom.GetIdx(), fragment)
            if grown is not None:
                candidates.append(grown)

    for bond in state.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if _split_out_substituent(state, bond) is None:
            continue
        trimmed = detach(state, bond)
        if trimmed is None:
            continue
        candidates.append(trimmed)
        site = _reattachment_site(state, bond, trimmed)
        if site is None:
            continue
        for fragment in fragment_library:
            swapped = attach(trimmed, site, fragment)
            if swapped is not None:
                candidates.append(swapped)

    return candidates
