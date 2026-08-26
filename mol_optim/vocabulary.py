"""The fragment vocabulary: precedented decorations, cut from the target's own actives.

This is the constraint the reward terms could not supply. The pIC50 run measured the
agent putting an N–N bond in 100% of its episodes against a reward that had no term for
it; a structural
alert is a term the agent trades against predicted pIC50, and a vocabulary has no
exchange rate. Every fragment here was cut out of a measured EGFR inhibitor, so the
question "could the agent build a pentazane" is answered by the vocabulary not
containing one, rather than by a penalty the agent can price.

Single-attachment pieces only. A BRICS fragment with two open ends is a linker or a
core, and inserting one is a different edit from hanging a group off a free valence —
that edit is not in this action space.

Run once:
    .venv/bin/python -m mol_optim.vocabulary --out data/egfr_fragments.sdf
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import BRICS

from mol_optim import bindingdb, graph_key, molio


@dataclass(frozen=True)
class Fragment:
    """One decoration, and the atom of it that takes the new bond."""

    mol: Chem.Mol  # the group with its BRICS dummy removed
    attachment_idx: int  # index into mol; the atom that loses an H to the new bond
    count: int  # how many of the target's actives this was cut from


def build(
    compounds: Sequence[bindingdb.Compound],
    size: int = 50,
    min_pic50: float = 7.0,
    max_heavy_atoms: int = 12,
) -> tuple[Fragment, ...]:
    """The `size` most frequent single-attachment BRICS fragments, most frequent first.

    min_pic50 7.0 is 100 nM: cut fragments out of compounds that bind, not out of
    everything ever measured against the target. max_heavy_atoms keeps the vocabulary to
    decorations — the raw decomposition's largest single-attachment piece is 33 heavy
    atoms, which is not a substituent, it is another molecule.
    """
    counts: Counter[str] = Counter()
    pieces: dict[str, Chem.Mol] = {}

    for compound in compounds:
        if compound.pic50 < min_pic50:
            continue
        for piece in BRICS.BRICSDecompose(compound.mol, returnMols=True):
            dummies = [
                atom.GetIdx() for atom in piece.GetAtoms() if atom.GetAtomicNum() == 0
            ]
            if len(dummies) != 1:
                continue
            if piece.GetNumHeavyAtoms() - 1 > max_heavy_atoms:
                continue
            # Hashed with the dummy still on, so a meta and a para attachment of one
            # ring are two fragments rather than one.
            key = graph_key.canonical_hash(piece)
            counts[key] += 1
            pieces.setdefault(key, piece)

    # Count then key: two fragments cut from the same number of actives must not swap
    # places between runs.
    ranked = sorted(counts, key=lambda key: (-counts[key], key))[:size]
    return tuple(_strip_dummy(pieces[key], counts[key]) for key in ranked)


def _strip_dummy(piece: Chem.Mol, count: int) -> Fragment:
    """The BRICS piece minus its dummy atom, remembering where the dummy was.

    The attachment atom gets its hydrogen back here and gives it up again when the
    fragment is attached, so the free-standing fragment is a real molecule and
    sanitizes on its own.
    """
    dummy_idx = next(
        atom.GetIdx() for atom in piece.GetAtoms() if atom.GetAtomicNum() == 0
    )
    attachment_idx = piece.GetAtomWithIdx(dummy_idx).GetNeighbors()[0].GetIdx()

    work = Chem.RWMol(piece)
    work.RemoveAtom(dummy_idx)
    mol = work.GetMol()
    Chem.SanitizeMol(mol)
    return Fragment(
        mol=mol,
        # RemoveAtom shifts every index above the removed one down by one.
        attachment_idx=attachment_idx - 1 if attachment_idx > dummy_idx else attachment_idx,
        count=count,
    )


def write(path: Path, fragments: Sequence[Fragment]) -> None:
    """The vocabulary as an SDF — an atom table, so the attachment index still points
    at the same atom when it is read back."""
    molio.write(
        path,
        tuple(fragment.mol for fragment in fragments),
        {
            "attachment_idx": [fragment.attachment_idx for fragment in fragments],
            "count": [fragment.count for fragment in fragments],
        },
    )


def load(path: Path) -> tuple[Fragment, ...]:
    """The vocabulary written by `write`, in the order it was written."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with: "
            f"python -m mol_optim.vocabulary --out {path}"
        )
    return tuple(
        Fragment(
            mol=mol,
            attachment_idx=int(mol.GetProp("attachment_idx")),
            count=int(mol.GetProp("count")),
        )
        for mol in molio.read(path)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/egfr_fragments.sdf"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--min-pic50", type=float, default=7.0)
    parser.add_argument("--max-heavy-atoms", type=int, default=12)
    args = parser.parse_args()

    compounds = bindingdb.load()
    fragments = build(compounds, args.size, args.min_pic50, args.max_heavy_atoms)
    write(args.out, fragments)

    total = sum(fragment.count for fragment in fragments)
    print(f"wrote {args.out}: {len(fragments)} fragments, {total} occurrences")
    for fragment in fragments:
        # [*:1] marks the attachment atom. Two fragments here can be the same molecule
        # attached at different atoms — ortho and para fluorophenyl — and printing a
        # plain SMILES makes them look like a duplicated entry.
        marked = Chem.RWMol(fragment.mol)
        marked.GetAtomWithIdx(fragment.attachment_idx).SetAtomMapNum(1)
        print(
            f"  {fragment.count:>5}  {fragment.mol.GetNumHeavyAtoms():>2} heavy atoms  "
            f"{Chem.MolToSmiles(marked)}"
        )
