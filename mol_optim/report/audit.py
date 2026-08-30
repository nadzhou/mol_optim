from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from mol_optim import config
from mol_optim.chem import molio, seeds

# Structures that have turned up in runs of this project. Aromatic forms are excluded
# where they are ordinary chemistry: a pyrazole is not a hydrazine.
MOTIFS: dict[str, Chem.Mol] = {
    "hemiaminal": Chem.MolFromSmarts("[#7][#6X4][OX2H1]"),
    "aminal": Chem.MolFromSmarts("[#7][#6X4][#7]"),
    "gem-diol": Chem.MolFromSmarts("[OX2H1][#6X4][OX2H1]"),
    "N-hydroxyl": Chem.MolFromSmarts("[#7][OX2H1]"),
    "N-N": Chem.MolFromSmarts("[#7;!a]-[#7;!a]"),
    "N-N-N": Chem.MolFromSmarts("[#7;!a]-[#7;!a]-[#7;!a]"),
}


@dataclass(frozen=True)
class Audit:
    motif_counts: dict[str, int]  # matches per motif, zeros included
    num_heavy_atoms: int
    num_nitrogens: int
    # Separate from the N-N motif: a fused triazine's NH reads as aromatic and that
    # SMARTS misses it, which is how the audit first came back clean.
    num_nitrogen_nitrogen_bonds: int
    scaffold_intact: bool | None  # None when no scaffold was given to check against


def audit(mol: Chem.Mol, scaffold: Chem.Mol | None = None) -> Audit:
    return Audit(
        motif_counts={
            name: len(mol.GetSubstructMatches(pattern))
            for name, pattern in MOTIFS.items()
        },
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_nitrogens=sum(
            1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7
        ),
        num_nitrogen_nitrogen_bonds=sum(
            1
            for bond in mol.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() == 7
            and bond.GetEndAtom().GetAtomicNum() == 7
        ),
        scaffold_intact=(
            None if scaffold is None else mol.HasSubstructMatch(scaffold)
        ),
    )


def scaffold_of(mol: Chem.Mol) -> Chem.Mol:
    """The Bemis-Murcko frame — what "the scaffold survived" is checked against."""
    return MurckoScaffold.GetScaffoldForMol(mol)


def run(settings: config.Settings) -> None:
    spec = settings.audit
    seed = seeds.molecule(settings.bindingdb.path, spec.seed_molecule)
    scaffold = None if seed is None else scaffold_of(seed)
    if scaffold is not None:
        print(f"seed {spec.seed_molecule}: {Chem.MolToSmiles(scaffold)}")
        print(f"  {audit(seed, scaffold)}\n")

    for sdf_path in spec.sdf:
        molecules = molio.read(sdf_path)
        audits = [audit(mol, scaffold) for mol in molecules]
        print(f"{sdf_path}")
        print(f"{'#':>3} {'atoms':>6} {'N':>3} {'N-N':>4} {'scaffold':>9}  motifs")
        for index, row in enumerate(audits):
            hits = ", ".join(
                f"{name} x{count}" for name, count in row.motif_counts.items() if count
            )
            intact = (
                "-" if row.scaffold_intact is None else ("yes" if row.scaffold_intact else "NO")
            )
            print(
                f"{index:>3} {row.num_heavy_atoms:>6} {row.num_nitrogens:>3} "
                f"{row.num_nitrogen_nitrogen_bonds:>4} {intact:>9}  {hits or '-'}"
            )

        print(f"\nover {len(audits)} molecules:")
        for name in MOTIFS:
            carrying = sum(1 for row in audits if row.motif_counts[name])
            print(f"  {name:>12}: {carrying}/{len(audits)}")
        carrying = sum(1 for row in audits if row.num_nitrogen_nitrogen_bonds)
        print(f"  {'any N-N bond':>12}: {carrying}/{len(audits)}")
        if scaffold is not None:
            intact = sum(1 for row in audits if row.scaffold_intact)
            print(f"  {'scaffold':>12}: {intact}/{len(audits)} intact")
        print()
