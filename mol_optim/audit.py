"""What did the agent actually build? Substructure counts over a run's molecules.

Every run so far has ended the same way: a reward curve that climbs, and a top-k of
molecules no chemist would order. The descriptor scorers found hemiaminals, the pIC50
regressor found chains of catenated nitrogen, and those were found by looking at a
drawing after the earlier motif list came back empty. So the list is a record of what
has been found, not a guarantee of what is there — when a new run scores well, look at
the picture too.

Pure: a molecule in, counts out. Nothing here reads a file or holds state.
"""

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from mol_optim import config, molio, seeds

# Each entry is a structure that has actually turned up in a run of this project, with
# the reward that produced it. Aromatic rings are excluded where the aromatic form is
# ordinary chemistry — a pyrazole is not a hydrazine.
MOTIFS: dict[str, Chem.Mol] = {
    # From the descriptor-scored runs: these fall apart in water.
    "hemiaminal": Chem.MolFromSmarts("[#7][#6X4][OX2H1]"),
    "aminal": Chem.MolFromSmarts("[#7][#6X4][#7]"),
    "gem-diol": Chem.MolFromSmarts("[OX2H1][#6X4][OX2H1]"),
    "N-hydroxyl": Chem.MolFromSmarts("[#7][OX2H1]"),
    # From the pIC50 regressor run: hydrazines and longer polyazanes.
    "N-N": Chem.MolFromSmarts("[#7;!a]-[#7;!a]"),
    "N-N-N": Chem.MolFromSmarts("[#7;!a]-[#7;!a]-[#7;!a]"),
}


@dataclass(frozen=True)
class Audit:
    """One molecule, read for the things past runs have gone wrong in."""

    motif_counts: dict[str, int]  # matches per motif, zeros included
    num_heavy_atoms: int
    num_nitrogens: int
    # Nitrogen bonded to nitrogen, aromatic or not. Counted separately from the N-N
    # motif because a fused triazine's contiguous NH read as aromatic and the SMARTS
    # misses them, which is how that audit came back clean the first time.
    num_nitrogen_nitrogen_bonds: int
    scaffold_intact: bool | None  # None when no scaffold was given to check against


def audit(mol: Chem.Mol, scaffold: Chem.Mol | None = None) -> Audit:
    """Every count for one molecule."""
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
    """The Bemis-Murcko frame, which is what "the scaffold survived" is checked against."""
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
