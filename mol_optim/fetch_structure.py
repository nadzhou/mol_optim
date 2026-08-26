"""Run once: a PDB entry in, a Vina-ready receptor and its co-crystal ligand out.

    .venv/bin/python -m mol_optim.fetch_structure

Two files come out of this and both are committed, for the reason the GSK3B forest is:
what a run docked against has to be inspectable later, not recomputed from whatever RCSB
serves that day. The entry is pinned by SHA-256 for the same reason.

The receptor is typed by Open Babel rather than Meeko. Meeko's polymer path raises
"Updated 1 H positions but deleted 3" on this entry's terminal residues — it queues three
hydrogens for deletion on one nitrogen and can place only one back. Open Babel is the
conventional route for a Vina receptor and this is a one-time conversion whose output is
read back and checked below.
"""

import hashlib
import subprocess
import urllib.request
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from mol_optim import molio

# EGFR kinase domain with erlotinib bound. Wild type, which is the construct the pIC50
# regressor was trained on — see plan.md Step 4, "One construct, not one UniProt id".
PDB_ID = "1M17"
LIGAND_CODE = "AQ4"  # erlotinib, as the entry names it
LIGAND_SMILES = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
SHA256 = "5b020cd186307d657de80611793554b074f6936a3443d33f81d9b1a58004962e"


def download(destination: Path) -> Path:
    """The entry itself, checked against the pin."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if SHA256 and digest != SHA256:
        raise ValueError(
            f"{destination} hashes to {digest}, not the pinned {SHA256}. "
            "RCSB re-released the entry; check what changed before updating the pin."
        )
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/structures"))
    args = parser.parse_args()

    entry = download(args.out_dir / f"{PDB_ID}.pdb")

    # Protein and ligand, split. Waters go: Vina has no term for a bridging water, and
    # leaving them in would block the site rather than mediate anything.
    protein, ligand_lines = [], []
    for line in entry.read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            # One conformer only. Meeko and Open Babel both read a second altloc as a
            # duplicate atom sitting on top of the first.
            if line[16] in (" ", "A"):
                protein.append(line[:16] + " " + line[17:])
        elif line.startswith("HETATM") and line[17:20].strip() == LIGAND_CODE:
            ligand_lines.append(line)

    protein_pdb = args.out_dir / f"{PDB_ID}_protein.pdb"
    protein_pdb.write_text("".join(protein) + "END\n")
    print(f"{len(protein)} protein atoms, {len(ligand_lines)} ligand atoms")

    # The ligand keeps its crystal coordinates and gets its bond orders from the
    # template — a PDB record carries neither bond orders nor charges.
    ligand_pdb = args.out_dir / f"{PDB_ID}_ligand.pdb"
    ligand_pdb.write_text("".join(ligand_lines) + "END\n")
    ligand = AllChem.AssignBondOrdersFromTemplate(
        Chem.MolFromSmiles(LIGAND_SMILES),
        Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True),
    )
    ligand.SetProp("_Name", f"{PDB_ID}_{LIGAND_CODE}")
    molio.write(args.out_dir / f"{PDB_ID}_ligand.sdf", (ligand,), {})
    ligand_pdb.unlink()

    receptor = args.out_dir / f"{PDB_ID}_receptor.pdbqt"
    subprocess.run(
        ["obabel", str(protein_pdb), "-O", str(receptor), "-xr", "-p", "7.4"],
        check=True,
        capture_output=True,
    )
    typed = [l for l in receptor.read_text().splitlines() if l.startswith("ATOM")]
    if not typed:
        raise ValueError(f"{receptor} has no atoms; Open Babel wrote nothing usable")
    print(f"wrote {receptor}: {len(typed)} typed atoms")
    print(f"wrote {args.out_dir / f'{PDB_ID}_ligand.sdf'}: the box is centred on it")
