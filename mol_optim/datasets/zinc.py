"""ZINC, the unlabelled molecules the encoder is pretrained on.

Molecules arrive as text because ZINC is published that way; nothing downstream of
`molecules()` handles a SMILES string. The URL and hash live in the config file, so
swapping in another unlabelled set is a config change, not a code change.
"""

import hashlib
import urllib.request
from pathlib import Path

from rdkit import Chem

from mol_optim import config

# The default in config.ZincSpec is tdc.metadata.name2id["zinc"] — the file
# tdc.generation.MolGen(name="ZINC") downloads: 249,455 drug-like molecules, the 250k set
# the MolDQN and JT-VAE papers use.


def molecules(path: Path, limit: int | None = None) -> tuple[Chem.Mol, ...]:
    """The first `limit` molecules, in file order. About 10 s for all of ZINC.

    File order is arbitrary but fixed, so a prefix is reproducible. The train/held-out
    split is a seeded shuffle in pretrain.py, not a cut of this order.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the 'zinc' step first.")
    with open(path) as data_file:
        header = next(data_file).strip()
        if header != "smiles":
            raise ValueError(f"expected a single 'smiles' column in {path}, got {header}")
        mols = []
        for line in data_file:
            record = line.strip().strip('"')
            if not record:
                continue
            mol = Chem.MolFromSmiles(record)
            if mol is None:
                # All 249,455 parse today; a failure means the pinned file changed.
                raise ValueError(f"RDKit could not read {record!r} in {path}")
            mols.append(mol)
            if limit is not None and len(mols) == limit:
                break
    return tuple(mols)


def run(settings: config.Settings) -> None:
    spec = settings.zinc
    if not spec.path.exists():
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec.url}")
        urllib.request.urlretrieve(spec.url, spec.path)
    digest = hashlib.sha256(spec.path.read_bytes()).hexdigest()
    if digest != spec.sha256:
        raise ValueError(
            f"{spec.path} hashes to {digest}, not the pinned {spec.sha256}. "
            "The upstream file changed; re-check it before trusting it."
        )
    mols = molecules(spec.path)
    atoms = sum(mol.GetNumAtoms() for mol in mols)
    print(
        f"{spec.path} — {len(mols)} molecules, {atoms} heavy atoms, "
        f"{atoms / len(mols):.1f} per molecule"
    )
