"""ZINC, the unlabelled molecules the encoder is pretrained on. Run once:

    python -m mol_optim.zinc

Downloads TDC's `zinc.tab` — 249,455 drug-like molecules, the 250k set the MolDQN and
JT-VAE papers use — checks the pinned hash, and reads every record once.

Molecules arrive as text here because ZINC is published that way; nothing downstream of
`molecules()` handles a SMILES string. The 12 MB file stays out of version control, and
the URL and hash below say exactly which molecules the checkpoint was pretrained on.
"""

import hashlib
import urllib.request
from pathlib import Path

from rdkit import Chem

# tdc.metadata.name2id["zinc"] — the file tdc.generation.MolGen(name="ZINC") downloads.
DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/4170963"
DATA_SHA256 = "b65ee88f1838586571fc41200ee60fb7b97da55da72823bed72dc315af2fb48b"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "zinc.tab"


def molecules(path: Path = DATA_PATH, limit: int | None = None) -> tuple[Chem.Mol, ...]:
    """The first `limit` ZINC molecules, in file order. About 10 s for all of them.

    File order is arbitrary but fixed, so a prefix is reproducible. The train/held-out
    split is a seeded shuffle in pretrain.py, not a cut of this order.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download it once with: python -m mol_optim.zinc"
        )
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


if __name__ == "__main__":
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(
            f"{DATA_PATH} hashes to {digest}, not the pinned {DATA_SHA256}. "
            "The upstream file changed; re-check it against TDC before trusting it."
        )
    mols = molecules()
    atoms = sum(mol.GetNumAtoms() for mol in mols)
    print(
        f"{DATA_PATH} — {len(mols)} molecules, {atoms} heavy atoms, "
        f"{atoms / len(mols):.1f} per molecule"
    )
