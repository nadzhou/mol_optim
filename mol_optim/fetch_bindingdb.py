"""BindingDB's IC50 table to the EGFR dataset the regressor trains on. Run once:

    python -m mol_optim.fetch_bindingdb

Downloads a dated BindingDB snapshot (593 MB) unless it is there, checks it against the
MD5 BindingDB publishes beside it, streams the 9 GB table once, and writes
`data/egfr_ic50.sdf`. Four cleaning decisions, each silent if skipped:

- One construct, not one UniProt id. P00533 covers 51 EGFR constructs here; pooling wild
  type with T790M puts one compound's two very different numbers under one label.
- Qualified values go. 5,720 of EGFR's 29,193 IC50 rows are ">" or "<" — the assay ran
  off its range. Kept as bare numbers they pile up at round values and get learned.
- nM to pIC50, see `bindingdb.to_pic50`.
- Duplicates median-aggregated by stereo-aware key, with the spread kept as a property.

Then the file is read back and every key recomputed: a handful of macrocycles change
name in transit, because their geometry does not survive being drawn in 2D, and they are
dropped rather than left to make "the same compound is in both splits" untrue.
"""

import hashlib
import io
import urllib.request
import zipfile
from collections import defaultdict

from rdkit import Chem

from mol_optim import bindingdb, graph_key, molio

SNAPSHOT_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
# Published beside the file. MD5 because that is the digest upstream commits to.
SNAPSHOT_MD5 = "dac667f2d194ae6744104a7f87549db5"
SNAPSHOT_PATH = bindingdb.DATASET_PATH.with_name("BindingDB_All_202608_tsv.zip")
TABLE_NAME = "BindingDB_All.tsv"

# Column positions in that table's 640, read once from its header.
SMILES, IC50, ORGANISM, TARGET_NAME, NUM_CHAINS, UNIPROT = 1, 9, 7, 6, 39, 44
TARGET_UNIPROT = "P00533"
TARGET_CONSTRUCT = "Epidermal growth factor receptor"  # wild type: no bracketed edits


if __name__ == "__main__":
    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SNAPSHOT_URL}")
        urllib.request.urlretrieve(SNAPSHOT_URL, SNAPSHOT_PATH)
    digest = hashlib.md5(SNAPSHOT_PATH.read_bytes()).hexdigest()
    if digest != SNAPSHOT_MD5:
        raise ValueError(
            f"{SNAPSHOT_PATH} hashes to {digest}, not the pinned {SNAPSHOT_MD5}. "
            "BindingDB dates its snapshots; a different digest is a different month."
        )

    measurements: dict[str, list[float]] = defaultdict(list)
    representative: dict[str, Chem.Mol] = {}
    scanned = target_rows = qualified = unusable_value = unreadable = 0

    with zipfile.ZipFile(SNAPSHOT_PATH).open(TABLE_NAME) as raw:
        table = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        header = next(table).rstrip("\n").split("\t")
        if header[IC50] != "IC50 (nM)" or header[UNIPROT] != (
            "UniProt (SwissProt) Primary ID of Target Chain 1"
        ):
            raise ValueError(
                f"{TABLE_NAME}'s columns moved: {IC50} is {header[IC50]!r} and "
                f"{UNIPROT} is {header[UNIPROT]!r}. The positions above are wrong for "
                "this snapshot."
            )

        for line in table:
            scanned += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= UNIPROT:
                continue
            if fields[UNIPROT].strip() != TARGET_UNIPROT:
                continue
            if fields[TARGET_NAME].strip() != TARGET_CONSTRUCT:
                continue
            # A multi-chain entry is a complex; the IC50 belongs to the complex.
            if fields[NUM_CHAINS].strip() not in ("1", ""):
                continue
            target_rows += 1

            value = fields[IC50].strip()
            if not value:
                continue
            if value[0] in "<>":
                qualified += 1
                continue
            try:
                ic50_nm = float(value)
                pic50 = bindingdb.to_pic50(ic50_nm)
            except ValueError:
                unusable_value += 1
                continue

            mol = Chem.MolFromSmiles(fields[SMILES])
            if mol is None or mol.GetNumAtoms() == 0:
                unreadable += 1
                continue
            # Salts: the measurement is the largest fragment's, not "compound + HCl".
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            mol = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())

            key = graph_key.stereo_hash(mol)
            measurements[key].append(pic50)
            representative.setdefault(key, mol)

    def write(keys: list[str]) -> None:
        bindingdb.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        molecules = []
        for key in keys:
            mol = Chem.Mol(representative[key])
            mol.SetProp("_Name", key)
            molecules.append(mol)
        molio.write(
            bindingdb.DATASET_PATH,
            tuple(molecules),
            {
                "pic50": [f"{bindingdb.median(measurements[k]):.4f}" for k in keys],
                "num_measurements": [len(measurements[k]) for k in keys],
                "pic50_spread": [
                    f"{max(measurements[k]) - min(measurements[k]):.4f}" for k in keys
                ],
                "target": [TARGET_UNIPROT] * len(keys),
                "snapshot": [SNAPSHOT_PATH.stem] * len(keys),
            },
        )

    keys = list(measurements)
    write(keys)
    from_disk = molio.read_named(bindingdb.DATASET_PATH)
    renamed = [key for key, mol in from_disk.items() if graph_key.stereo_hash(mol) != key]
    if renamed:
        keys = [key for key in keys if key not in set(renamed)]
        write(keys)
        from_disk = molio.read_named(bindingdb.DATASET_PATH)
        still_renamed = [
            key for key, mol in from_disk.items() if graph_key.stereo_hash(mol) != key
        ]
        if still_renamed:
            raise ValueError(
                f"{len(still_renamed)} compounds still change name on the round trip "
                "after dropping the ones that did; the key is not stable"
            )

    labels = [bindingdb.median(measurements[key]) for key in keys]
    repeated = sum(1 for key in keys if len(measurements[key]) > 1)
    print(
        f"scanned {scanned} rows\n"
        f"{target_rows} for {TARGET_CONSTRUCT} ({TARGET_UNIPROT}), single chain\n"
        f"dropped {qualified} qualified, {unusable_value} unusable values, "
        f"{unreadable} unreadable structures, {len(renamed)} renamed by the round trip\n"
        f"wrote {bindingdb.DATASET_PATH} — {len(keys)} compounds, "
        f"{repeated} of them measured more than once, "
        f"pIC50 {min(labels):.2f} to {max(labels):.2f}"
    )
