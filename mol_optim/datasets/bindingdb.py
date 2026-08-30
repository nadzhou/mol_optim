import hashlib
import io
import math
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import graph_key, molio

# Column positions in that table's 640, checked against its header in `run`.
SMILES, IC50, ORGANISM, TARGET_NAME, NUM_CHAINS, UNIPROT = 1, 9, 7, 6, 39, 44


@dataclass(frozen=True)
class Compound:
    mol: Chem.Mol
    pic50: float
    num_measurements: int  # how many BindingDB rows the median was taken over
    pic50_spread: float  # max - min across those rows; 0.0 for a single measurement
    scaffold: str


def to_pic50(ic50_nm: float) -> float:
    """IC50 in nanomolar to pIC50 = -log10(IC50 in molar). 1 nM is 9.0, 1 uM is 6.0."""
    if ic50_nm <= 0.0:
        raise ValueError(f"IC50 must be positive, got {ic50_nm} nM")
    return 9.0 - math.log10(ic50_nm)


def median(values: list[float]) -> float:
    # Median, not mean: lab-to-lab disagreements reach 8 logs.
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load(path: Path) -> tuple[Compound, ...]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the 'bindingdb' step first.")
    named = molio.read_named(path)
    return tuple(
        Compound(
            mol=mol,
            pic50=float(mol.GetProp("pic50")),
            num_measurements=int(mol.GetProp("num_measurements")),
            pic50_spread=float(mol.GetProp("pic50_spread")),
            scaffold=graph_key.scaffold_hash(mol),
        )
        for mol in named.values()
    )


def run(settings: config.Settings) -> None:
    spec = settings.bindingdb
    if not spec.archive.exists():
        spec.archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec.url}")
        urllib.request.urlretrieve(spec.url, spec.archive)
    digest = hashlib.md5(spec.archive.read_bytes()).hexdigest()
    if digest != spec.md5:
        raise ValueError(
            f"{spec.archive} hashes to {digest}, not the pinned {spec.md5}. "
            "BindingDB dates its snapshots; a different digest is a different month."
        )

    measurements: dict[str, list[float]] = defaultdict(list)
    representative: dict[str, Chem.Mol] = {}
    scanned = target_rows = qualified = unusable_value = unreadable = 0

    with zipfile.ZipFile(spec.archive).open(spec.table) as raw:
        table = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        header = next(table).rstrip("\n").split("\t")
        if header[IC50] != "IC50 (nM)" or header[UNIPROT] != (
            "UniProt (SwissProt) Primary ID of Target Chain 1"
        ):
            raise ValueError(
                f"{spec.table}'s columns moved: {IC50} is {header[IC50]!r} and "
                f"{UNIPROT} is {header[UNIPROT]!r}. The positions above are wrong for "
                "this snapshot."
            )

        for line in table:
            scanned += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= UNIPROT:
                continue
            if fields[UNIPROT].strip() != spec.uniprot:
                continue
            if fields[TARGET_NAME].strip() != spec.construct:
                continue
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
                pic50 = to_pic50(ic50_nm)
            except ValueError:
                unusable_value += 1
                continue

            mol = Chem.MolFromSmiles(fields[SMILES])
            if mol is None or mol.GetNumAtoms() == 0:
                unreadable += 1
                continue
            # The measurement is the largest fragment's, not "compound + HCl".
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            mol = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())

            key = graph_key.stereo_hash(mol)
            measurements[key].append(pic50)
            representative.setdefault(key, mol)

    def write(keys: list[str]) -> None:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        molecules = []
        for key in keys:
            mol = Chem.Mol(representative[key])
            mol.SetProp("_Name", key)
            molecules.append(mol)
        molio.write(
            spec.path,
            tuple(molecules),
            {
                "pic50": [f"{median(measurements[k]):.4f}" for k in keys],
                "num_measurements": [len(measurements[k]) for k in keys],
                "pic50_spread": [
                    f"{max(measurements[k]) - min(measurements[k]):.4f}" for k in keys
                ],
                "target": [spec.uniprot] * len(keys),
                "snapshot": [spec.archive.stem] * len(keys),
            },
        )

    # Read back with keys recomputed: a few macrocycles change name in transit and
    # are dropped, or they would land in both splits.
    keys = list(measurements)
    write(keys)
    from_disk = molio.read_named(spec.path)
    renamed = [key for key, mol in from_disk.items() if graph_key.stereo_hash(mol) != key]
    if renamed:
        keys = [key for key in keys if key not in set(renamed)]
        write(keys)
        from_disk = molio.read_named(spec.path)
        still_renamed = [
            key for key, mol in from_disk.items() if graph_key.stereo_hash(mol) != key
        ]
        if still_renamed:
            raise ValueError(
                f"{len(still_renamed)} compounds still change name on the round trip "
                "after dropping the ones that did; the key is not stable"
            )

    labels = [median(measurements[key]) for key in keys]
    repeated = sum(1 for key in keys if len(measurements[key]) > 1)
    print(
        f"scanned {scanned} rows\n"
        f"{target_rows} for {spec.construct} ({spec.uniprot}), single chain\n"
        f"dropped {qualified} qualified, {unusable_value} unusable values, "
        f"{unreadable} unreadable structures, {len(renamed)} renamed by the round trip\n"
        f"wrote {spec.path} — {len(keys)} compounds, "
        f"{repeated} of them measured more than once, "
        f"pIC50 {min(labels):.2f} to {max(labels):.2f}"
    )
