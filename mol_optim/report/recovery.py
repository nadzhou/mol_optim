import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from mol_optim import config
from mol_optim.chem import graph_key, molio, seeds
from mol_optim.datasets import bindingdb

ACTIVE = 8.0  # 10 nM — a lead series, the same threshold seeds.choose uses
POTENT = 9.0  # 1 nM

# Cap on drawn analogs. Every analog the run found is drawn regardless.
PANEL_ANALOGS = 16


@dataclass(frozen=True)
class Recovery:
    num_episodes: int
    num_distinct: int  # distinct terminal molecules the run produced
    num_analogs: int  # held-out analogs of this seed, by constitutional graph
    found: tuple[bindingdb.Compound, ...]  # the analogs the run actually built
    produced_seed: bool  # did the run ever hand the seed straight back
    num_known: int  # produced and measured, but neither an analog nor the seed
    num_novel: int  # produced and in no measured set


def held_out_analogs(
    compounds: tuple[bindingdb.Compound, ...], seed: bindingdb.Compound
) -> dict[str, bindingdb.Compound]:
    """Every measured compound on the seed's scaffold, keyed by constitutional graph.

    The seed is dropped: an agent that takes the no-op every step would otherwise
    "recover" the molecule it was handed.
    """
    seed_key = graph_key.canonical_hash(seed.mol)
    analogs: dict[str, bindingdb.Compound] = {}
    for compound in compounds:
        if compound.scaffold != seed.scaffold:
            continue
        key = graph_key.canonical_hash(compound.mol)
        if key == seed_key:
            continue
        # 576 records collapse to 566 graphs for seed 0; keep the more-measured record.
        if key not in analogs or compound.num_measurements > analogs[key].num_measurements:
            analogs[key] = compound
    return analogs


def measure(
    log_path: Path,
    analogs: dict[str, bindingdb.Compound],
    measured: frozenset[str],
    seed_key: str,
) -> Recovery:
    """`measured` is every hash in the dataset — what separates `known` from `novel`."""
    with open(log_path) as log_file:
        hashes = [row["graph_hash"] for row in csv.DictReader(log_file)]
    distinct = set(hashes)
    hits = distinct & set(analogs)
    rest = distinct - hits - {seed_key}
    return Recovery(
        num_episodes=len(hashes),
        num_distinct=len(distinct),
        num_analogs=len(analogs),
        found=tuple(sorted((analogs[key] for key in hits), key=lambda c: -c.pic50)),
        produced_seed=seed_key in distinct,
        num_known=len(rest & measured),
        num_novel=len(rest - measured),
    )


def panel(
    out_path: Path,
    seed: bindingdb.Compound,
    top: tuple[Chem.Mol, ...],
    top_legends: list[str],
    analogs: dict[str, bindingdb.Compound],
    found_keys: set[str],
) -> None:
    """Three stacked grids: the seed, what the run built, what it was trying to build.

    One picture per run, because a table saying "found 1 of 51" does not tell you whether
    the miss was a near-miss or a molecule falling apart. Only the drawing does.
    """
    # Found first, then the most potent: a run that finds nothing still gets a picture.
    ranked = sorted(analogs.items(), key=lambda kv: (kv[0] not in found_keys, -kv[1].pic50))
    shown = ranked[:PANEL_ANALOGS]
    omitted = len(ranked) - len(shown)

    blocks = [
        (
            f"SEED  pIC50 {seed.pic50:.2f}",
            [seed.mol],
            [seed.mol.GetProp("_Name")],
        ),
        (
            f"BUILT BY THE RUN  ({len(top)} best by reward)",
            list(top),
            list(top_legends),
        ),
        (
            f"HELD OUT  ({len(found_keys)} of {len(analogs)} found"
            + (f", {omitted} more not drawn)" if omitted else ")"),
            [compound.mol for _, compound in shown],
            [
                f"{'FOUND' if key in found_keys else 'missed'}  pIC50 {c.pic50:.2f}"
                for key, c in shown
            ],
        ),
    ]

    images = []
    for title, molecules, legends in blocks:
        flat = []
        for mol in molecules:
            copy = Chem.Mol(mol)
            AllChem.Compute2DCoords(copy)
            flat.append(copy)
        grid = Draw.MolsToGridImage(
            flat,
            molsPerRow=min(5, len(flat)),
            subImgSize=(300, 250),
            legends=legends,
            returnPNG=False,
        )
        banded = Image.new("RGB", (grid.width, grid.height + 34), "white")
        banded.paste(grid, (0, 34))
        Draw.rdMolDraw2D.MolDraw2DCairo(1, 1)  # ensures Cairo is loaded before text draw
        from PIL import ImageDraw

        ImageDraw.Draw(banded).text((10, 10), title, fill="black")
        images.append(banded)

    width = max(image.width for image in images)
    combined = Image.new("RGB", (width, sum(i.height for i in images)), "white")
    offset = 0
    for image in images:
        combined.paste(image, (0, offset))
        offset += image.height
    combined.save(out_path)


def run(settings: config.Settings) -> None:
    spec = settings.recovery
    if spec.seed_molecule is None:
        raise ValueError("recovery needs a seed_molecule: the scaffold it scores against")
    compounds = bindingdb.load(settings.bindingdb.path)
    seed = seeds.choose(compounds)[spec.seed_molecule]
    analogs = held_out_analogs(compounds, seed)
    seed_key = graph_key.canonical_hash(seed.mol)
    measured = frozenset(graph_key.canonical_hash(c.mol) for c in compounds)

    active = sum(1 for c in analogs.values() if c.pic50 >= ACTIVE)
    potent = sum(1 for c in analogs.values() if c.pic50 >= POTENT)
    print(
        f"seed {spec.seed_molecule}: measured pIC50 {seed.pic50:.2f}, "
        f"{len(analogs)} held-out analogs ({active} active, {potent} potent), "
        f"pIC50 {min(c.pic50 for c in analogs.values()):.2f} to "
        f"{max(c.pic50 for c in analogs.values()):.2f}\n"
    )

    print(
        f"{'run':<28} {'eps':>6} {'distinct':>9} {'found':>6} {'>=8':>4} {'>=9':>4} "
        f"{'recovery':>9} {'known':>7} {'novel':>7} {'seed?':>6}"
    )
    union: set[str] = set()
    for log_path in spec.logs:
        recovery = measure(log_path, analogs, measured, seed_key)
        hit_keys = {graph_key.canonical_hash(c.mol) for c in recovery.found}
        union |= hit_keys
        print(
            f"{log_path.stem:<28} {recovery.num_episodes:>6} {recovery.num_distinct:>9} "
            f"{len(recovery.found):>6} "
            f"{sum(1 for c in recovery.found if c.pic50 >= ACTIVE):>4} "
            f"{sum(1 for c in recovery.found if c.pic50 >= POTENT):>4} "
            f"{100 * len(recovery.found) / len(analogs):>8.1f}% "
            f"{recovery.num_known:>7} {recovery.num_novel:>7} "
            f"{'yes' if recovery.produced_seed else 'no':>6}"
        )
        for compound in recovery.found:
            print(f"      {compound.pic50:>6.2f}  {compound.mol.GetProp('_Name')}")

        top_path = log_path.with_name(f"{log_path.stem}_top.sdf")
        if top_path.exists():
            top = molio.read(top_path)
            panel(
                log_path.with_name(f"{log_path.stem}_panel.png"),
                seed,
                top,
                [f"reward {mol.GetProp('reward')}" for mol in top],
                analogs,
                hit_keys,
            )

    if len(spec.logs) > 1:
        print(
            f"\nunion across {len(spec.logs)} runs: {len(union)} of {len(analogs)} "
            f"analogs, recovery {100 * len(union) / len(analogs):.1f}%"
        )
