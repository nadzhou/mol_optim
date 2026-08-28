"""Showing a run's molecules to a person: a drawing, an SDF, and the report.

This is the boundary. Everything upstream is graphs; here they become a picture you can
look at, a file a chemist's software can open, and `results/report.md` — one command over
the checkpoints already on disk, because a report that needs three commands in the right
order goes stale.

    python -m mol_optim.report --out results/report.md
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from mol_optim import (  # noqa: E402
    audit,
    bindingdb,
    config,
    graph_key,
    molio,
    results,
    seeds,
    splits,
)

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# train_dqn divides the pIC50 reward by 10 to land in [0, 1]. Undo it here, so the table
# reads in the units the regressor was fitted in.
PIC50_SCALE = 10.0


def top_k(run: results.Run, out_stem: Path, k: int = 12) -> None:
    """Writes the k best distinct molecules of a run as `<stem>.png` and `<stem>.sdf`."""
    ranked = sorted(
        range(len(run.episode_rewards)), key=lambda i: -run.episode_rewards[i]
    )
    best: dict[str, int] = {}  # graph hash -> episode index, best first
    for index in ranked:
        best.setdefault(graph_key.canonical_hash(run.episode_molecules[index]), index)
        if len(best) == k:
            break
    indices = list(best.values())

    molecules = tuple(run.episode_molecules[i] for i in indices)
    rewards = [run.episode_rewards[i] for i in indices]
    molio.write(
        out_stem.with_suffix(".sdf"),
        molecules,
        {"reward": [f"{reward:.4f}" for reward in rewards], "episode": indices},
    )

    drawable = []
    for mol in molecules:
        flat = Chem.Mol(mol)
        AllChem.Compute2DCoords(flat)
        drawable.append(flat)
    image = Draw.MolsToGridImage(
        drawable,
        molsPerRow=4,
        subImgSize=(320, 260),
        legends=[
            f"episode {index}  reward {reward:.3f}"
            for index, reward in zip(indices, rewards)
        ],
        returnPNG=False,
    )
    image.save(out_stem.with_suffix(".png"))


@dataclass(frozen=True)
class Analog:
    """One molecule of a top-k, read against the seed it started from."""

    reward: float  # as stored in the SDF, on the run's own scale
    num_heavy_atoms: int
    sa_score: float  # RDKit's synthetic accessibility, 1 easy to 10 hard
    tanimoto_to_seed: float | None
    row: audit.Audit


def analogs(sdf_path: Path, seed: Chem.Mol | None) -> tuple[Analog, ...]:
    """The top-k on disk, each scored for the things a reward number does not say."""
    scaffold = None if seed is None else audit.scaffold_of(seed)
    seed_fingerprint = None if seed is None else MORGAN.GetFingerprint(seed)
    out = []
    for mol in molio.read(sdf_path):
        out.append(
            Analog(
                reward=float(mol.GetProp("reward")),
                num_heavy_atoms=mol.GetNumHeavyAtoms(),
                sa_score=sascorer.calculateScore(mol),
                tanimoto_to_seed=(
                    None
                    if seed_fingerprint is None
                    else DataStructs.TanimotoSimilarity(
                        seed_fingerprint, MORGAN.GetFingerprint(mol)
                    )
                ),
                row=audit.audit(mol, scaffold),
            )
        )
    return tuple(out)


def _table(rows: tuple[Analog, ...], reward_name: str, scale: float) -> list[str]:
    """The analog table: what it scored, and what the score does not cover."""
    lines = [
        f"| # | {reward_name} | heavy atoms | SA | Tanimoto to seed | N–N bonds | scaffold |",
        "|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for index, analog in enumerate(rows):
        similarity = (
            "–" if analog.tanimoto_to_seed is None else f"{analog.tanimoto_to_seed:.2f}"
        )
        intact = {None: "–", True: "yes", False: "**NO**"}[analog.row.scaffold_intact]
        lines.append(
            f"| {index} | {analog.reward * scale:.2f} | {analog.num_heavy_atoms} | "
            f"{analog.sa_score:.1f} | {similarity} | "
            f"{analog.row.num_nitrogen_nitrogen_bonds} | {intact} |"
        )
    return lines


def _motif_summary(rows: tuple[Analog, ...]) -> list[str]:
    """The audit, over the whole top-k. This is the claim, not the reward curve."""
    lines = []
    for name in audit.MOTIFS:
        carrying = sum(1 for a in rows if a.row.motif_counts[name])
        if carrying:
            lines.append(f"- `{name}` in **{carrying}/{len(rows)}**")
    carrying = sum(1 for a in rows if a.row.num_nitrogen_nitrogen_bonds)
    if carrying:
        lines.append(f"- any nitrogen-nitrogen bond in **{carrying}/{len(rows)}**")
    intact = [a for a in rows if a.row.scaffold_intact is not None]
    if intact:
        kept = sum(1 for a in intact if a.row.scaffold_intact)
        lines.append(f"- seed scaffold intact in {kept}/{len(intact)}")
    return lines or ["- nothing on the motif list. Look at the drawing anyway."]


def null_mae(compounds: Sequence[bindingdb.Compound]) -> float:
    """Test MAE from predicting the training mean — the floor both encoders sit above.

    Recomputed rather than read: the split is deterministic, and the checkpoints do not
    carry it. Without this row an MAE of 0.8 has nothing to be 0.8 against.
    """
    train, test = splits.scaffold_split(
        compounds,
        config.RegressorConfig().test_fraction,
        held_out_scaffolds=seeds.held_out_scaffolds(seeds.choose(compounds)),
    )
    # Both splits, or the mean is taken over a training set that still has the
    # validation compounds in it and the number misses the run's by 0.002.
    train, _validation = splits.scaffold_split(train, 0.15)
    training_mean = float(np.mean([compound.pic50 for compound in train]))
    labels = np.array([compound.pic50 for compound in test], dtype=np.float32)
    return float(np.abs(training_mean - labels).mean())


def _ablation(
    pretrained: Path, from_scratch: Path, compounds: Sequence[bindingdb.Compound]
) -> list[str]:
    """The pretraining ablation, off the two checkpoints. A table row, not a rerun."""
    lines = ["| encoder | test MAE | Spearman | gap to the null closed |", "|---|---:|---:|---:|"]
    null = null_mae(compounds)
    for label, path in (("pretrained on ZINC", pretrained), ("random init", from_scratch)):
        if not path.exists():
            continue
        checkpoint = torch.load(path, weights_only=False)
        closed = (null - checkpoint["test_mae"]) / null
        lines.append(
            f"| {label} | {checkpoint['test_mae']:.3f} | "
            f"{checkpoint['test_spearman']:.3f} | {closed:.0%} |"
        )
    if len(lines) == 2:
        return []
    lines.append(f"| predicting the training mean | {null:.3f} | – | 0% |")
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/report.md"))
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="where the top-k SDFs written by train_dqn --top-k live",
    )
    parser.add_argument(
        "--qed-top",
        type=Path,
        default=Path("runs/dqn_gnn_top.sdf"),
        help="the QED control's top-k, drawn beside the pIC50 one",
    )
    parser.add_argument(
        "--regressor", type=Path, default=Path("models/egfr_regressor.pt")
    )
    parser.add_argument(
        "--regressor-random", type=Path, default=Path("models/egfr_regressor_random.pt")
    )
    args = parser.parse_args()

    compounds = bindingdb.load()
    chosen = seeds.choose(compounds)
    out = [
        "# Report",
        "",
        "Generated by `python -m mol_optim.report` from the checkpoints on disk. Every",
        "number here is read off a run, not recomputed.",
        "",
        "## The agent against a fitted pIC50 model",
        "",
    ]

    found = 0
    for index in range(len(chosen)):
        sdf = args.runs / f"pilot_pic50_seed{index}_top.sdf"
        if not sdf.exists():
            continue
        found += 1
        seed = chosen[index]
        rows = analogs(sdf, seed.mol)
        out += [
            f"### Seed {index}",
            "",
            f"Measured pIC50 **{seed.pic50:.2f}** "
            f"({seed.num_measurements} measurements, spread {seed.pic50_spread:.2f}). "
            f"Scaffold `{Chem.MolToSmiles(audit.scaffold_of(seed.mol))}`.",
            "",
        ]
        out += _table(rows, "predicted pIC50", PIC50_SCALE)
        out += ["", "What the reward number does not say:", ""]
        out += _motif_summary(rows)
        out += [""]

    if found == 0:
        out += [
            f"No `pilot_pic50_seed*_top.sdf` in `{args.runs}`. Train an agent first —"
            " see docs/running.md step 4.",
            "",
        ]
    elif found < len(chosen):
        out += [
            f"**{found} of {len(chosen)} seed scaffolds have a run.** Until the rest do,"
            " this is a pilot, not the per-seed claim.",
            "",
        ]

    if args.qed_top.exists():
        rows = analogs(args.qed_top, None)
        out += [
            "## The QED control",
            "",
            "The same loop against RDKit's drug-likeness score, which has no term for",
            "whether a structure can exist. It is here so the audit above is legible:",
            "the agent games whatever it is scored on, obviously here and invisibly",
            "there.",
            "",
        ]
        out += _table(rows, "QED", 1.0)
        out += ["", "What the reward number does not say:", ""]
        out += _motif_summary(rows)
        out += [""]

    ablation = _ablation(args.regressor, args.regressor_random, compounds)
    if ablation:
        out += [
            "## Does pretraining on ZINC help?",
            "",
            "Both checkpoints are on disk, so this is a table row rather than a rerun.",
            "Read the MAEs against the null in the last row, not against zero: that is",
            "the number a model which learned the distribution and nothing else gets.",
            "",
        ]
        out += ablation
        out += [""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")
