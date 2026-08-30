import matplotlib
import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

matplotlib.use("Agg")  # no display on this machine; write files only
import matplotlib.pyplot as plt

from mol_optim import config
from mol_optim.chem import seeds, splits
from mol_optim.datasets import bindingdb
from mol_optim.nets import regressor

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def run(settings: config.Settings, spec: config.PlotSpec) -> None:
    checkpoint = torch.load(spec.inputs[0], weights_only=False)
    cfg = checkpoint["config"]
    models = []
    for state in checkpoint["models"]:
        model = regressor.Regressor(cfg)
        model.load_state_dict(state)
        models.append(model)

    compounds = bindingdb.load(settings.bindingdb.path)
    train_keys = set(checkpoint["train_keys"])
    # Checked against the keys the run recorded: the wrong test set plots as training data.
    train_compounds, test_compounds = splits.scaffold_split(
        compounds,
        checkpoint["regressor_config"].test_fraction,
        held_out_scaffolds=seeds.held_out_scaffolds(seeds.choose(compounds)),
    )
    leaked = [c for c in test_compounds if c.mol.GetProp("_Name") in train_keys]
    if leaked:
        raise ValueError(f"{len(leaked)} test compounds are in the checkpoint's train_keys")

    measured = np.array([c.pic50 for c in test_compounds], dtype=np.float32)
    prediction = regressor.predict(models, [c.mol for c in test_compounds], cfg)
    error = np.abs(prediction.mean - measured)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter_axes, calibration_axes, spread_axes, domain_axes = axes.flatten()

    scatter_axes.scatter(measured, prediction.mean, s=6, alpha=0.25, edgecolors="none")
    limits = (measured.min() - 0.3, measured.max() + 0.3)
    scatter_axes.plot(limits, limits, color="0.35", ls="--", lw=1, label="perfect")
    scatter_axes.set_xlim(limits)
    scatter_axes.set_ylim(limits)
    scatter_axes.set_xlabel("measured pIC50")
    scatter_axes.set_ylabel("predicted pIC50 (ensemble mean)")
    scatter_axes.set_title(
        f"{len(test_compounds)} held-out compounds, MAE {error.mean():.2f}, "
        f"Spearman {regressor.spearman(prediction.mean, measured):.2f}"
    )
    # The molecules the RL run starts from.
    chosen_seeds = seeds.choose(compounds)
    seed_predicted = regressor.predict(models, [s.mol for s in chosen_seeds], cfg).mean
    scatter_axes.scatter(
        [s.pic50 for s in chosen_seeds],
        seed_predicted,
        s=90,
        color="tab:red",
        marker="D",
        edgecolors="white",
        zorder=3,
        label="RL seed molecules",
    )
    ceiling = prediction.mean.max()
    scatter_axes.axhline(ceiling, color="tab:red", ls=":", lw=1)
    scatter_axes.text(
        limits[0] + 0.2,
        ceiling + 0.12,
        f"highest prediction anywhere: {ceiling:.1f}",
        color="tab:red",
        fontsize=9,
    )
    scatter_axes.legend(loc="upper left", fontsize=9)
    scatter_axes.grid(alpha=0.25)

    # Calibration by decile. A slope under 1 caps the reward an agent can be paid.
    measured_order = np.argsort(measured)
    measured_deciles = np.array_split(measured_order, 10)
    decile_measured = [measured[part].mean() for part in measured_deciles]
    decile_predicted = [prediction.mean[part].mean() for part in measured_deciles]
    calibration_axes.plot(limits, limits, color="0.35", ls="--", lw=1, label="perfect")
    calibration_axes.plot(
        decile_measured, decile_predicted, marker="o", color="tab:green", label="measured"
    )
    slope = np.polyfit(decile_measured, decile_predicted, 1)[0]
    calibration_axes.set_xlabel("measured pIC50 (decile mean)")
    calibration_axes.set_ylabel("predicted pIC50 (decile mean)")
    calibration_axes.set_title(f"calibration: slope {slope:.2f}, not 1.00")
    calibration_axes.set_xlim(limits)
    calibration_axes.set_ylim(limits)
    calibration_axes.legend(loc="upper left", fontsize=9)
    calibration_axes.grid(alpha=0.25)

    order = np.argsort(prediction.spread)
    deciles = np.array_split(order, 10)
    spread_axes.plot(
        [prediction.spread[part].mean() for part in deciles],
        [error[part].mean() for part in deciles],
        marker="o",
    )
    spread_axes.set_xlabel("ensemble disagreement (standard deviation, pIC50)")
    spread_axes.set_ylabel("mean absolute error in that decile")
    spread_correlation = regressor.spearman(prediction.spread, error)
    spread_axes.set_title(f"disagreement vs error: rank correlation {spread_correlation:.2f}")
    spread_axes.grid(alpha=0.25)

    train_fingerprints = [MORGAN.GetFingerprint(c.mol) for c in train_compounds]
    nearest = np.array(
        [
            max(
                DataStructs.BulkTanimotoSimilarity(
                    MORGAN.GetFingerprint(c.mol), train_fingerprints
                )
            )
            for c in test_compounds
        ]
    )  # [num_test]
    domain_order = np.argsort(nearest)
    domain_deciles = np.array_split(domain_order, 10)
    domain_axes.plot(
        [nearest[part].mean() for part in domain_deciles],
        [error[part].mean() for part in domain_deciles],
        marker="o",
        color="tab:orange",
    )
    domain_axes.set_xlabel("nearest training compound (Tanimoto)")
    domain_axes.set_ylabel("mean absolute error in that decile")
    domain_correlation = regressor.spearman(nearest, error)
    domain_axes.set_title(f"similarity vs error: rank correlation {domain_correlation:.2f}")
    domain_axes.grid(alpha=0.25)

    figure.tight_layout()
    spec.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(spec.out, dpi=150)
    print(
        f"test MAE {error.mean():.4f}  "
        f"Spearman {regressor.spearman(prediction.mean, measured):.4f}\n"
        f"ensemble disagreement against error: rank correlation {spread_correlation:.3f} "
        f"(top decile MAE {error[deciles[-1]].mean():.3f} against "
        f"{error[deciles[0]].mean():.3f} in the bottom)\n"
        f"nearest training compound against error: rank correlation "
        f"{domain_correlation:.3f} (least similar decile MAE "
        f"{error[domain_deciles[0]].mean():.3f} at Tanimoto "
        f"{nearest[domain_deciles[0]].mean():.2f}, most similar "
        f"{error[domain_deciles[-1]].mean():.3f} at "
        f"{nearest[domain_deciles[-1]].mean():.2f})\n"
        f"calibration slope {slope:.2f}, highest prediction {ceiling:.2f} against a "
        f"measured maximum of {measured.max():.2f}\n"
        f"wrote {spec.out}"
    )
