"""Training the pIC50 regressor on BindingDB EGFR — the whole thing, flat and in order.

    python -m mol_optim.train_regressor --checkpoint models/egfr_regressor.pt

plan.md Step 4. Reports test MAE, RMSE and Spearman on a scaffold split, against the
null of predicting the training mean.

Three splits, not two: the training half is scaffold-split again for validation, which
picks the epoch to stop at. Picking it by watching the test set is the oldest way to
report a number that does not survive new data.
"""

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from mol_optim import (
    bindingdb,
    config,
    determinism,
    featurize,
    pretrain,
    regressor,
    seeds,
    splits,
)


def train_one(
    cfg: config.Config,
    regressor_cfg: config.RegressorConfig,
    train_compounds: Sequence[bindingdb.Compound],
    validation_compounds: Sequence[bindingdb.Compound],
    seed: int,
    pretrained_encoder: Path | None,
    report_every: int = 0,
) -> tuple[regressor.Regressor, float, int]:
    """One network. Returns it at its best validation epoch, with that MAE and epoch."""
    determinism.seed_everything(seed)
    model = regressor.Regressor(cfg)
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrain.load_encoder(pretrained_encoder, cfg))
    optimizer = torch.optim.Adam(model.parameters(), lr=regressor_cfg.learning_rate)

    train_mols = [compound.mol for compound in train_compounds]
    labels = torch.tensor(
        [compound.pic50 for compound in train_compounds], dtype=torch.float32
    )  # [num_train]
    validation_mols = [compound.mol for compound in validation_compounds]
    validation_labels = np.array(
        [compound.pic50 for compound in validation_compounds], dtype=np.float32
    )  # [num_validation]

    rng = np.random.default_rng(seed)
    epoch_order = np.arange(len(train_mols))
    best_mae, best_epoch, best_state = float("inf"), -1, model.state_dict()

    for epoch in range(regressor_cfg.epochs):
        model.train()
        rng.shuffle(epoch_order)
        for start in range(
            0, len(epoch_order) - regressor_cfg.batch_size + 1, regressor_cfg.batch_size
        ):
            rows = epoch_order[start : start + regressor_cfg.batch_size]
            batch = featurize.tensors(
                featurize.graphs([train_mols[i] for i in rows]), 0.0, cfg
            )
            loss = ((model(batch) - labels[rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), regressor_cfg.grad_clip_norm)
            optimizer.step()

        predicted = regressor.predict([model], validation_mols, cfg).mean
        mae = float(np.abs(predicted - validation_labels).mean())
        if mae < best_mae:
            # In memory, not on disk: the run writes one checkpoint at the end.
            best_mae, best_epoch = mae, epoch
            best_state = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }
        if report_every and (epoch + 1) % report_every == 0:
            print(
                f"  epoch {epoch + 1:3d}  validation MAE {mae:.4f}  "
                f"best {best_mae:.4f} at {best_epoch + 1}",
                flush=True,
            )

    model.load_state_dict(best_state)
    return model, best_mae, best_epoch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        help="the ZINC checkpoint from Step 3b; omitted means random init, the null",
    )
    parser.add_argument("--report-every", type=int, default=0)
    args = parser.parse_args()

    cfg = config.Config(seed=args.seed)
    regressor_cfg = config.RegressorConfig(
        seed=args.seed, epochs=args.epochs, ensemble_size=args.ensemble
    )
    compounds = bindingdb.load()
    # Held out, not optionally: a regressor trained on a seed's series already knows the
    # answer where the run begins. To test the alternative, drop the argument.
    chosen_seeds = seeds.choose(compounds)
    train_compounds, test_compounds = splits.scaffold_split(
        compounds,
        regressor_cfg.test_fraction,
        held_out_scaffolds=seeds.held_out_scaffolds(chosen_seeds),
    )
    # By scaffold too, or the stopping epoch is chosen on molecules it has seen.
    train_compounds, validation_compounds = splits.scaffold_split(train_compounds, 0.15)
    print(
        f"{len(compounds)} compounds — {len(train_compounds)} train, "
        f"{len(validation_compounds)} validation, {len(test_compounds)} test, "
        f"{len(chosen_seeds)} seed scaffolds held out of training"
    )

    started = time.perf_counter()
    models, validation_maes = [], []
    for member in range(regressor_cfg.ensemble_size):
        model, mae, epoch = train_one(
            cfg,
            regressor_cfg,
            train_compounds,
            validation_compounds,
            seed=regressor_cfg.seed + member,
            pretrained_encoder=args.pretrained_encoder,
            report_every=args.report_every,
        )
        models.append(model)
        validation_maes.append(mae)
        print(
            f"member {member}: validation MAE {mae:.4f}, best at epoch {epoch + 1}, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )

    test_mols = [compound.mol for compound in test_compounds]
    test_labels = np.array(
        [compound.pic50 for compound in test_compounds], dtype=np.float32
    )
    prediction = regressor.predict(models, test_mols, cfg)
    error = prediction.mean - test_labels
    single = [
        float(np.abs(regressor.predict([model], test_mols, cfg).mean - test_labels).mean())
        for model in models
    ]
    # The null: a regressor that cannot beat this learned the distribution and nothing else.
    training_mean = float(np.mean([compound.pic50 for compound in train_compounds]))

    print(
        f"\ntest MAE {np.abs(error).mean():.4f}  RMSE {np.sqrt((error ** 2).mean()):.4f}  "
        f"Spearman {regressor.spearman(prediction.mean, test_labels):.4f}\n"
        f"single models MAE {np.mean(single):.4f} +- {np.std(single):.4f}  "
        f"ensemble spread {prediction.spread.mean():.4f}\n"
        f"predicting the training mean ({training_mean:.2f}) gives MAE "
        f"{np.abs(training_mean - test_labels).mean():.4f}\n"
        f"in {time.perf_counter() - started:.0f}s"
    )

    if args.checkpoint is not None:
        torch.save(
            {
                "models": [model.state_dict() for model in models],
                "config": cfg,
                "regressor_config": regressor_cfg,
                "featurization": featurize.signature(),
                "pretrained_encoder": str(args.pretrained_encoder),
                # Step 5's applicability domain needs to know what this was fitted on.
                "train_keys": [
                    compound.mol.GetProp("_Name") for compound in train_compounds
                ],
                "seed_keys": [seed.mol.GetProp("_Name") for seed in chosen_seeds],
                "test_mae": float(np.abs(error).mean()),
                "test_spearman": regressor.spearman(prediction.mean, test_labels),
            },
            args.checkpoint,
        )
        print(f"wrote {args.checkpoint}")
