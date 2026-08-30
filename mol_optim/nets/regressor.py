import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from mol_optim import config, determinism
from mol_optim.chem import featurize, seeds, splits
from mol_optim.datasets import bindingdb
from mol_optim.nets import encoder, pretrain


class Regressor(nn.Module):
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(cfg.hidden_dim, cfg.num_message_passing_layers)
        self.linear_1 = nn.Linear(cfg.hidden_dim, 128)
        self.linear_2 = nn.Linear(128, 32)
        self.linear_3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, batch: featurize.Batch) -> torch.Tensor:
        x = self.activation(self.linear_1(self.encoder(batch)))  # [num_graphs, 128]
        x = self.activation(self.linear_2(x))  # [num_graphs, 32]
        return self.linear_3(x).squeeze(-1)  # [num_graphs]


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray  # [num_molecules] predicted pIC50
    spread: np.ndarray  # [num_molecules] standard deviation across the ensemble


def predict(
    models: Sequence[Regressor], mols: Sequence[Chem.Mol], cfg: config.Config
) -> Prediction:
    """Ensemble mean and standard deviation, in batches of 256.

    Steps remaining is 0.0: there is no episode here, and this head never reads it.
    """
    columns = []
    for model in models:
        model.eval()
        with torch.no_grad():
            columns.append(
                torch.cat(
                    [
                        model(
                            featurize.tensors(
                                featurize.graphs(mols[start : start + 256]), 0.0, cfg
                            )
                        )
                        for start in range(0, len(mols), 256)
                    ]
                ).numpy()
            )
    stacked = np.stack(columns)  # [num_models, num_molecules]
    return Prediction(mean=stacked.mean(axis=0), spread=stacked.std(axis=0))


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation: is the compound the model likes best the one that binds best?

    The only thing the RL agent uses the regressor for. MAE measures something else.
    """
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — duplicated pIC50 values are common at round numbers."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative.

    The rank-sum identity, so no scikit-learn. Ties count a half, which `_ranks` handles.
    """
    pooled = np.concatenate([positive, negative])
    ranks = _ranks(pooled) + 1.0  # [len(pooled)], 1-based
    rank_sum = ranks[: len(positive)].sum()
    best_case = len(positive) * (len(positive) + 1) / 2.0
    return float((rank_sum - best_case) / (len(positive) * len(negative)))


def train_one(
    cfg: config.Config,
    regressor_cfg: config.RegressorConfig,
    train_compounds: Sequence[bindingdb.Compound],
    validation_compounds: Sequence[bindingdb.Compound],
    seed: int,
    pretrained_encoder: Path | None,
    report_every: int = 0,
) -> tuple[Regressor, float, int]:
    """One network. Returns it at its best validation epoch, with that MAE and epoch."""
    determinism.seed_everything(seed)
    model = Regressor(cfg)
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

        predicted = predict([model], validation_mols, cfg).mean
        mae = float(np.abs(predicted - validation_labels).mean())
        if mae < best_mae:
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


def run(settings: config.Settings) -> None:
    spec = settings.regressor
    cfg = config.Config(seed=spec.cfg.seed)
    compounds = bindingdb.load(settings.bindingdb.path)
    # Held out, not optionally: otherwise the reward knows the answer where runs begin.
    chosen_seeds = seeds.choose(compounds)
    train_compounds, test_compounds = splits.scaffold_split(
        compounds,
        spec.cfg.test_fraction,
        held_out_scaffolds=seeds.held_out_scaffolds(chosen_seeds),
    )
    train_compounds, validation_compounds = splits.scaffold_split(train_compounds, 0.15)
    print(
        f"{len(compounds)} compounds — {len(train_compounds)} train, "
        f"{len(validation_compounds)} validation, {len(test_compounds)} test, "
        f"{len(chosen_seeds)} seed scaffolds held out of training"
    )

    started = time.perf_counter()
    models, validation_maes = [], []
    for member in range(spec.cfg.ensemble_size):
        model, mae, epoch = train_one(
            cfg,
            spec.cfg,
            train_compounds,
            validation_compounds,
            seed=spec.cfg.seed + member,
            pretrained_encoder=spec.pretrained_encoder,
            report_every=spec.report_every,
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
    prediction = predict(models, test_mols, cfg)
    error = prediction.mean - test_labels
    single = [
        float(np.abs(predict([model], test_mols, cfg).mean - test_labels).mean())
        for model in models
    ]
    # The null: a regressor that cannot beat this learned the distribution and nothing else.
    training_mean = float(np.mean([compound.pic50 for compound in train_compounds]))

    print(
        f"\ntest MAE {np.abs(error).mean():.4f}  RMSE {np.sqrt((error ** 2).mean()):.4f}  "
        f"Spearman {spearman(prediction.mean, test_labels):.4f}\n"
        f"single models MAE {np.mean(single):.4f} +- {np.std(single):.4f}  "
        f"ensemble spread {prediction.spread.mean():.4f}\n"
        f"predicting the training mean ({training_mean:.2f}) gives MAE "
        f"{np.abs(training_mean - test_labels).mean():.4f}\n"
        f"in {time.perf_counter() - started:.0f}s"
    )

    if spec.checkpoint is not None:
        spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": [model.state_dict() for model in models],
                "config": cfg,
                "regressor_config": spec.cfg,
                "featurization": featurize.signature(),
                "pretrained_encoder": str(spec.pretrained_encoder),
                "train_keys": [
                    compound.mol.GetProp("_Name") for compound in train_compounds
                ],
                "seed_keys": [seed.mol.GetProp("_Name") for seed in chosen_seeds],
                "test_mae": float(np.abs(error).mean()),
                "test_spearman": spearman(prediction.mean, test_labels),
            },
            spec.checkpoint,
        )
        print(f"wrote {spec.checkpoint}")
