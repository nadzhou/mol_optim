"""A within-series ranker: what a substitution does to potency, not what a molecule binds at.

The pIC50 regressor is fitted to absolute potency and validated on MAE over a scaffold
split. Both are the wrong quantity for lead optimization, and the gap is measured: its
within-series Spearman is +0.38 on series it trained on and +0.36 on series it did not,
so it ranks close analogs barely better than chance whether or not it has seen them. An
agent rewarded by it is told that most real analogs of its seed are worse than the seed.

So this is trained on the quantity the agent needs. The network scores one molecule, and
the loss is on the *difference* between two molecules of the same scaffold series:

    loss = ((s(a) - s(b)) - (y_a - y_b)) ** 2

Two consequences of scoring molecules rather than pairs. The score is exactly
antisymmetric by construction — s(a) - s(b) = -(s(b) - s(a)) — with no need to train both
orderings or to average them at inference. And the RL loop can read s(mol) directly as a
reward, with no reference molecule to carry around, because an additive constant per
series does not change an argmax.

Only within-series pairs are used. A pair drawn from two different scaffolds teaches
absolute potency again, which is the thing that already does not work.
"""

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
from mol_optim.nets import encoder, pretrain, regressor


class Ranker(nn.Module):
    """The regressor's shape exactly, so a difference between the two is the objective."""

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


def score(
    models: Sequence[Ranker], mols: Sequence[Chem.Mol], cfg: config.Config
) -> np.ndarray:
    """Ensemble mean score, in batches of 256. Comparable within a series, not across."""
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
    return np.stack(columns).mean(axis=0)  # [num_molecules]


def within_series_spearman(
    predicted: Sequence[np.ndarray], measured: Sequence[np.ndarray]
) -> float:
    """Median rank correlation over series. The number the whole exercise turns on.

    Per series and then a median, not one correlation over the pooled compounds: pooling
    lets a model score every quinazoline above every pyrimidine and call that ranking,
    which is exactly the credit the regressor was getting.
    """
    rhos = [
        regressor.spearman(p, m)
        for p, m in zip(predicted, measured)
        if len(p) >= 5 and m.std() > 0.3
    ]
    return float(np.median(rhos)) if rhos else float("nan")


def series_of(
    compounds: Sequence[bindingdb.Compound], min_size: int
) -> list[tuple[bindingdb.Compound, ...]]:
    """Scaffold series big enough and varied enough to carry a ranking signal."""
    return [
        tuple(group)
        for group in splits.by_scaffold(compounds).values()
        if len(group) >= min_size
        and np.std([c.pic50 for c in group]) > 0.3
    ]


def train_one(
    cfg: config.Config,
    ranker_cfg: config.RankerConfig,
    train_series: Sequence[tuple[bindingdb.Compound, ...]],
    validation_series: Sequence[tuple[bindingdb.Compound, ...]],
    seed: int,
    pretrained_encoder: Path | None,
    report_every: int = 0,
) -> tuple[Ranker, float, int]:
    determinism.seed_everything(seed)
    model = Ranker(cfg)
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrain.load_encoder(pretrained_encoder, cfg))
    optimizer = torch.optim.Adam(model.parameters(), lr=ranker_cfg.learning_rate)
    rng = np.random.default_rng(seed)

    # Every within-series pair, once, as (molecule a, molecule b, y_a - y_b). Enumerated
    # up front rather than sampled per epoch so an epoch is a fixed pass over fixed data
    # and two runs at one seed are the same run.
    left, right, target = [], [], []
    for group in train_series:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if len(left) >= ranker_cfg.max_pairs:
                    break
                left.append(group[i].mol)
                right.append(group[j].mol)
                target.append(group[i].pic50 - group[j].pic50)
    delta = torch.tensor(target, dtype=torch.float32)  # [num_pairs]

    validation_mols = [[c.mol for c in g] for g in validation_series]
    validation_labels = [
        np.array([c.pic50 for c in g], dtype=np.float32) for g in validation_series
    ]
    order = np.arange(len(left))
    best_rho, best_epoch, best_state = -1.0, -1, model.state_dict()

    for epoch in range(ranker_cfg.epochs):
        model.train()
        rng.shuffle(order)
        for start in range(
            0, len(order) - ranker_cfg.batch_size + 1, ranker_cfg.batch_size
        ):
            rows = order[start : start + ranker_cfg.batch_size]
            # Both sides in one graph batch: the a's then the b's, so one forward pass
            # covers the pair and the encoder sees each molecule the same way.
            batch = featurize.tensors(
                featurize.graphs([left[i] for i in rows] + [right[i] for i in rows]),
                0.0,
                cfg,
            )
            scores = model(batch)  # [2 * batch_size]
            predicted = scores[: len(rows)] - scores[len(rows) :]  # [batch_size]
            loss = ((predicted - delta[rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), ranker_cfg.grad_clip_norm)
            optimizer.step()

        rho = within_series_spearman(
            [score([model], mols, cfg) for mols in validation_mols], validation_labels
        )
        if not np.isnan(rho) and rho > best_rho:
            best_rho, best_epoch = rho, epoch
            best_state = {n: t.clone() for n, t in model.state_dict().items()}
        if report_every and (epoch + 1) % report_every == 0:
            print(
                f"  epoch {epoch + 1:3d}  validation rho {rho:+.4f}  "
                f"best {best_rho:+.4f} at {best_epoch + 1}",
                flush=True,
            )

    if best_epoch < 0:
        raise ValueError("no epoch produced a validation rho; nothing to select on")
    model.load_state_dict(best_state)
    return model, best_rho, best_epoch


def run(settings: config.Settings) -> None:
    spec = settings.ranker
    cfg = config.Config(seed=spec.cfg.seed)
    compounds = bindingdb.load(settings.bindingdb.path)
    chosen_seeds = seeds.choose(compounds)
    train_compounds, test_compounds = splits.scaffold_split(
        compounds,
        spec.cfg.test_fraction,
        held_out_scaffolds=seeds.held_out_scaffolds(chosen_seeds),
    )
    test_series = series_of(test_compounds, spec.cfg.min_series_size)

    # Validation holds out whole *series*, not a scaffold slice of the compounds. A
    # 15 per cent split of compounds leaves each validation scaffold with one or two
    # members, none of them a series — which silently gave zero validation series, a
    # NaN every epoch, and a checkpoint of the untrained weights.
    series = series_of(train_compounds, spec.cfg.min_series_size)
    cut = max(1, round(0.15 * len(series)))
    validation_series, train_series = series[:cut], series[cut:]
    if not train_series or not validation_series:
        raise ValueError(
            f"{len(series)} series of >= {spec.cfg.min_series_size} compounds with "
            f"pIC50 spread > 0.3; not enough to split into train and validation"
        )
    print(
        f"{len(compounds)} compounds — {len(train_series)} train series, "
        f"{len(validation_series)} validation, {len(test_series)} test; "
        f"{sum(len(g) * (len(g) - 1) // 2 for g in train_series)} within-series pairs"
    )

    started = time.perf_counter()
    models = []
    for member in range(spec.cfg.ensemble_size):
        model, rho, epoch = train_one(
            cfg,
            spec.cfg,
            train_series,
            validation_series,
            seed=spec.cfg.seed + member,
            pretrained_encoder=spec.pretrained_encoder,
            report_every=spec.report_every,
        )
        models.append(model)
        print(
            f"member {member}: validation rho {rho:+.4f}, best at epoch {epoch + 1}, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )

    test_mols = [[c.mol for c in g] for g in test_series]
    test_labels = [np.array([c.pic50 for c in g], dtype=np.float32) for g in test_series]
    rho = within_series_spearman(
        [score(models, mols, cfg) for mols in test_mols], test_labels
    )
    # The seeds' own series, named one by one. These are held out of training entirely,
    # and they are the series the RL reward has to rank — the +0.23 the regressor scored
    # on seed 0's analogs is what this line is compared against.
    by_scaffold = splits.by_scaffold(compounds)
    seed_rhos = []
    for index, chosen in enumerate(chosen_seeds):
        group = by_scaffold[chosen.scaffold]
        if len(group) < 5:
            continue
        measured = np.array([c.pic50 for c in group], dtype=np.float32)
        predicted = score(models, [c.mol for c in group], cfg)
        seed_rhos.append((index, len(group), regressor.spearman(predicted, measured)))

    print(
        f"\ntest within-series Spearman {rho:+.4f} over {len(test_series)} series\n"
        f"in {time.perf_counter() - started:.0f}s\n"
    )
    print(f"{'seed':>5} {'series size':>12} {'Spearman':>10}")
    for index, size, seed_rho in seed_rhos:
        print(f"{index:>5} {size:>12} {seed_rho:>+10.4f}")

    if spec.checkpoint is not None:
        spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "models": [m.state_dict() for m in models],
                "config": cfg,
                "ranker_config": spec.cfg,
                "featurization": featurize.signature(),
                "train_keys": [c.mol.GetProp("_Name") for c in train_compounds],
                "seed_keys": [s.mol.GetProp("_Name") for s in chosen_seeds],
                "test_within_series_spearman": rho,
            },
            spec.checkpoint,
        )
        print(f"wrote {spec.checkpoint}")
