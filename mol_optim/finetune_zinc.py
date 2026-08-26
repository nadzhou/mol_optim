"""Does the pretrained encoder help when it is fine-tuned rather than frozen?

    python -m mol_optim.finetune_zinc --target logp

The frozen probe in `pretrain.logp_probe` measures what a pooled embedding hands a
linear map, and loses on six of seven properties. This measures the question that
matters — is the checkpoint a better place to start gradient descent — with the pIC50
regressor's shape in miniature: two runs identical but the encoder's starting weights.

A proxy: ZINC molecules, RDKit labels, no assay noise. Step 4 ran the real comparison on
BindingDB pIC50 and it came out the same way.
"""

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Crippen, QED

from mol_optim import config, determinism, encoder, featurize, pretrain, zinc

TARGETS = {"logp": Crippen.MolLogP, "qed": QED.qed}


class Regressor(nn.Module):
    """The shared encoder plus a small head — one property per molecule."""

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(cfg.hidden_dim, cfg.num_message_passing_layers)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, batch: featurize.Batch) -> torch.Tensor:
        return self.head(self.encoder(batch)).squeeze(-1)  # [num_graphs]


def finetune(
    cfg: config.Config,
    train_molecules: Sequence[Chem.Mol],
    test_molecules: Sequence[Chem.Mol],
    label_fn: Callable[[Chem.Mol], float],
    seed: int,
    epochs: int,
    pretrained_encoder: Path | None,
) -> tuple[float, float]:
    """Test MAE and R^2 after fine-tuning, from the checkpoint or from random init.

    Seeded before the network is built, so the head is identical either way and the
    encoder's initialization is the only difference between the two runs.
    """
    determinism.seed_everything(seed)
    model = Regressor(cfg)
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrain.load_encoder(pretrained_encoder, cfg))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_labels = torch.tensor(
        [label_fn(mol) for mol in train_molecules], dtype=torch.float32
    )  # [num_train]
    rng = np.random.default_rng(seed)
    epoch_order = np.arange(len(train_molecules))

    for _ in range(epochs):
        rng.shuffle(epoch_order)
        for start in range(0, len(epoch_order) - 127, 128):
            rows = epoch_order[start : start + 128]
            batch = featurize.tensors(
                featurize.graphs([train_molecules[i] for i in rows]), 0.0, cfg
            )
            loss = ((model(batch) - train_labels[rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_labels = np.array(
        [label_fn(mol) for mol in test_molecules], dtype=np.float32
    )  # [num_test]
    with torch.no_grad():
        predicted = np.concatenate(
            [
                model(
                    featurize.tensors(
                        featurize.graphs(test_molecules[start : start + 256]), 0.0, cfg
                    )
                ).numpy()
                for start in range(0, len(test_molecules), 256)
            ]
        )  # [num_test]
    residual = ((test_labels - predicted) ** 2).sum()
    total = ((test_labels - test_labels.mean()) ** 2).sum()
    return float(np.abs(predicted - test_labels).mean()), float(1.0 - residual / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=tuple(TARGETS), default="logp")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train", type=int, default=3500, help="the rest are test")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/zinc_encoder.pt"))
    args = parser.parse_args()

    cfg = config.Config()
    pretrain_cfg = config.PretrainConfig()
    # The pretraining run's own held-out molecules, reproduced from the seed: training
    # on molecules it was pretrained on would measure memorization, invisibly.
    _, holdout = pretrain.holdout_split(
        zinc.molecules(), pretrain_cfg, np.random.default_rng(pretrain_cfg.seed)
    )
    train_molecules, test_molecules = holdout[: args.train], holdout[args.train :]
    print(
        f"{args.target}: {len(train_molecules)} molecules to fine-tune on, "
        f"{len(test_molecules)} to score, {args.epochs} epochs"
    )

    for seed in args.seeds:
        pretrained = finetune(
            cfg, train_molecules, test_molecules, TARGETS[args.target], seed,
            args.epochs, args.checkpoint,
        )
        random_init = finetune(
            cfg, train_molecules, test_molecules, TARGETS[args.target], seed,
            args.epochs, None,
        )
        print(
            f"seed {seed}  pretrained MAE {pretrained[0]:.4f} R^2 {pretrained[1]:.3f}   "
            f"random init MAE {random_init[0]:.4f} R^2 {random_init[1]:.3f}",
            flush=True,
        )
