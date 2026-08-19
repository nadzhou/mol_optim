"""AttrMask pretraining on ZINC: mask atoms, name them from the graph around them.

plan.md Step 3b. One checkpoint initializes both the RL encoder (Step 2) and the pIC50
regressor (Step 4), so what the encoder learns here about local chemical context is
what both start from instead of inferring it from 5-10k labelled points.

The mask is an all-zero atom feature row. That choice is the whole design:

- It carries nothing. No element, no charge, no hydrogen count, no ring flags. The AttrMask
  bug is a mask the head can see through, and it does not look like a bug — the loss
  falls, the curve is beautiful, and the encoder has learned to copy one input column.
- It does not widen the input. A dedicated mask *column* would make the pretrained
  encoder a different shape from the RL encoder, and the checkpoint would not load.

What survives masking is the atom's own bond count and the bonds' own features, because
those live on the edges. That is the task working as intended — degree is context — and
it is why the shuffled-context control below does not fall all the way to the prior.
"""

import dataclasses
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Crippen

from mol_optim import config, determinism, encoder, featurize

NUM_ELEMENTS = len(featurize.ATOM_TYPES) + 1  # the featurization's "other" bucket included


class MaskedAtomPredictor(nn.Module):
    """The pretraining network: the shared GNN encoder, plus one linear element head.

    The head is deliberately one layer. Anything deeper can name a masked atom from a
    representation the encoder did not have to make chemical sense of, and it is the
    encoder that gets kept.
    """

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.encoder = encoder.GraphEncoder(cfg.hidden_dim, cfg.num_message_passing_layers)
        self.element_head = nn.Linear(cfg.hidden_dim, NUM_ELEMENTS)

    def forward(self, batch: featurize.Batch, rows: np.ndarray) -> torch.Tensor:
        h = self.encoder.node_embeddings(batch)  # [total_atoms, hidden]
        return self.element_head(h[torch.from_numpy(rows)])  # [len(rows), NUM_ELEMENTS]


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One evaluation pass over masked atoms."""

    loss: float  # mean cross-entropy, nats
    accuracy: float  # fraction whose element is the argmax


@dataclasses.dataclass(frozen=True)
class Result:
    """What a pretraining run returns. One entry per epoch, plus the two references.

    The trained network comes back with the numbers because the logP probe and the
    shuffled-context control both read it, and neither should have to go through a
    checkpoint file to do so.
    """

    model: MaskedAtomPredictor
    holdout_molecules: tuple[Chem.Mol, ...]  # what the numbers below were measured on
    train_losses: tuple[float, ...]
    holdout: tuple[Measurement, ...]  # real graphs
    control: tuple[Measurement, ...]  # same graphs, atom features shuffled
    prior: Measurement  # the model that never looks at the graph
    seconds: float


def masked(
    graph_set: featurize.Graphs,
    fraction: float,
    rng: np.random.Generator,
    cfg: config.Config,
) -> tuple[featurize.Batch, np.ndarray]:
    """The batch the encoder reads, with `fraction` of atom feature rows zeroed.

    Steps remaining is 0.0 because there is no episode here. It reaches the DQN head
    and never the encoder, and the encoder is all this trains.
    """
    num_atoms = len(graph_set.atom_codes)
    rows = rng.choice(num_atoms, size=max(1, round(fraction * num_atoms)), replace=False)
    batch = featurize.tensors(graph_set, 0.0, cfg)
    # featurize.tensors builds its arrays fresh, so zeroing in place here cannot reach
    # the caller's graph_set or any batch built from it earlier.
    batch.atom_features[rows] = 0.0
    return batch, rows


def with_shuffled_atoms(
    graph_set: featurize.Graphs, rng: np.random.Generator
) -> featurize.Graphs:
    """The control: the same bonds, every atom's features dealt to a random position.

    Structure survives, chemistry does not. A model that reads the neighbourhood loses
    what the neighbourhood was telling it, so held-out loss on these graphs is the
    number that says the task depends on context at all. It does not reach the prior:
    an atom's bond count and its bonds' types still arrive through its edges.
    """
    permutation = rng.permutation(len(graph_set.atom_codes))
    return dataclasses.replace(graph_set, atom_codes=graph_set.atom_codes[permutation])


def measure(
    model: MaskedAtomPredictor,
    truth_sets: Sequence[featurize.Graphs],
    context_sets: Sequence[featurize.Graphs],
    pretrain_cfg: config.PretrainConfig,
    cfg: config.Config,
) -> Measurement:
    """Masked-element loss and accuracy, no gradient.

    `truth_sets` carries the answers and `context_sets` the features the encoder reads.
    They are the same list for the ordinary measurement; the control passes
    with_shuffled_atoms(...) as the context. Each batch's rng is seeded from its
    position, so both measurements mask the same atoms and the two numbers compare atom
    for atom.
    """
    total_loss, total_correct, total_masked = 0.0, 0, 0
    for index, (truth, context) in enumerate(zip(truth_sets, context_sets)):
        batch, rows = masked(
            context,
            pretrain_cfg.mask_fraction,
            np.random.default_rng([pretrain_cfg.seed, index]),
            cfg,
        )
        elements = truth.atom_codes[rows, 0].astype(np.int64)
        targets = torch.from_numpy(elements)  # [num_masked]
        with torch.no_grad():
            logits = model(batch, rows)  # [num_masked, NUM_ELEMENTS]
            total_loss += float(
                nn.functional.cross_entropy(logits, targets, reduction="sum")
            )
            total_correct += int((logits.argmax(dim=-1) == targets).sum())
        total_masked += len(rows)
    return Measurement(total_loss / total_masked, total_correct / total_masked)


def marginal(train_codes: np.ndarray, holdout_codes: np.ndarray) -> Measurement:
    """The reference: guess the training set's element distribution, ignore the graph.

    ZINC is 73.6% carbon, so accuracy alone flatters any model; this is the loss a
    pretraining run has to beat to have learned anything at all. Counts start at one so
    an element that appears only in the held-out set costs a large number rather than
    an infinite one.
    """
    counts = np.bincount(train_codes[:, 0], minlength=NUM_ELEMENTS) + 1
    probabilities = counts / counts.sum()  # [NUM_ELEMENTS]
    targets = holdout_codes[:, 0]
    return Measurement(
        loss=float(-np.log(probabilities[targets]).mean()),
        accuracy=float((targets == probabilities.argmax()).mean()),
    )


def logp_probe(
    graph_encoder: encoder.GraphEncoder,
    fit_molecules: Sequence[Chem.Mol],
    test_molecules: Sequence[Chem.Mol],
    cfg: config.Config,
) -> float:
    """R^2 of a least-squares line from frozen embeddings to Crippen logP, on test.

    The cheap first evidence that pretraining did anything (plan.md Step 3b): the same
    probe on a randomly initialized encoder is the number to beat. Frozen means frozen —
    the only fitted parameters are the hidden_dim + 1 weights of the linear map.
    """

    def embeddings(mols: Sequence[Chem.Mol]) -> np.ndarray:
        with torch.no_grad():
            blocks = [
                graph_encoder(
                    featurize.tensors(featurize.graphs(mols[start : start + 256]), 0.0, cfg)
                )
                for start in range(0, len(mols), 256)
            ]
        pooled = torch.cat(blocks).numpy()  # [num_molecules, hidden]
        ones = np.ones((len(pooled), 1), dtype=np.float32)  # the intercept column
        return np.concatenate([pooled, ones], axis=1)  # [num_molecules, hidden + 1]

    fit_logp = np.array(
        [Crippen.MolLogP(mol) for mol in fit_molecules], dtype=np.float32
    )  # [num_fit]
    test_logp = np.array(
        [Crippen.MolLogP(mol) for mol in test_molecules], dtype=np.float32
    )  # [num_test]
    weights, *_ = np.linalg.lstsq(embeddings(fit_molecules), fit_logp, rcond=None)
    predicted = embeddings(test_molecules) @ weights  # [num_test]
    residual = ((test_logp - predicted) ** 2).sum()
    total = ((test_logp - test_logp.mean()) ** 2).sum()
    return float(1.0 - residual / total)


def save_encoder(
    path: Path,
    model: MaskedAtomPredictor,
    cfg: config.Config,
    pretrain_cfg: config.PretrainConfig,
    holdout: Measurement,
) -> None:
    """Writes the encoder with everything needed to refuse a wrong load later."""
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "element_head": model.element_head.state_dict(),
            "config": cfg,
            "pretrain_config": pretrain_cfg,
            "featurization": featurize.signature(),
            "holdout_loss": holdout.loss,
            "holdout_accuracy": holdout.accuracy,
        },
        path,
    )


def load_encoder(path: Path, cfg: config.Config) -> dict[str, torch.Tensor]:
    """The pretrained encoder weights, or a loud failure.

    A checkpoint that silently does not load is the most common reason pretraining
    "doesn't help", and a checkpoint that loads into a different featurization is worse:
    every weight lands on a column that means something else. Both are refused here,
    which is why the featurization hash is written into the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with: python -m mol_optim.pretrain"
        )
    # weights_only=False: this file holds our own Config dataclasses, and it is a file
    # this repo wrote, not one off the internet.
    checkpoint = torch.load(path, weights_only=False)
    if checkpoint["featurization"] != featurize.signature():
        raise ValueError(
            f"{path} was pretrained on featurization {checkpoint['featurization']}, and "
            f"featurize.py is now {featurize.signature()}. The columns mean different "
            "things; re-run python -m mol_optim.pretrain."
        )
    saved = checkpoint["config"]
    shape = (saved.hidden_dim, saved.num_message_passing_layers)
    wanted = (cfg.hidden_dim, cfg.num_message_passing_layers)
    if shape != wanted:
        raise ValueError(
            f"{path} holds a {shape[0]}-wide, {shape[1]}-layer encoder; this config asks "
            f"for {wanted[0]}-wide, {wanted[1]}-layer."
        )
    return checkpoint["encoder"]


def pretrain(
    cfg: config.Config,
    pretrain_cfg: config.PretrainConfig,
    molecules: Sequence[Chem.Mol],
    log_path: Path | None = None,
    checkpoint_path: Path | None = None,
    report_every: int = 0,
) -> Result:
    """Trains MaskedAtomPredictor on ZINC and returns every number the run produced."""
    determinism.seed_everything(pretrain_cfg.seed)
    rng = np.random.default_rng(pretrain_cfg.seed)

    # A seeded shuffle, not a cut of file order: the file's order is arbitrary but it is
    # not random, and a held-out block from one end could be a block of one supplier.
    order = rng.permutation(len(molecules))
    holdout_index = order[: pretrain_cfg.num_holdout]
    train_index = order[pretrain_cfg.num_holdout :]
    if len(train_index) == 0:
        raise ValueError(
            f"{len(molecules)} molecules and num_holdout={pretrain_cfg.num_holdout} "
            "leaves nothing to train on"
        )

    # The held-out set is measured every epoch, so it is featurized once. The training
    # molecules are featurized per batch: 0.13 ms each against a 23 ms training step,
    # and it keeps one array per batch alive instead of one per molecule.
    holdout_sets = [
        featurize.graphs(
            [molecules[i] for i in holdout_index[start : start + pretrain_cfg.batch_size]]
        )
        for start in range(0, len(holdout_index), pretrain_cfg.batch_size)
    ]
    control_sets = [with_shuffled_atoms(graph_set, rng) for graph_set in holdout_sets]
    # The element histogram is read off 5000 training molecules rather than all of them:
    # that is 115k atoms, which pins carbon, nitrogen and oxygen — the three that carry
    # almost all of the loss — to better than a tenth of a percent, and featurizing 249k
    # molecules here would cost more than the first epoch.
    prior = marginal(
        featurize.graphs([molecules[i] for i in train_index[:5000]]).atom_codes,
        np.concatenate([graph_set.atom_codes for graph_set in holdout_sets]),
    )

    model = MaskedAtomPredictor(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_cfg.learning_rate)

    log_file = open(log_path, "w") if log_path is not None else None
    if log_file is not None:
        # The prior repeats on every row rather than living in a header comment: it is
        # the line every other number in the file is read against, and a plot that has
        # to be told the baseline by hand is a plot that can be told the wrong one.
        log_file.write(
            "epoch,train_loss,holdout_loss,holdout_accuracy,control_loss,"
            "control_accuracy,prior_loss,prior_accuracy\n"
        )

    train_losses: list[float] = []
    holdout_measurements: list[Measurement] = []
    control_measurements: list[Measurement] = []
    started = time.perf_counter()

    for epoch in range(pretrain_cfg.epochs):
        rng.shuffle(train_index)
        batch_losses: list[float] = []

        # A short final batch is dropped: the mask fraction is a rounding of the batch's
        # atom count, and a tenth-size batch would carry a tenth-size vote on the epoch.
        last_start = len(train_index) - pretrain_cfg.batch_size
        for start in range(0, last_start + 1, pretrain_cfg.batch_size):
            graph_set = featurize.graphs(
                [molecules[i] for i in train_index[start : start + pretrain_cfg.batch_size]]
            )
            batch, rows = masked(graph_set, pretrain_cfg.mask_fraction, rng, cfg)
            elements = graph_set.atom_codes[rows, 0].astype(np.int64)
            targets = torch.from_numpy(elements)  # [num_masked]

            loss = nn.functional.cross_entropy(model(batch, rows), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), pretrain_cfg.grad_clip_norm
            )
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        train_loss = sum(batch_losses) / len(batch_losses)
        holdout = measure(model, holdout_sets, holdout_sets, pretrain_cfg, cfg)
        control = measure(model, holdout_sets, control_sets, pretrain_cfg, cfg)
        train_losses.append(train_loss)
        holdout_measurements.append(holdout)
        control_measurements.append(control)

        if log_file is not None:
            log_file.write(
                f"{epoch},{train_loss:.6f},{holdout.loss:.6f},{holdout.accuracy:.6f},"
                f"{control.loss:.6f},{control.accuracy:.6f},"
                f"{prior.loss:.6f},{prior.accuracy:.6f}\n"
            )
            log_file.flush()
        if report_every and (epoch + 1) % report_every == 0:
            print(
                f"epoch {epoch + 1:3d}  train {train_loss:.4f}  "
                f"holdout {holdout.loss:.4f} ({holdout.accuracy:.3f} correct)  "
                f"shuffled context {control.loss:.4f}  prior {prior.loss:.4f}  "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )

    if log_file is not None:
        log_file.close()
    if checkpoint_path is not None:
        save_encoder(checkpoint_path, model, cfg, pretrain_cfg, holdout_measurements[-1])

    return Result(
        model=model,
        holdout_molecules=tuple(molecules[i] for i in holdout_index),
        train_losses=tuple(train_losses),
        holdout=tuple(holdout_measurements),
        control=tuple(control_measurements),
        prior=prior,
        seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    import argparse

    from mol_optim import zinc

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--molecules", type=int, default=None, help="default: all of ZINC"
    )
    parser.add_argument("--holdout", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report-every", type=int, default=1)
    args = parser.parse_args()

    cfg = config.Config(seed=args.seed)
    pretrain_cfg = config.PretrainConfig(
        seed=args.seed,
        num_molecules=args.molecules,
        num_holdout=args.holdout,
        epochs=args.epochs,
    )
    molecules = zinc.molecules(limit=pretrain_cfg.num_molecules)
    print(f"{len(molecules)} ZINC molecules, {pretrain_cfg.num_holdout} held out")

    run = pretrain(
        cfg,
        pretrain_cfg,
        molecules,
        log_path=args.log,
        checkpoint_path=args.checkpoint,
        report_every=args.report_every,
    )
    print(
        f"prior {run.prior.loss:.4f} ({run.prior.accuracy:.3f} correct)  "
        f"holdout {run.holdout[-1].loss:.4f} ({run.holdout[-1].accuracy:.3f})  "
        f"shuffled context {run.control[-1].loss:.4f} ({run.control[-1].accuracy:.3f})  "
        f"in {run.seconds:.0f}s"
    )

    # The logP probe, on the run's own held-out molecules, so nothing the encoder
    # trained on reaches it. Fit on half, score on the other half, and run the same probe
    # on an untrained encoder for the number to beat.
    probe_molecules = run.holdout_molecules
    half = len(probe_molecules) // 2
    torch.manual_seed(pretrain_cfg.seed + 1)  # a different draw from the run's own init
    untrained = MaskedAtomPredictor(cfg)
    fit, test = probe_molecules[:half], probe_molecules[half:]
    pretrained_r2 = logp_probe(run.model.encoder, fit, test, cfg)
    random_r2 = logp_probe(untrained.encoder, fit, test, cfg)
    print(f"logP probe R^2  pretrained {pretrained_r2:.3f}  random init {random_r2:.3f}")
    if args.checkpoint is not None:
        # Read back what was just written: a checkpoint that does not load is the
        # failure this step exists to rule out, and it costs a second to rule out here.
        reloaded = MaskedAtomPredictor(cfg)
        reloaded.encoder.load_state_dict(load_encoder(args.checkpoint, cfg))
        print(f"wrote {args.checkpoint} and loaded it back")
