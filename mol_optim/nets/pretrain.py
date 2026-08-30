"""AttrMask pretraining on ZINC: mask atoms, name them from the graph around them.

One checkpoint initializes both the RL encoder and the pIC50 regressor. The mask is an
all-zero feature row: it carries nothing and does not widen the input.
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

from mol_optim import config, determinism
from mol_optim.chem import featurize
from mol_optim.datasets import zinc
from mol_optim.nets import encoder

NUM_ELEMENTS = len(featurize.ATOM_TYPES) + 1  # the featurization's "other" bucket included


class MaskedAtomPredictor(nn.Module):
    """The shared encoder plus one linear element head.

    One layer deliberately: anything deeper can name the atom from a representation the
    encoder never had to make chemical sense of, and it is the encoder that gets kept.
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

    Steps remaining is 0.0: no episode here, and it reaches the DQN head, not the encoder.
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
    """The control: same bonds, every atom's features dealt to a random position.

    Structure survives, chemistry does not, so the loss here says whether the task
    depends on context. It stays under the prior because degree still arrives by edge.
    """
    permutation = rng.permutation(len(graph_set.atom_codes))
    return dataclasses.replace(graph_set, atom_codes=graph_set.atom_codes[permutation])


def holdout_split(
    molecules: Sequence[Chem.Mol],
    pretrain_cfg: config.PretrainConfig,
    rng: np.random.Generator,
) -> tuple[tuple[Chem.Mol, ...], tuple[Chem.Mol, ...]]:
    """(training, held out), by a seeded shuffle rather than a cut of file order.

    ZINC's order is not random — a block from one end could be one supplier. Takes the
    generator rather than seeding its own, so a caller can reproduce a finished run's
    held-out molecules without re-running the pretraining.
    """
    order = rng.permutation(len(molecules))
    return (
        tuple(molecules[i] for i in order[pretrain_cfg.num_holdout :]),
        tuple(molecules[i] for i in order[: pretrain_cfg.num_holdout]),
    )


def measure(
    model: MaskedAtomPredictor,
    truth_sets: Sequence[featurize.Graphs],
    context_sets: Sequence[featurize.Graphs],
    pretrain_cfg: config.PretrainConfig,
    cfg: config.Config,
) -> Measurement:
    """Masked-element loss and accuracy, no gradient.

    `truth_sets` carries the answers, `context_sets` the features the encoder reads —
    the same list normally, shuffled context for the control. Each batch's rng is seeded
    from its position, so both mask the same atoms and the numbers compare directly.
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
    """The reference: guess the training element distribution, ignore the graph.

    ZINC is 73.6% carbon, so accuracy flatters any model; this loss is what a run has to
    beat. Counts start at one so an unseen element costs a large number, not infinity.
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

    A sanity check, not evidence — the pIC50 regressor settled the pretraining
    question. The only
    fitted parameters are the hidden_dim + 1 weights of the linear map.
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

    A checkpoint that silently does not load is the usual reason pretraining "doesn't
    help"; one that loads against another featurization is worse. Both refused here.
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

    train_molecules, holdout_molecules = holdout_split(molecules, pretrain_cfg, rng)
    if len(train_molecules) == 0:
        raise ValueError(
            f"{len(molecules)} molecules and num_holdout={pretrain_cfg.num_holdout} "
            "leaves nothing to train on"
        )

    # Held-out is featurized once, training per batch: 0.13 ms a molecule against a
    # 23 ms step, and one array alive per batch instead of one per molecule.
    holdout_sets = [
        featurize.graphs(holdout_molecules[start : start + pretrain_cfg.batch_size])
        for start in range(0, len(holdout_molecules), pretrain_cfg.batch_size)
    ]
    control_sets = [with_shuffled_atoms(graph_set, rng) for graph_set in holdout_sets]
    # 5000 molecules is 115k atoms, which pins C, N and O to a tenth of a percent;
    # featurizing all 249k here would cost more than the first epoch.
    prior = marginal(
        featurize.graphs(train_molecules[:5000]).atom_codes,
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

    # Shuffled in place, so each epoch reshuffles the last one's order.
    epoch_order = np.arange(len(train_molecules))

    for epoch in range(pretrain_cfg.epochs):
        rng.shuffle(epoch_order)
        batch_losses: list[float] = []

        # Short final batch dropped: the mask fraction rounds off the batch's atom count.
        last_start = len(train_molecules) - pretrain_cfg.batch_size
        for start in range(0, last_start + 1, pretrain_cfg.batch_size):
            graph_set = featurize.graphs(
                [
                    train_molecules[i]
                    for i in epoch_order[start : start + pretrain_cfg.batch_size]
                ]
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
        holdout_molecules=holdout_molecules,
        train_losses=tuple(train_losses),
        holdout=tuple(holdout_measurements),
        control=tuple(control_measurements),
        prior=prior,
        seconds=time.perf_counter() - started,
    )


def run(settings: config.Settings) -> None:
    spec = settings.pretrain
    cfg = config.Config(seed=spec.cfg.seed)
    molecules = zinc.molecules(settings.zinc.path, limit=spec.cfg.num_molecules)
    print(f"{len(molecules)} molecules, {spec.cfg.num_holdout} held out")

    result = pretrain(
        cfg,
        spec.cfg,
        molecules,
        log_path=spec.log,
        checkpoint_path=spec.checkpoint,
        report_every=spec.report_every,
    )
    print(
        f"prior {result.prior.loss:.4f} ({result.prior.accuracy:.3f} correct)  "
        f"holdout {result.holdout[-1].loss:.4f} ({result.holdout[-1].accuracy:.3f})  "
        f"shuffled context {result.control[-1].loss:.4f} "
        f"({result.control[-1].accuracy:.3f})  in {result.seconds:.0f}s"
    )

    # The logP probe, on the run's own held-out molecules, so nothing the encoder trained
    # on reaches it. Fit on half, score on the other half, and run the same probe on an
    # untrained encoder for the number to beat.
    probe_molecules = result.holdout_molecules
    half = len(probe_molecules) // 2
    torch.manual_seed(spec.cfg.seed + 1)  # a different draw from the run's own init
    untrained = MaskedAtomPredictor(cfg)
    fit, test = probe_molecules[:half], probe_molecules[half:]
    pretrained_r2 = logp_probe(result.model.encoder, fit, test, cfg)
    random_r2 = logp_probe(untrained.encoder, fit, test, cfg)
    print(f"logP probe R^2  pretrained {pretrained_r2:.3f}  random init {random_r2:.3f}")

    if spec.checkpoint is not None:
        # Read back what was just written: a checkpoint that does not load is the failure
        # this step exists to rule out, and it costs a second to rule out here.
        reloaded = MaskedAtomPredictor(cfg)
        reloaded.encoder.load_state_dict(load_encoder(spec.checkpoint, cfg))
        print(f"wrote {spec.checkpoint} and loaded it back")
