"""The pIC50 regressor.

The overfit test is the one to keep: it takes seconds and it fails loudly for every
broken forward pass, dead gradient and mis-shaped label there is. The random-label
control is the one that catches a leaking split, which no amount of staring at a test
MAE will do.
"""

import dataclasses

import numpy as np
import pytest
import torch

from mol_optim import (
    config,
    determinism,
    featurize,
    regressor,
    splits,
    train_regressor,
)

CFG = config.Config()


def test_spearman_is_rank_correlation_and_averages_ties():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert regressor.spearman(values, values) == pytest.approx(1.0)
    assert regressor.spearman(values, -values) == pytest.approx(-1.0)
    # Perfect correlation on ranks, not on values: a monotone squash changes nothing.
    assert regressor.spearman(values, values**3) == pytest.approx(1.0)
    assert regressor._ranks(np.array([5.0, 5.0, 1.0, 9.0])).tolist() == [1.5, 1.5, 0.0, 3.0]


def test_the_head_does_not_read_molecule_size(compounds):
    # The claim in regressor.py: unlike the DQN's head, this one sees the pooled
    # embedding only. Heavy-atom count and steps remaining reach featurize.tensors as
    # graph features, and if they reached this network the RL agent would have "add
    # atoms" handed to it as a direction.
    torch.manual_seed(0)
    model = regressor.Regressor(CFG)
    mols = [compound.mol for compound in compounds[:8]]
    graphs = featurize.graphs(mols)
    with torch.no_grad():
        first = model(featurize.tensors(graphs, 0.0, CFG))
        second = model(featurize.tensors(graphs, 39.0, CFG))
    assert torch.equal(first, second)


def test_the_ensemble_reports_the_mean_and_the_disagreement(compounds):
    mols = [compound.mol for compound in compounds[:16]]
    torch.manual_seed(0)
    models = [regressor.Regressor(CFG) for _ in range(3)]
    prediction = regressor.predict(models, mols, CFG)
    columns = np.stack(
        [regressor.predict([model], mols, CFG).mean for model in models]
    )  # [3, 16]

    assert prediction.mean == pytest.approx(columns.mean(axis=0), abs=1e-5)
    assert prediction.spread == pytest.approx(columns.std(axis=0), abs=1e-5)
    # Differently initialized networks must actually disagree, or the spread the reward
    # subtracts as uncertainty is a column of zeros.
    assert prediction.spread.mean() > 0.0


def overfit(compounds, steps: int) -> float:
    """Train MAE after `steps` full-batch gradient steps on these compounds alone."""
    determinism.seed_everything(0)
    model = regressor.Regressor(CFG)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.RegressorConfig().learning_rate)
    batch = featurize.tensors(
        featurize.graphs([compound.mol for compound in compounds]), 0.0, CFG
    )
    labels = torch.tensor(
        [compound.pic50 for compound in compounds], dtype=torch.float32
    )  # [num_compounds]
    for _ in range(steps):
        loss = ((model(batch) - labels) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((model(batch) - labels).abs().mean())


@pytest.mark.slow
def test_can_overfit_twenty_molecules(compounds):
    # 5000 steps, not the 500 the plan guessed: at 300 it sits at 0.39 and at 2000 at
    # 0.16. The network memorizes twenty points, it is simply slow about it, and the
    # first reading of that plateau as an architecture problem was wrong.
    assert overfit(compounds[:20], steps=5000) < 0.1


@pytest.mark.slow
def test_shuffled_labels_give_chance_performance(compounds):
    # The control for a leaking split. If a model trained on scrambled labels still
    # ranks the test set, the split leaks or something in the features encodes the
    # label, and no test MAE would have told us.
    train, test = splits.scaffold_split(compounds, 0.2)
    train, validation = splits.scaffold_split(train, 0.15)
    labels = [compound.pic50 for compound in train]
    np.random.default_rng(0).shuffle(labels)
    scrambled = tuple(
        dataclasses.replace(compound, pic50=label)
        for compound, label in zip(train, labels)
    )

    model, _, _ = train_regressor.train_one(
        CFG,
        config.RegressorConfig(epochs=10),
        train_compounds=scrambled,
        validation_compounds=validation,
        seed=0,
        pretrained_encoder=None,
    )
    predicted = regressor.predict([model], [c.mol for c in test], CFG).mean
    truth = np.array([compound.pic50 for compound in test], dtype=np.float32)
    assert abs(regressor.spearman(predicted, truth)) < 0.15
