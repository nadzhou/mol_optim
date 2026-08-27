"""Masked-atom pretraining on ZINC.

Two failure modes here produce a beautiful loss curve and a worthless encoder, and
nothing else in the suite notices either one. A mask the head can see through makes the
task free. A checkpoint that does not load, or loads against a featurization whose
columns have moved, makes the pretraining a no-op that looks exactly like a hard
research problem. The first four tests and the checkpoint tests are those two.

The learning tests train, so they are slow: a run is the only way to see a loss fall.
"""

import numpy as np
import pytest
import torch

from mol_optim import config, environment, featurize, pretrain, zinc

CFG = config.Config()
# Skipped before MOLECULES is evaluated: a missing download has to skip the file,
# not error out of collection.
if not zinc.DATA_PATH.exists():
    pytest.skip(
        f"{zinc.DATA_PATH} is missing; download it with: python -m mol_optim.zinc",
        allow_module_level=True,
    )
# Enough molecules for a few hundred gradient steps, and 0.2 s to parse.
MOLECULES = zinc.molecules(limit=4000)
SMALL = config.PretrainConfig(num_holdout=1000, epochs=3)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def trained() -> pretrain.Result:
    """One 3-epoch run, shared by the tests that need a trained encoder."""
    return pretrain.pretrain(CFG, SMALL, MOLECULES)


def test_the_head_cannot_see_the_masked_atom():
    # The AttrMask bug, and the reason the mask is an all-zero row. A lone masked atom
    # has no neighbours, so nothing distinguishes a masked carbon from a masked oxygen —
    # if these two disagree by any amount, the atom's own features reach the head and
    # the task is free.
    torch.manual_seed(0)
    model = pretrain.MaskedAtomPredictor(CFG)
    single_atoms = environment.valid_actions(None, CFG)  # one molecule per cfg.atom_types
    assert len(single_atoms) > 1

    logits = [
        model(*pretrain.masked(featurize.graphs([atom]), 1.0, rng(), CFG))
        for atom in single_atoms
    ]
    for other in logits[1:]:
        assert torch.equal(logits[0], other)


def test_masking_covers_the_requested_fraction():
    graph_set = featurize.graphs(MOLECULES[:64])
    num_atoms = len(graph_set.atom_codes)
    _, rows = pretrain.masked(graph_set, 0.15, rng(), CFG)

    assert len(rows) == round(0.15 * num_atoms)
    assert len(set(rows.tolist())) == len(rows)  # no atom masked twice


def test_masking_leaves_the_graphs_it_was_given_alone():
    # The targets are read off graph_set.atom_codes *after* the batch is built, so a
    # mask that reached back into the codes would zero the answers as well: the target
    # would become element 0, carbon, and 74% of the labels would silently be right.
    graph_set = featurize.graphs(MOLECULES[:16])
    before = graph_set.atom_codes.copy()
    pretrain.masked(graph_set, 0.15, rng(), CFG)

    assert np.array_equal(graph_set.atom_codes, before)
    unmasked = featurize.tensors(graph_set, 0.0, CFG)
    assert float(unmasked.atom_features.sum()) == len(before) * len(featurize.ATOM_BLOCKS)


def test_the_prior_is_the_element_distribution_and_nothing_else():
    codes = featurize.graphs(MOLECULES[:500]).atom_codes
    prior = pretrain.marginal(codes, codes)

    # ZINC is about 74% carbon; the accuracy of always saying carbon is the number an
    # accuracy figure has to beat, and the loss is the number a run has to beat.
    assert 0.70 < prior.accuracy < 0.78
    assert prior.loss == pytest.approx(0.89, abs=0.05)


def test_a_checkpoint_from_another_featurization_is_refused(tmp_path):
    # Pretraining on one featurization and fine-tuning on another is a total, silent
    # waste: every weight lands on a column that now means something else.
    path = tmp_path / "encoder.pt"
    torch.manual_seed(0)
    pretrain.save_encoder(
        path, pretrain.MaskedAtomPredictor(CFG), CFG, SMALL, pretrain.Measurement(0.5, 0.8)
    )
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["featurization"] = "0000000000000000"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="mol_optim.pretrain"):
        pretrain.load_encoder(path, CFG)


def test_a_checkpoint_of_another_shape_is_refused(tmp_path):
    narrow = config.Config(hidden_dim=32)
    path = tmp_path / "encoder.pt"
    torch.manual_seed(0)
    pretrain.save_encoder(
        path,
        pretrain.MaskedAtomPredictor(narrow),
        narrow,
        SMALL,
        pretrain.Measurement(0.5, 0.8),
    )

    with pytest.raises(ValueError, match="32-wide"):
        pretrain.load_encoder(path, CFG)


@pytest.mark.slow
def test_holdout_loss_falls_below_the_prior(trained):
    assert trained.holdout[-1].loss < trained.prior.loss
    assert trained.holdout[-1].loss < trained.holdout[0].loss


@pytest.mark.slow
def test_real_context_beats_shuffled_context(trained):
    # The control. Same molecules, same masked atoms, atom features dealt to random
    # positions: if the two are equal, the encoder is naming atoms from something other
    # than the graph around them and the pretraining is not doing what it claims.
    assert trained.holdout[-1].loss < trained.control[-1].loss