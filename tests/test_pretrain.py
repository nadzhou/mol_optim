"""Step 3b: masked-atom pretraining on ZINC.

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
from rdkit import Chem

from mol_optim import config, dqn, environment, featurize, pretrain, zinc

CFG = config.Config()
# Enough molecules for a few hundred gradient steps, and 0.2 s to parse.
MOLECULES = zinc.molecules(limit=4000)
SMALL = config.PretrainConfig(num_holdout=1000, epochs=3)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def trained() -> pretrain.Result:
    """One 3-epoch run, shared by the tests that need a trained encoder."""
    return pretrain.pretrain(CFG, SMALL, MOLECULES)


def test_a_masked_row_is_empty_and_every_other_row_is_intact():
    graph_set = featurize.graphs(MOLECULES[:16])
    batch, rows = pretrain.masked(graph_set, 0.15, rng(), CFG)

    assert float(batch.atom_features[rows].sum()) == 0.0
    untouched = np.setdiff1d(np.arange(len(graph_set.atom_codes)), rows)
    # One 1 per field: the mask took whole atoms, not columns.
    assert torch.equal(
        batch.atom_features[untouched].sum(dim=1),
        torch.full((len(untouched),), float(len(featurize.ATOM_BLOCKS))),
    )


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


def test_shuffling_atoms_keeps_the_graph_and_moves_the_chemistry():
    graph_set = featurize.graphs(MOLECULES[:16])
    shuffled = pretrain.with_shuffled_atoms(graph_set, rng())

    assert np.array_equal(shuffled.edge_index, graph_set.edge_index)
    assert np.array_equal(shuffled.bond_codes, graph_set.bond_codes)
    assert not np.array_equal(shuffled.atom_codes, graph_set.atom_codes)
    # The same atoms, in other places: a permutation, not new chemistry.
    assert np.array_equal(
        np.sort(shuffled.atom_codes, axis=0), np.sort(graph_set.atom_codes, axis=0)
    )


def test_the_prior_is_the_element_distribution_and_nothing_else():
    codes = featurize.graphs(MOLECULES[:500]).atom_codes
    prior = pretrain.marginal(codes, codes)

    # ZINC is about 74% carbon; the accuracy of always saying carbon is the number an
    # accuracy figure has to beat, and the loss is the number a run has to beat.
    assert 0.70 < prior.accuracy < 0.78
    assert prior.loss == pytest.approx(0.89, abs=0.05)


def test_checkpoint_roundtrip_is_exact(tmp_path):
    torch.manual_seed(0)
    model = pretrain.MaskedAtomPredictor(CFG)
    path = tmp_path / "encoder.pt"
    pretrain.save_encoder(path, model, CFG, SMALL, pretrain.Measurement(0.5, 0.8))

    loaded = pretrain.MaskedAtomPredictor(CFG)
    loaded.encoder.load_state_dict(pretrain.load_encoder(path, CFG))
    batch = featurize.tensors(featurize.graphs(MOLECULES[:32]), 0.0, CFG)
    with torch.no_grad():
        assert torch.allclose(model.encoder(batch), loaded.encoder(batch), atol=1e-6)


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


def test_the_pretrained_encoder_lands_in_the_q_network(tmp_path):
    torch.manual_seed(0)
    pretrained = pretrain.MaskedAtomPredictor(CFG)
    path = tmp_path / "encoder.pt"
    pretrain.save_encoder(path, pretrained, CFG, SMALL, pretrain.Measurement(0.5, 0.8))

    torch.manual_seed(1)  # a different draw, so an unloaded encoder cannot pass by luck
    q_network = dqn.MolDQN(CFG)
    head_before = q_network.linear_1.weight.clone()
    q_network.encoder.load_state_dict(pretrain.load_encoder(path, CFG))

    for name, parameter in q_network.encoder.named_parameters():
        assert torch.equal(parameter, dict(pretrained.encoder.named_parameters())[name])
    # Only the encoder is pretrained. The Q head has no meaning before a reward exists.
    assert torch.equal(q_network.linear_1.weight, head_before)
    batch = featurize.tensors(featurize.graphs(MOLECULES[:8]), 7, CFG)
    assert q_network(batch).shape == (8, 1)


def test_the_loss_falls_on_a_handful_of_molecules():
    # The pretraining equivalent of overfitting 20 molecules: if 100 steps on 32
    # molecules cannot drive the masked-element loss below the prior, the gradient does
    # not reach the encoder and no amount of ZINC will help.
    graph_set = featurize.graphs(MOLECULES[:32])
    prior = pretrain.marginal(graph_set.atom_codes, graph_set.atom_codes)
    torch.manual_seed(0)
    model = pretrain.MaskedAtomPredictor(CFG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    generator = rng()
    for _ in range(100):
        batch, rows = pretrain.masked(graph_set, 0.15, generator, CFG)
        targets = torch.from_numpy(graph_set.atom_codes[rows, 0].astype(np.int64))
        loss = torch.nn.functional.cross_entropy(model(batch, rows), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final = pretrain.measure(model, [graph_set], [graph_set], SMALL, CFG)
    assert final.loss < prior.loss


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


def test_the_logp_probe_reads_logp_out_of_a_random_encoder():
    # The probe is asserted on a *random* encoder on purpose. A mean-pooled random GNN
    # is already a strong predictor of Crippen logP — logP is a sum of per-atom
    # contributions, and pooled random features carry composition — so this number is
    # the null that pretraining has to beat, and it is the number that says the probe
    # machinery works at all. The pretrained encoder is not asserted to beat it, and
    # measured over a full ZINC run it mostly does not — see plan.md Step 3b, where the
    # frozen probe loses on six of seven properties while fine-tuning from the same
    # checkpoint wins every comparison. The probe is a sanity check, not the evidence.
    torch.manual_seed(1)
    untrained = pretrain.MaskedAtomPredictor(CFG)
    probe_molecules = MOLECULES[-600:]
    random_r2 = pretrain.logp_probe(
        untrained.encoder, probe_molecules[:300], probe_molecules[300:], CFG
    )
    assert 0.3 < random_r2 < 1.0
