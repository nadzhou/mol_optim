"""The GNN encoder and the Q network on top of it.

Three of these catch bugs that train perfectly well and give the wrong answer:
permutation invariance, batch-vs-single agreement, and steps remaining reaching the
head. plan.md Step 2.
"""

import time

import numpy as np
import pytest
import torch
from rdkit import Chem

from mol_optim import config, dqn, environment, featurize, graph_key
from tests.molecules import NAMED

CFG = config.Config()
RAGGED = (NAMED["methane"], NAMED["aspirin"], NAMED["ethanol"], NAMED["caffeine"])


def network(cfg: config.Config = CFG) -> dqn.MolDQN:
    torch.manual_seed(0)
    return dqn.MolDQN(cfg)


def score(mols, steps_remaining, model) -> torch.Tensor:
    with torch.no_grad():
        return model(featurize.tensors(featurize.graphs(mols), steps_remaining, CFG))


def test_embedding_is_invariant_to_atom_ordering():
    # If this fails the network is keying on atom index. It trains fine and generalizes
    # badly, and no other test in the suite notices.
    model = network()
    mol = NAMED["paracetamol"]
    permutation = [int(i) for i in np.random.default_rng(0).permutation(mol.GetNumAtoms())]
    renumbered = graph_key.normalize(Chem.RenumberAtoms(mol, permutation))
    assert torch.allclose(
        score([mol], 5, model), score([renumbered], 5, model), atol=1e-5
    )


def test_scoring_is_batch_invariant():
    # Catches an aggregation that lets one candidate's atoms leak into another's pooled
    # embedding. Ragged on purpose: 1 atom next to 14.
    model = network()
    one_at_a_time = torch.cat([score([mol], 9, model) for mol in RAGGED])
    batched = score(RAGGED, 9, model)
    assert torch.allclose(one_at_a_time, batched, atol=1e-5)


def test_a_real_candidate_set_scores_the_same_batched_as_alone():
    # The same property on the shapes the training loop actually sees: every candidate
    # reachable from one molecule, in one forward pass.
    model = network()
    candidates = environment.valid_actions(NAMED["caffeine"], CFG)
    assert len(candidates) > 20
    one_at_a_time = torch.cat([score([mol], 3, model) for mol in candidates])
    assert torch.allclose(one_at_a_time, score(candidates, 3, model), atol=1e-5)


def test_steps_remaining_changes_the_q_value():
    # Trivial, but if the feature is dropped in the swap the MDP silently becomes
    # non-stationary: the same molecule carries one Q value for two different states.
    model = network()
    assert not torch.allclose(
        score([NAMED["aspirin"]], 1, model), score([NAMED["aspirin"]], 39, model)
    )


def test_two_molecules_with_the_same_atoms_and_different_bonds_score_differently():
    # Mean pooling over atoms alone cannot tell these apart; only the message passing
    # can. If they agree, the graph structure is not reaching the head.
    model = network()
    hexane, cyclohexane = NAMED["hexane"], NAMED["cyclohexane"]
    assert not torch.allclose(score([hexane], 4, model), score([cyclohexane], 4, model))


def test_every_parameter_gets_a_real_gradient():
    model = network()
    batch = featurize.tensors(featurize.graphs(RAGGED), 6, CFG)
    model(batch).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.any(parameter.grad != 0), name


def test_the_head_reads_the_pooled_embedding_and_the_graph_features():
    model = network()
    batch = featurize.tensors(featurize.graphs(RAGGED), 6, CFG)
    embedded = model.encoder(batch)
    assert embedded.shape == (len(RAGGED), CFG.hidden_dim)
    assert model(batch).shape == (len(RAGGED), 1)


@pytest.mark.slow
def test_step_latency_under_budget():
    # Measured here or discovered in week four. The candidate set is the whole cost of a
    # step: every candidate is featurized and scored at every step of every episode.
    model = network()
    mol = NAMED["caffeine"]
    candidates = environment.valid_actions(mol, CFG)
    timings = []
    for _ in range(20):
        started = time.perf_counter()
        with torch.no_grad():
            model(featurize.tensors(featurize.graphs(candidates), 7, CFG))
        timings.append(time.perf_counter() - started)
    median = sorted(timings)[len(timings) // 2]
    print(f"{len(candidates)} candidates, median {1000 * median:.1f} ms")
    assert median < 0.5
