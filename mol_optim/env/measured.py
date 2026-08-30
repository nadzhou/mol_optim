"""The positive control: measured pIC50 by lookup, not a model's guess at it.

The regressor reward answers "how potent does the model think this is". This one answers
"how potent is it, actually" — for the molecules BindingDB has a number for, and zero for
everything else. Nothing is fitted, so nothing can be gamed: an agent scored this way
either builds compounds that were really measured or collects nothing.

That is what makes it the control. docs/held_out_evaluation.md's central claim is that
the reward model, not the search, is what keeps recovery at 2.3%. The claim is an
inference until a run with an honest reward is put beside a run with a fitted one, on the
same MDP, same encoder and same budget. If the honest reward recovers many more analogs,
the reward model is the bottleneck. If it recovers about as few, the search is.

The reward is sparse in a way the regressor's is not — 10,850 molecules have a number
and every other graph scores 0 — so read the two curves as bounds on what the search can
find, never as an algorithm comparison.
"""

from pathlib import Path

from rdkit import Chem

from mol_optim.chem import graph_key
from mol_optim.datasets import bindingdb


def load(dataset_path: Path) -> dict[str, float]:
    """Measured pIC50 per constitutional graph.

    Keyed by canonical_hash, not by name: the agent builds a graph and has no name for
    it. Where two records collapse to one graph the more-measured one wins, which is the
    same rule report/recovery.py counts hits by — so a molecule this reward pays for is
    exactly a molecule that shows up as recovered.
    """
    table: dict[str, float] = {}
    best_measurements: dict[str, int] = {}
    for compound in bindingdb.load(dataset_path):
        key = graph_key.canonical_hash(compound.mol)
        if compound.num_measurements > best_measurements.get(key, -1):
            table[key] = compound.pic50
            best_measurements[key] = compound.num_measurements
    return table


def score(table: dict[str, float], mol: Chem.Mol | None) -> float:
    """One molecule, pIC50 units. Zero for anything nobody has measured."""
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    return table.get(graph_key.canonical_hash(mol), 0.0)
