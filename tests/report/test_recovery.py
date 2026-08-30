"""Recovery of held-out analogs: the metric a run is judged on.

The seed exclusion and the deduplication are the two places this can silently overcount
— an agent that takes the no-op every step would otherwise "recover" the molecule it was
handed, and one measured compound logged under three BindingDB records would count three
times.
"""

import csv
from pathlib import Path

from rdkit import Chem

from mol_optim.chem import graph_key, seeds
from mol_optim.datasets import bindingdb
from mol_optim.report import recovery


def _compound(smiles: str, pic50: float, scaffold: str, num_measurements: int = 1):
    mol = Chem.MolFromSmiles(smiles)
    return bindingdb.Compound(
        mol=mol,
        pic50=pic50,
        num_measurements=num_measurements,
        pic50_spread=0.0,
        scaffold=scaffold,
    )


def test_seed_is_not_its_own_analog():
    seed = _compound("c1ccccc1N", 10.0, "benzene")
    same_graph = _compound("Nc1ccccc1", 9.0, "benzene")  # the seed, written differently
    other = _compound("c1ccccc1O", 7.0, "benzene")
    off_scaffold = _compound("CCCCO", 6.0, "alkane")

    analogs = recovery.held_out_analogs((seed, same_graph, other, off_scaffold), seed)

    assert list(analogs) == [graph_key.canonical_hash(other.mol)]


def test_duplicate_records_collapse_to_one_analog():
    seed = _compound("c1ccccc1N", 10.0, "benzene")
    once = _compound("c1ccccc1O", 7.0, "benzene", num_measurements=1)
    twice = _compound("Oc1ccccc1", 7.4, "benzene", num_measurements=9)

    analogs = recovery.held_out_analogs((seed, once, twice), seed)

    # Same graph, so one analog, labelled by the record with more measurements behind it.
    assert len(analogs) == 1
    assert next(iter(analogs.values())).pic50 == 7.4


def test_measure_counts_the_intersection(tmp_path: Path):
    found = _compound("c1ccccc1O", 7.0, "benzene")
    missed = _compound("c1ccccc1Cl", 8.5, "benzene")
    analogs = {graph_key.canonical_hash(c.mol): c for c in (found, missed)}

    log_path = tmp_path / "run.csv"
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["episode", "reward", "mean_loss", "epsilon", "graph_hash"])
        writer.writerow([0, 0.5, 0.0, 1.0, graph_key.canonical_hash(found.mol)])
        writer.writerow([1, 0.5, 0.0, 0.5, graph_key.canonical_hash(found.mol)])
        writer.writerow([2, 0.4, 0.0, 0.1, "not a molecule anyone measured"])

    # measured = both analogs, so the unmeasured third row is what lands in `novel`.
    result = recovery.measure(
        log_path, analogs, frozenset(analogs), seed_key="the seed, unused here"
    )

    assert (result.num_episodes, result.num_distinct, result.num_analogs) == (3, 2, 2)
    assert [c.pic50 for c in result.found] == [7.0]
    assert (result.num_known, result.num_novel) == (0, 1)


def test_seed_zero_analogs_match_the_documented_count(compounds):
    """docs/held_out_evaluation.md's table is scored against these numbers."""
    seed = seeds.choose(compounds)[0]
    analogs = recovery.held_out_analogs(compounds, seed)

    assert len(analogs) == 565
    assert sum(1 for c in analogs.values() if c.pic50 >= recovery.ACTIVE) == 163
    assert sum(1 for c in analogs.values() if c.pic50 >= recovery.POTENT) == 36
