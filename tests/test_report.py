"""The run-to-person boundary: the top-k drawing, the SDF, and the report over them.

Deduplication is the whole point of the top-k. An agent that finds one good molecule
finds it in hundreds of episodes, so a top-12 that does not deduplicate is a grid of
twelve copies of the same picture — which looks like a result, and is one molecule.

The report's own job is to say what the reward number does not, so what is asserted here
is that the audit columns survive the round trip through the SDF.
"""

import pytest

from mol_optim import molio, report, results
from tests.molecules import NAMED

DISTINCT = (
    NAMED["methane"],
    NAMED["ethanol"],
    NAMED["benzene"],
    NAMED["aspirin"],
    NAMED["caffeine"],
)


def run(rewards, molecules) -> results.Run:
    return results.Run(
        episode_rewards=tuple(rewards),
        episode_molecules=tuple(molecules),
        seconds=1.0,
    )


def test_one_molecule_found_three_hundred_times_is_one_entry(tmp_path):
    # The failure this module exists to prevent: without the dedup this SDF has four
    # records and the grid has four identical pictures.
    stem = tmp_path / "top"
    repeated = [NAMED["aspirin"]] * 300 + [NAMED["caffeine"]]
    rewards = [0.9] * 300 + [0.4]
    report.top_k(run(rewards, repeated), stem, k=12)

    written = molio.read(stem.with_suffix(".sdf"))
    assert len(written) == 2


def test_analogs_read_back_the_reward_and_the_audit(tmp_path):
    # The report reads a top-k off disk, so the reward has to survive as a number and
    # the motif counts have to be recomputed from the graph, not carried alongside it.
    stem = tmp_path / "top"
    report.top_k(run([0.9, 0.4], [NAMED["caffeine"], NAMED["aspirin"]]), stem)

    rows = report.analogs(stem.with_suffix(".sdf"), None)
    assert [row.reward for row in rows] == [0.9, 0.4]
    assert rows[0].num_heavy_atoms == NAMED["caffeine"].GetNumHeavyAtoms()
    assert all(row.sa_score > 0 for row in rows)
    # No seed was given, so there is nothing to be similar to and no scaffold to keep.
    assert all(row.tanimoto_to_seed is None for row in rows)
    assert all(row.row.scaffold_intact is None for row in rows)


def test_a_seed_gives_similarity_and_a_scaffold_check(tmp_path):
    stem = tmp_path / "top"
    report.top_k(run([0.9, 0.4], [NAMED["aspirin"], NAMED["methane"]]), stem)

    rows = report.analogs(stem.with_suffix(".sdf"), NAMED["aspirin"])
    # Against itself: identical, and its own Murcko frame is trivially present.
    assert rows[0].tanimoto_to_seed == pytest.approx(1.0)
    assert rows[0].row.scaffold_intact is True
    # Methane keeps no benzene ring, which is the check that has to be able to fail.
    assert rows[1].tanimoto_to_seed < 0.1
    assert rows[1].row.scaffold_intact is False
