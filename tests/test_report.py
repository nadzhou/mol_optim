"""The run-to-person boundary: the top-k drawing and the SDF beside it.

Deduplication is the whole point of this module. An agent that finds one good molecule
finds it in hundreds of episodes, so a top-12 that does not deduplicate is a grid of
twelve copies of the same picture — which looks like a result, and is one molecule.
"""

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