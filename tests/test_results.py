"""What a run reports about itself.

Both properties here are read straight into the paper's table, and both are quiet when
wrong: a tail taken from the wrong end reports the exploration phase as the result, and
an argmax that returns the index instead of the molecule fails much later, in a drawing.
"""

import pytest

from mol_optim import molio, results
from tests.molecules import NAMED

MOLECULES = (NAMED["methane"], NAMED["ethanol"], NAMED["benzene"], NAMED["aspirin"])


def run(rewards: tuple[float, ...], molecules=None) -> results.Run:
    molecules = molecules or tuple(MOLECULES[i % len(MOLECULES)] for i in range(len(rewards)))
    return results.Run(
        episode_rewards=rewards, episode_molecules=molecules, seconds=1.0
    )


def test_the_final_mean_is_the_last_hundred_episodes_and_not_the_whole_run():
    # 500 episodes of nothing followed by 100 of 1.0 is a run that learned. Averaging
    # all 600 reports 0.167 and the comparison against random silently fails.
    finished = run(tuple([0.0] * 500 + [1.0] * 100))
    assert finished.final_mean_reward == pytest.approx(1.0)


# --- top_k: the drawing and SDF a run writes behind --top-k ---
#
# Deduplication is the whole point. An agent that finds one good molecule finds it in
# hundreds of episodes, so a top-12 that does not deduplicate is a grid of twelve
# copies of one picture -- which looks like a result, and is one molecule.

def test_one_molecule_found_three_hundred_times_is_one_entry(tmp_path):
    # The failure this module exists to prevent: without the dedup this SDF has four
    # records and the grid has four identical pictures.
    stem = tmp_path / "top"
    repeated = tuple([NAMED["aspirin"]] * 300 + [NAMED["caffeine"]])
    rewards = tuple([0.9] * 300 + [0.4])
    results.top_k(run(rewards, repeated), stem, k=12)

    written = molio.read(stem.with_suffix(".sdf"))
    assert len(written) == 2
