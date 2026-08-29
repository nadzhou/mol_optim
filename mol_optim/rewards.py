"""The reward functions an agent can be pointed at, and the table that names them.

`build` is what the config file's `reward = "..."` resolves through — a new reward is a
function here plus one line in REWARDS.
"""

from pathlib import Path
from typing import Callable

from rdkit import Chem

from mol_optim import config, reward_pic50

# The trainers divide the pIC50 reward by this to land in [0, 1], where the published
# learning rate and the MSE loss were tuned. Zero stays zero, so the applicability-domain
# filter still reads as "nothing".
PIC50_SCALE = 10.0


def _pic50(
    spec: config.AgentSpec, dataset_path: Path
) -> Callable[[Chem.Mol | None], float]:
    reward = reward_pic50.load(spec.regressor, dataset_path)
    return lambda mol: reward_pic50.score(reward, mol) / PIC50_SCALE


REWARDS = {"pic50": _pic50}


def build(
    spec: config.AgentSpec, dataset_path: Path
) -> Callable[[Chem.Mol | None], float]:
    if spec.reward not in REWARDS:
        raise ValueError(
            f"no reward named {spec.reward!r}; there is {', '.join(sorted(REWARDS))}"
        )
    return REWARDS[spec.reward](spec, dataset_path)
