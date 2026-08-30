"""The pIC50 regressor as the RL reward, with guardrails.

An agent optimizing a surrogate finds its mistakes. Three guardrails fire in order: zero
the reward below `domain_floor` Tanimoto to the training set, subtract `pessimism` times
the ensemble spread, clip at the most potent training compound. The regressor's own
evaluation measured the domain as the thing with evidence behind it (MAE 1.15 at Tanimoto
0.41 against 0.65 at 0.94) and disagreement as nearly flat (0.08).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mol_optim import config
from mol_optim.datasets import bindingdb
from mol_optim.nets import regressor

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# The agents divide `score` by this to land in [0, 1], where the published learning
# rate and the MSE loss were tuned. Zero stays zero, so the applicability-domain
# filter still reads as "nothing".
PIC50_SCALE = 10.0


@dataclass(frozen=True)
class Reward:
    models: tuple[regressor.Regressor, ...]
    cfg: config.Config
    train_fingerprints: tuple  # Morgan fingerprints of the training compounds
    ceiling: float  # the most potent pIC50 in training
    domain_floor: float
    pessimism: float


def load(
    checkpoint_path: Path,
    dataset_path: Path,
    domain_floor: float = 0.45,
    pessimism: float = 0.5,
) -> Reward:
    # 0.45 sits under the least similar held-out target of seed 0 (0.490, median 0.655)
    # and over everything a PPO run at 0.3 preferred (0.30-0.40). At 0.3 the agent
    # optimized pressed against the floor instead of inside the domain.
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing. Run the 'regressor' step first."
        )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    cfg = checkpoint["config"]
    models = []
    for state in checkpoint["models"]:
        model = regressor.Regressor(cfg)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    by_key = {
        compound.mol.GetProp("_Name"): compound for compound in bindingdb.load(dataset_path)
    }
    train_compounds = [by_key[key] for key in checkpoint["train_keys"]]
    return Reward(
        models=tuple(models),
        cfg=cfg,
        train_fingerprints=tuple(
            MORGAN.GetFingerprint(compound.mol) for compound in train_compounds
        ),
        ceiling=max(compound.pic50 for compound in train_compounds),
        domain_floor=domain_floor,
        pessimism=pessimism,
    )


def nearest_training_similarity(reward: Reward, mol: Chem.Mol) -> float:
    """Tanimoto to the closest compound the regressor was fitted on."""
    return max(
        DataStructs.BulkTanimotoSimilarity(
            MORGAN.GetFingerprint(mol), list(reward.train_fingerprints)
        )
    )


def score_many(reward: Reward, mols: Sequence[Chem.Mol]) -> np.ndarray:
    """Reward per molecule, pIC50 units, zero outside the domain.

    Batched: the RL loop scores a whole candidate set per step, five models deep.
    """
    prediction = regressor.predict(reward.models, mols, reward.cfg)
    pessimistic = prediction.mean - reward.pessimism * prediction.spread
    clipped = np.minimum(pessimistic, reward.ceiling)
    similarity = np.array(
        [nearest_training_similarity(reward, mol) for mol in mols]
    )  # [num_molecules]
    return np.where(similarity < reward.domain_floor, 0.0, clipped)


def score(reward: Reward, mol: Chem.Mol | None) -> float:
    """One molecule. The signature environment.step wants."""
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    return float(score_many(reward, [mol])[0])
