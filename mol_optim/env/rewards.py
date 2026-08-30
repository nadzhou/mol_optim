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

# Agents divide `score` by this to land in [0, 1]. Zero stays zero.
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
    # 0.45 sits under seed 0's least similar target (0.490) and over everything a
    # PPO run at 0.3 preferred (0.30-0.40).
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
