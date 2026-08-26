"""The pIC50 regressor as the RL reward, with guardrails.

An agent optimizing a surrogate finds its mistakes — the descriptor-scored runs built
hemiaminals against a
published model. Three guardrails fire in order: zero the reward below `domain_floor`
Tanimoto to the training set, subtract `pessimism` x ensemble spread, clip at the most
potent training compound. The regressor's own evaluation measured the domain as the
one with evidence (error
1.15 at Tanimoto 0.41 against 0.65 at 0.94) and disagreement as nearly flat (0.08).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mol_optim import bindingdb, config, regressor

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Reward:
    """The ensemble and what it was fitted on, loaded once."""

    models: tuple[regressor.Regressor, ...]
    cfg: config.Config
    train_fingerprints: tuple  # Morgan fingerprints of the training compounds
    ceiling: float  # the most potent pIC50 in training
    domain_floor: float
    pessimism: float


def load(
    checkpoint_path: Path,
    domain_floor: float = 0.3,
    pessimism: float = 0.5,
) -> Reward:
    """The ensemble, plus the training set it is allowed to have opinions near.

    domain_floor 0.3 sits below the least-similar test decile (0.41), so it zeroes
    only molecules further out than anything the reported MAE describes.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing. Build it with: "
            "python -m mol_optim.train_regressor --checkpoint " + str(checkpoint_path)
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
        compound.mol.GetProp("_Name"): compound for compound in bindingdb.load()
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
