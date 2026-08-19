"""The Step 3 reward: TDC's GSK3B oracle, read straight off the molecular graph.

The oracle is a random forest of 100 trees over a 2048-bit ECFP4 fingerprint, trained
on ExCAPE-DB GSK3-beta actives and published through Therapeutics Data Commons. Its
purpose here is one question: when the reward curve is flat, is the loop broken or is
the reward broken? This reward is fixed and published, so a flat curve means the loop.

The forest arrives as arrays written by fetch_gsk3b.py rather than as a live PyTDC
call, for two reasons. PyTDC's oracle takes a SMILES string, and nothing in this loop
writes a molecule as text (plan.md, "the state is a graph, and so is its name"). And
PyTDC pins scikit-learn==1.2.2 and numpy<2 against this venv's numpy 2.5, because the
published pickle only loads under the scikit-learn that wrote it.

Every split in the forest is a fingerprint bit: threshold 0.5 over a 0/1 feature, so
"go right" means "this bit is set". That is what makes the walk below three lines.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gsk3b_forest.npz"

# Radius 2, 2048 bits: the featurization the published forest was fitted on. Changing
# either number leaves a model that still returns a number, and the number is noise.
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Forest:
    """100 decision trees, concatenated into one node table.

    A leaf's children are the leaf itself, so a walk that has arrived stays put and the
    fixed-length loop in `score` needs no per-tree bookkeeping to know when to stop.
    """

    bit: np.ndarray  # [num_nodes] int16, fingerprint bit tested here; 0 at a leaf
    left: np.ndarray  # [num_nodes] int32, next node when that bit is 0
    right: np.ndarray  # [num_nodes] int32, next node when that bit is 1
    probability: np.ndarray  # [num_nodes] float32, p(active) — only leaves carry it
    roots: np.ndarray  # [num_trees] int32, where each tree starts
    depth: int  # deepest path in any tree, so the walk below is that long


def load(path: Path = MODEL_PATH) -> Forest:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it once with: python -m mol_optim.fetch_gsk3b"
        )
    arrays = np.load(path)
    return Forest(
        bit=arrays["bit"],
        left=arrays["left"],
        right=arrays["right"],
        probability=arrays["probability"],
        roots=arrays["roots"],
        depth=int(arrays["depth"]),
    )


def score(forest: Forest, mol: Chem.Mol | None) -> float:
    """p(GSK3-beta active), 0..1. The fraction of trees whose leaf votes active."""
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    bits = MORGAN.GetFingerprintAsNumPy(mol).astype(bool)  # [2048]

    # All 100 trees walk together, one array of current nodes. Per-tree Python loops
    # cost ~4x here: the arrays are small, but the walk is 30-odd rounds deep.
    node = forest.roots  # [num_trees]
    for _ in range(forest.depth):
        node = np.where(bits[forest.bit[node]], forest.right[node], forest.left[node])
    return float(forest.probability[node].mean())
