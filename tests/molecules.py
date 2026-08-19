"""The test molecule set, read from an SDF of molblocks.

Loaded at import time so test modules can use these in module-level constants. The SDF
was generated once by a fixture script; the pipeline itself never reads or writes a
molecule as a SMILES string.
"""

from pathlib import Path

from mol_optim import molio

NAMED = molio.read_named(Path(__file__).parent / "fixtures" / "molecules.sdf")

# Chemotypes that exercise different code paths: chains, aromatics, fused rings,
# heteroatoms, and a molecule with nothing but single bonds.
START_MOLECULES = [
    NAMED[name]
    for name in (
        "methane",
        "ethanol",
        "benzene",
        "aspirin",
        "cyclohexane",
        "caffeine",
        "sorbitol",
    )
]
