"""The test molecule set, read from an SDF of molblocks.

Loaded at import time so test modules can use these in module-level constants. The SDF
was generated once by a fixture script; the pipeline itself never reads or writes a
molecule as a SMILES string.
"""

from pathlib import Path

from mol_optim import molio

NAMED = molio.read_named(Path(__file__).parent / "fixtures" / "molecules.sdf")

# Three chemotypes, not seven. Every test that parametrizes over these is proving one
# property, and a property that holds for a chain, an aromatic with heteroatoms and a
# fused N-heterocycle does not need four more molecules to say so again.
START_MOLECULES = [NAMED[name] for name in ("ethanol", "aspirin", "caffeine")]
