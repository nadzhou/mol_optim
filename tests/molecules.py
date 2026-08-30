from pathlib import Path

from mol_optim.chem import molio

NAMED = molio.read_named(Path(__file__).parent / "fixtures" / "molecules.sdf")

# Three chemotypes, not seven. Every test that parametrizes over these is proving one
# property, and a property that holds for a chain, an aromatic with heteroatoms and a
# fused N-heterocycle does not need four more molecules to say so again.
START_MOLECULES = [NAMED[name] for name in ("ethanol", "aspirin", "caffeine")]


def size_reward(mol) -> float:
    """A cheap stand-in reward for the loop tests: 0..1, rising with heavy-atom count.

    The loops are tested for determinism, shapes and termination, none of which depend on
    what the reward means — so this is a pure arithmetic function rather than a fitted
    model the test would have to build first.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    return min(mol.GetNumHeavyAtoms() / 40.0, 1.0)
