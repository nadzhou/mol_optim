"""The molecules the RL run starts from.

Distinct chemotypes rather than the top-k most potent, because the top-k of an EGFR set
are forty analogs of one quinazoline. Largest active scaffold clusters first: a series
with forty measured analogs is one a chemist has already walked around.

Used twice — as the RL episodes' starting molecules, and as the scaffolds held out of
the regressor's training set. Both, or the regressor knows the answer where it starts.
"""

from typing import Sequence

from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mol_optim import bindingdb, splits

# For telling chemotypes apart: two Murcko frames can still be one family to a chemist.
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# Above this, two seeds would send the agent around the same corner twice.
MAX_SEED_SIMILARITY = 0.6


def choose(
    compounds: Sequence[bindingdb.Compound],
    num_seeds: int = 5,
    min_pic50: float = 8.0,
) -> tuple[bindingdb.Compound, ...]:
    """One representative from each of the largest active scaffold clusters.

    min_pic50 8.0 is 10 nM — a lead series, not a screening hit, and EGFR has hundreds.

    Two filters on top of "largest clusters first", both from reading what it picked
    without them: the representative's label must be measured more than once and agree
    within a log (the first pick spread 2.73 logs across five), and a cluster is skipped
    if its representative is within MAX_SEED_SIMILARITY of a seed already taken (two of
    the first five differed only in where a ring nitrogen sat).
    """
    actives = [compound for compound in compounds if compound.pic50 >= min_pic50]
    seeds: list[bindingdb.Compound] = []
    fingerprints = []

    for group in splits.by_scaffold(actives).values():  # largest cluster first
        # Ties broken by name so the choice does not move between runs.
        by_potency = sorted(
            group, key=lambda c: (-c.pic50, c.mol.GetProp("_Name"))
        )
        settled = [
            c for c in by_potency if c.num_measurements > 1 and c.pic50_spread <= 1.0
        ]
        candidate = (settled or by_potency)[0]

        fingerprint = MORGAN.GetFingerprint(candidate.mol)
        if any(
            DataStructs.TanimotoSimilarity(fingerprint, other) > MAX_SEED_SIMILARITY
            for other in fingerprints
        ):
            continue
        seeds.append(candidate)
        fingerprints.append(fingerprint)
        if len(seeds) == num_seeds:
            break
    return tuple(seeds)


def held_out_scaffolds(seeds: Sequence[bindingdb.Compound]) -> frozenset[str]:
    """The scaffolds that must not appear in the regressor's training set."""
    return frozenset(seed.scaffold for seed in seeds)
