"""How many real analogs can the action space reach at all, before any agent runs?

Every recovery number is a fraction of 565 held-out analogs, and that denominator
assumes the MDP can build them. It mostly cannot. Two separate limits, both measured
here:

**Composition.** One edit adds at most one atom, and `cfg.atom_types` says which
elements it may add. An analog carrying a fluorine the seed does not have is
unreachable at any depth, not merely far away. For the analogs that are reachable in
principle, the per-element count difference is a sound lower bound on the number of
edits: each edit changes the atom multiset by at most one insertion or one deletion,
and bond edits change it not at all.

**Distance.** Breadth-first search over `environment.valid_actions` gives the exact
count reachable within k edits. It is exact, so it is the number an agent at that
budget is really measured against — but the frontier grows about 40-fold per level,
so depth 4 is minutes and depth 5 is tens of minutes. Past that the composition bound
is what there is.

No training, no reward, no network. This is a property of the action space and the
seed alone, which is why it is worth having before spending a run on either.
"""

import collections
import time

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import graph_key, seeds
from mol_optim.datasets import bindingdb
from mol_optim.env import environment
from mol_optim.report import recovery


def edit_lower_bound(
    seed: Chem.Mol, target: Chem.Mol, cfg: config.Config
) -> int | None:
    """Fewest edits that could turn `seed` into `target`, or None if no number of them can.

    A lower bound, not the distance: it counts atoms only, so two molecules with the same
    formula and different connectivity come back 0.
    """
    seed_atoms = collections.Counter(atom.GetSymbol() for atom in seed.GetAtoms())
    target_atoms = collections.Counter(atom.GetSymbol() for atom in target.GetAtoms())
    for element, count in target_atoms.items():
        if count > seed_atoms[element] and element not in cfg.atom_types:
            return None
    if not cfg.allow_removal and any(
        count > target_atoms[element] for element, count in seed_atoms.items()
    ):
        return None
    return sum(
        abs(target_atoms[element] - seed_atoms[element])
        for element in set(seed_atoms) | set(target_atoms)
    )


def run(settings: config.Settings) -> None:
    spec = settings.reachable
    if spec.seed_molecule is None:
        raise ValueError("reachable needs a seed_molecule: the scaffold it enumerates from")
    cfg = spec.cfg
    compounds = bindingdb.load(settings.bindingdb.path)
    seed = seeds.choose(compounds)[spec.seed_molecule]
    analogs = recovery.held_out_analogs(compounds, seed)
    print(
        f"seed {spec.seed_molecule}: {Chem.MolToSmiles(seed.mol)}\n"
        f"{len(analogs)} held-out analogs; action space adds "
        f"{', '.join(cfg.atom_types)}, removal {'on' if cfg.allow_removal else 'off'}\n"
    )

    bounds = {key: edit_lower_bound(seed.mol, c.mol, cfg) for key, c in analogs.items()}
    seed_atoms = collections.Counter(atom.GetSymbol() for atom in seed.mol.GetAtoms())
    blocked = collections.Counter()
    for key, bound in bounds.items():
        if bound is None:
            # Only the elements this analog actually needs more of than the seed has.
            # Counting every element outside atom_types instead credits Br with blocking
            # 143 analogs the seed's own bromine covers.
            atoms = collections.Counter(a.GetSymbol() for a in analogs[key].mol.GetAtoms())
            blocked.update(
                element
                for element, count in atoms.items()
                if element not in cfg.atom_types and count > seed_atoms[element]
            )
    unreachable = sum(1 for bound in bounds.values() if bound is None)
    print(
        f"unreachable at any depth: {unreachable} of {len(analogs)} "
        f"({100 * unreachable / len(analogs):.0f}%) — they carry an element the action "
        f"space cannot add\n  blocking elements: "
        + ", ".join(f"{e} in {n}" for e, n in blocked.most_common())
    )

    reachable_bounds = sorted(b for b in bounds.values() if b is not None)
    print(f"\n{'edits':>5} {'at most':>8}  (upper bound from the composition difference)")
    for depth in range(1, spec.max_depth + 1):
        print(f"{depth:>5} {sum(1 for b in reachable_bounds if b <= depth):>8}")
    print(f"{'any':>5} {len(reachable_bounds):>8}\n")

    # The exact count. Deduplication is on the same canonical hash the recovery step
    # matches on, so "reached" here and "found" there mean the same thing.
    print(f"{'edits':>5} {'new states':>11} {'total':>10} {'reached':>8} {'>=8':>4} {'>=9':>4}")
    seen = {graph_key.canonical_hash(seed.mol)}
    frontier = [seed.mol]
    for depth in range(1, spec.max_depth + 1):
        started = time.perf_counter()
        # The last level's molecules are counted and dropped. Keeping them costs about a
        # kilobyte each, and seed 1's third level is millions.
        keep = depth < spec.max_depth
        num_new = 0
        next_frontier = []
        for mol in frontier:
            for candidate in environment.valid_actions(mol, cfg):
                key = graph_key.canonical_hash(candidate)
                if key not in seen:
                    seen.add(key)
                    num_new += 1
                    if keep:
                        next_frontier.append(candidate)
        frontier = next_frontier
        reached = [analogs[key] for key in seen & set(analogs)]
        print(
            f"{depth:>5} {num_new:>11} {len(seen):>10} {len(reached):>8} "
            f"{sum(1 for c in reached if c.pic50 >= recovery.ACTIVE):>4} "
            f"{sum(1 for c in reached if c.pic50 >= recovery.POTENT):>4}"
            f"   [{time.perf_counter() - started:.0f}s]",
            flush=True,
        )
        if keep and not frontier:
            break  # nothing new at this level, so nothing deeper either
