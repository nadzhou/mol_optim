import time

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments, graph_key, seeds
from mol_optim.datasets import bindingdb
from mol_optim.env import environment
from mol_optim.report import recovery


def run(settings: config.Settings) -> None:
    spec = settings.reachable
    if spec.seed_molecule is None:
        raise ValueError("reachable needs a seed_molecule: the scaffold it enumerates from")
    compounds = bindingdb.load(settings.bindingdb.path)
    seed = seeds.choose(compounds)[spec.seed_molecule]
    analogs = recovery.held_out_analogs(compounds, seed)
    library = fragments.library([compound.mol for compound in compounds])
    print(
        f"seed {spec.seed_molecule}: {Chem.MolToSmiles(seed.mol)}\n"
        f"{len(analogs)} held-out analogs; {len(library)} substituents\n"
    )
    print(f"{'edits':>5} {'new states':>11} {'total':>10} {'reached':>8} {'>=8':>4} {'>=9':>4}")
    _enumerate(seed, analogs, library, spec.max_depth)


def _enumerate(seed, analogs, library, max_depth) -> None:
    # Same canonical hash the recovery step matches on.
    seen = {graph_key.canonical_hash(seed.mol)}
    frontier = [seed.mol]
    for depth in range(1, max_depth + 1):
        started = time.perf_counter()
        # The last level is counted and dropped: seed 1's third level is millions.
        keep = depth < max_depth
        num_new = 0
        next_frontier = []
        for mol in frontier:
            for candidate in environment.valid_actions(mol, library):
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
