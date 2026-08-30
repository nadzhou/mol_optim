"""Evolutionary search against measured pIC50 — the honest-reward arm of the baseline.

A copy of evolutionary.run with two lines changed, the same way dqn_measured.py is a copy
of dqn.run: the pair is read as a pair, and both are deleted together when the question
they answer is settled.
"""

from dataclasses import replace

from mol_optim import config
from mol_optim.agents import evolutionary
from mol_optim.chem import graph_key, seeds
from mol_optim.env import measured
from mol_optim.report import results


def run(settings: config.Settings, spec: config.AgentSpec) -> results.Run:
    table = measured.load(settings.bindingdb.path)
    reward_fn = lambda mol: measured.score(table, mol) / 10.0  # pIC50 into [0, 1.1]
    init_mol = seeds.molecule(settings.bindingdb.path, spec.seed_molecule)
    if init_mol is not None:
        # Same reason as dqn_measured: leave the seed in the table and standing still is
        # a near-optimal policy.
        table.pop(graph_key.canonical_hash(init_mol), None)
        print(
            f"starting from seed {spec.seed_molecule}: "
            f"{init_mol.GetNumHeavyAtoms()} heavy atoms, reward {reward_fn(init_mol):.4f}"
        )
    return evolutionary.search(
        replace(spec.cfg, init_mol=init_mol),
        reward_fn,
        log_path=settings.runs / f"{spec.name}.csv",
        report_every=spec.report_every,
    )
