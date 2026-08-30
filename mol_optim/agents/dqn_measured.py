"""The DQN against measured pIC50 — the same training loop, an honest reward.

A copy of dqn.run with two lines changed, rather than a `if spec.reward == ...` branch
in dqn.py: this is a control that will be read beside the real run and deleted when the
question it answers is settled.
"""

from dataclasses import replace

from mol_optim import config
from mol_optim.agents import dqn
from mol_optim.chem import graph_key, seeds
from mol_optim.env import measured
from mol_optim.report import results


def run(settings: config.Settings, spec: config.AgentSpec) -> results.Run:
    table = measured.load(settings.bindingdb.path)
    reward_fn = lambda mol: measured.score(table, mol) / 10.0  # pIC50 into [0, 1.1]
    init_mol = seeds.molecule(settings.bindingdb.path, spec.seed_molecule)
    if init_mol is not None:
        # The seed is measured at 10.00, so leaving it in makes "take the no-op forever"
        # a near-optimal policy and the run says nothing about what search can find.
        # report/recovery.py excludes the seed from its analogs for the same reason.
        table.pop(graph_key.canonical_hash(init_mol), None)
        print(
            f"starting from seed {spec.seed_molecule}: "
            f"{init_mol.GetNumHeavyAtoms()} heavy atoms, reward {reward_fn(init_mol):.4f}"
        )
    return dqn.train(
        replace(spec.cfg, init_mol=init_mol),
        reward_fn,
        log_path=settings.runs / f"{spec.name}.csv",
        checkpoint_path=settings.runs / f"{spec.name}.pt",
        report_every=spec.report_every,
        pretrained_encoder=spec.pretrained_encoder,
    )
