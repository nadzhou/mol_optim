"""The one entry point: `mol-optim <config.toml>`.

Everything else in the package is a library. This reads the config file, then runs the
steps it names, in the order it names them, printing a banner before each so a long run's
output says where it is.

The three tables below are the extension points. A new agent is a module with a
`run(settings, spec)` returning a results.Run, plus one line in AGENTS. A new plot is the
same with a PlotSpec. A new dataset or a retrained regressor is a change to the config
file alone.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy
import rdkit
import torch

from mol_optim import config
from mol_optim.chem import graph_key
from mol_optim.datasets import bindingdb, subset, zinc
from mol_optim.nets import pretrain, ranker, regressor
from mol_optim.agents import (
    dqn,
    dqn_measured,
    evolutionary,
    evolutionary_measured,
    ppo,
    random_walk,
)
from mol_optim.report import (
    audit,
    plot_pretrain,
    plot_regressor,
    plot_run,
    reachable,
    recovery,
    results,
)

AGENTS = {
    "dqn": dqn.run,
    "dqn_measured": dqn_measured.run,  # the positive control: measured pIC50, not a model
    # The search-strength baseline: same action space, same budget, no learning.
    "evolutionary": evolutionary.run,
    "evolutionary_measured": evolutionary_measured.run,
    "ppo": ppo.run,
    "random": random_walk.run,
}

PLOTS = {
    "run": plot_run.run,
    "pretrain": plot_pretrain.run,
    "regressor": plot_regressor.run,
}


def _agents(settings: config.Settings) -> None:
    settings.runs.mkdir(parents=True, exist_ok=True)
    for spec in settings.agents:
        if spec.kind not in AGENTS:
            raise ValueError(
                f"agent {spec.name!r} has kind {spec.kind!r}; "
                f"there is {', '.join(sorted(AGENTS))}"
            )
        print(f"-- {spec.name} ({spec.kind})", flush=True)

        # Written before the run, not after, so a run that crashes still says what it
        # was. A CSV whose settings live only in a config file someone has since edited
        # is a number nobody can reproduce — this repo has one of those already, the
        # 0.859 in results/README.md that a later run could not match.
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        manifest = settings.runs / f"{spec.name}.json"
        manifest.write_text(
            json.dumps(
                {
                    "agent": dataclasses.asdict(spec),
                    "bindingdb_path": str(settings.bindingdb.path),
                    "commit": commit.stdout.strip() or "not a git checkout",
                    "versions": {
                        "python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "rdkit": rdkit.__version__,
                        "numpy": numpy.__version__,
                    },
                    "torch_num_threads": torch.get_num_threads(),
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

        run = AGENTS[spec.kind](settings, spec)
        best_molecule, best_reward = run.best
        print(
            f"final_mean_reward {run.final_mean_reward:.4f}  in {run.seconds:.1f}s\n"
            f"best: {best_reward:.4f}  {best_molecule.GetNumHeavyAtoms()} heavy atoms  "
            f"{graph_key.canonical_hash(best_molecule)}"
        )
        top_stem = settings.runs / f"{spec.name}_top"
        results.top_k(run, top_stem, k=spec.top_k)
        print(f"wrote {top_stem}.png and {top_stem}.sdf")


def _plots(settings: config.Settings) -> None:
    for spec in settings.plots:
        if spec.kind not in PLOTS:
            raise ValueError(
                f"plot {spec.out} has kind {spec.kind!r}; "
                f"there is {', '.join(sorted(PLOTS))}"
            )
        PLOTS[spec.kind](settings, spec)


STEPS = {
    "zinc": zinc.run,
    "bindingdb": bindingdb.run,
    "subset": subset.run,  # the same table filtered to one element set
    "pretrain": pretrain.run,
    "regressor": regressor.run,
    "ranker": ranker.run,  # the within-series ranking reward
    "agents": _agents,
    "audit": audit.run,
    "reachable": reachable.run,
    "recovery": recovery.run,
    "plots": _plots,
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: mol-optim <config.toml>")
    settings = config.load(Path(sys.argv[1]))
    unknown = [step for step in settings.steps if step not in STEPS]
    if unknown:
        # Checked before anything runs, so a typo in the last step does not surface
        # twenty minutes into the first one.
        raise SystemExit(
            f"no step named {', '.join(unknown)}; there is {', '.join(STEPS)}"
        )
    for step in settings.steps:
        print(f"\n==> {step}", flush=True)
        STEPS[step](settings)
