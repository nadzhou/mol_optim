# Running things

There is one command:

```bash
mol-optim configs/config.toml
```

It reads the config file, runs the steps that file's `steps` list names, in that order,
and prints a banner before each. Everything else in the package is a library — no module
is a script, and nothing is run with `python -m`.

The virtual environment has to be active (`source .venv/bin/activate`), and paths in the
config file are relative to the directory you run the command from, which is the
repository root. Runs write into `runs/`, which is not tracked; the plots worth keeping
are written straight into [`results/`](../results/README.md).

[`configs/config.toml`](../configs/config.toml) is the default pipeline. Copy it and edit the copy to run
something else — the whole point is that a different experiment is a different file, not
a different command line.

## The steps

Each name in `steps` maps to one entry in `mol_optim/cli.py`'s `STEPS` table.

| Step | Reads | Writes | Roughly |
|---|---|---|---|
| `zinc` | the URL and hash in `[zinc]` | `data/zinc.tab`, 12 MB | one download |
| `bindingdb` | the URL and hash in `[bindingdb]` | `data/egfr_ic50.sdf`, from a 593 MB download | one download, then a 9 GB scan |
| `pretrain` | `data/zinc.tab` | `models/zinc_encoder.pt` | 7 min, 3.3 GB of memory |
| `regressor` | `data/egfr_ic50.sdf` | `models/egfr_regressor.pt` | 11 min |
| `agents` | the regressor and encoder checkpoints | one CSV, checkpoint and top-k per `[[agents]]` table | 2–5 min each |
| `audit` | the SDFs named in `[audit]` | its counts, to stdout | seconds |
| `plots` | whatever each `[[plots]]` table names | a PNG each, under `results/` | seconds |

Both downloads are pinned by checksum, so a silently changed upstream file fails loudly
instead of quietly moving your numbers. The encoder checkpoint carries a hash of the
featurization and refuses to load if it does not match, which is what stops a changed
feature table from being loaded into a network trained against the old one.

## Changing the run

**A different agent.** Add a `[[agents]]` table. `kind` picks the training loop out of
`cli.AGENTS` — `dqn`, `ppo` or `random` — and `name` names everything that run writes.
The tables run in the order they appear, so the random floor and the DQN that has to beat
it live in one file:

```toml
[[agents]]
kind = "random"
name = "random_pic50_seed0"
reward = "pic50"
seed_molecule = 0
episodes = 1000
max_steps_per_episode = 6
```

Anything in an agent table that is not an agent setting is a `Config` or `PPOConfig` knob,
so `gamma`, `learning_rate` and `epsilon_start` go in the same flat list. A key neither
dataclass has is an error, not a setting that silently does nothing.

`seed_molecule` indexes the five seed scaffolds a run can start from, each evaluated
against its own measured pIC50. `max_steps_per_episode = 6` is deliberate: at 40 edits the
agent leaves the regressor's applicability domain and the prediction stops meaning
anything. `3` rather than `6` is the one change measured to improve recovery of real
held-out compounds — 8 found against 7, 2 measured actives against 0 — and it cuts the N-N
rate in the top-12 from 12/12 to 7/12.

PPO needs `seed_molecule`: its value head reads a state graph, and the empty molecule has
none. DQN scores candidates only, so it does not have that problem. Give PPO the same
environment budget as the DQN if you want the two to compare — 63 updates of 16 episodes
is 1008 episodes against the DQN's 1000.

**A different dataset.** Change `url`, the checksum and `path` under `[zinc]`, or the same
plus `uniprot` and `construct` under `[bindingdb]`. `construct` is a name, not an id:
P00533 covers 51 EGFR constructs, and pooling wild type with T790M puts one compound's two
very different numbers under one label.

**A retrained regressor.** Drop `pretrained_encoder` from `[regressor]` to fit from a
random init — that is the null the ZINC pretraining is measured against, and it is worth
about 0.06 MAE. Point an agent at a different checkpoint with its own `regressor` key.

**A new reward.** A function in `mol_optim/rewards.py` and one line in that module's
`REWARDS` table; then `reward = "<name>"` in an agent table.

## Look at what it built

A climbing reward curve is not the result. The `audit` step is: it counts the
substructures past runs have gone wrong in — hemiaminals, N-hydroxyls, chains of
catenated nitrogen — and checks that the seed's scaffold survived. The `agents` step has
already written each run's best distinct molecules as a drawing and an SDF.

## Tests

```bash
pytest -m "not slow"   # 104 tests, about 14 seconds
pytest                 # adds two 1000-episode pIC50 runs and a pretraining run
```

The slow tests need `models/egfr_regressor.pt` and `data/egfr_ic50.sdf`. On a fresh
checkout they skip rather than error.
