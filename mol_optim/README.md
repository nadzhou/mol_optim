# Module map

Start with `cli.py` for the shape of a run, then `env/environment.py` and
`agents/dqn.py` — between them those two hold the whole loop. Everything is plain
functions over `Chem.Mol` and tensors. There are no wrapper classes, no config globals,
and no training framework: hyperparameters travel as frozen dataclasses passed
explicitly, and the reward arrives as a function, so reading a call site tells you what
runs.

`cli.py` is the only entry point. Every other module is a library — none is a script, and
none has a `__main__` block.

The subpackages are ordered by dependency depth: `chem` imports `config` only, `datasets`
imports `chem`, `nets` imports `chem`, `env` imports `nets`, `agents` imports `env`. An
import that points backwards up that list is a bug you can see in the import line.

## The entry point

| File | What it is |
|---|---|
| `cli.py` | `mol-optim <config.toml>`: the step, agent and plot tables, and the loop over them |
| `config.py` | every knob as a frozen dataclass, and the TOML file that fills them |
| `determinism.py` | seeds every source of randomness the loop touches |

## `chem/` — RDKit primitives

| File | What it is |
|---|---|
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `splits.py` | scaffold split, and holding the seed scaffolds out of training |
| `seeds.py` | the chemotypes the RL run starts from, held out of the regressor |

## `datasets/` — run once, then forget

| File | What it is |
|---|---|
| `zinc.py` | the unlabelled set in, molecular graphs out |
| `bindingdb.py` | BindingDB's 9 GB table in, one target's compounds out, and back in |

## `nets/` — the networks and their training

| File | What it is |
|---|---|
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `pretrain.py` | masked-atom pretraining on ZINC; writes the shared encoder |
| `regressor.py` | the pIC50 network, the ensemble that predicts with a spread, and the training that writes the checkpoint |
| `q_network.py` | the Q network: the encoder plus a head that reads steps remaining |
| `policy.py` | PPO's two heads, and the segment softmax a ragged action set needs |

## `env/` — the MDP

| File | What it is |
|---|---|
| `environment.py` | candidate enumeration by RWMol edits and fragment attachment, `step` |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `rewards.py` | the reward: the pIC50 regressor, behind three guardrails |

## `agents/`

| File | What it is |
|---|---|
| `dqn.py` | the training loop, flat |
| `ppo.py` | the same MDP under PPO — the algorithm arm of the comparison |
| `random_walk.py` | tier 0 of the ladder — the number DQN has to beat |

## `report/` — looking at the output

| File | What it is |
|---|---|
| `results.py` | what a run returns, and its top-k as a drawing and an SDF |
| `audit.py` | what the agent built: motif counts and whether the scaffold survived |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |
| `plot_regressor.py` | predicted against measured, and whether disagreement predicts error |
