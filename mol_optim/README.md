# Module map

Start with `cli.py` for the shape of a run, then `environment.py` and `train_dqn.py` —
between them those two hold the whole loop. Everything is plain functions over `Chem.Mol`
and tensors. There are no wrapper classes, no config globals, and no training framework:
hyperparameters travel as frozen dataclasses passed explicitly, and the reward arrives as
a function, so reading a call site tells you what runs.

`cli.py` is the only entry point. Every other module is a library — none is a script, and
none has a `__main__` block.

## The entry point

| File | What it is |
|---|---|
| `cli.py` | `mol-optim <config.toml>`: the step, agent and plot tables, and the loop over them |
| `config.py` | every knob as a frozen dataclass, and the TOML file that fills them |

## The loop

| File | What it is |
|---|---|
| `determinism.py` | seeds every source of randomness the loop touches |
| `environment.py` | the MDP: candidate enumeration by RWMol edits and fragment attachment, `step` |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `dqn.py` | the Q network: the encoder plus a head that reads steps remaining |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `train_dqn.py` | the training loop, flat |
| `ppo.py` | PPO's two heads, and the segment softmax a ragged action set needs |
| `train_ppo.py` | the same MDP under PPO — the algorithm arm of the comparison |
| `baseline_random.py` | tier 0 of the ladder — the number DQN has to beat |
| `results.py` | what a run returns, and its top-k as a drawing and an SDF |

## Rewards

| File | What it is |
|---|---|
| `rewards.py` | the table an agent's `reward = "..."` resolves through |
| `reward_pic50.py` | the reward itself: the pIC50 regressor, behind three guardrails |

## Data and the pIC50 model

| File | What it is |
|---|---|
| `bindingdb.py` | the EGFR dataset: pIC50 units, aggregation, loading |
| `splits.py` | scaffold split, and holding the seed scaffolds out of training |
| `seeds.py` | the chemotypes the RL run starts from, held out of the regressor |
| `regressor.py` | the pIC50 network and the ensemble that predicts with a spread |
| `train_regressor.py` | the regressor training loop, ensemble, and test report |
| `pretrain.py` | masked-atom pretraining on ZINC; writes the shared encoder |

## Run once, then forget

| File | What it is |
|---|---|
| `zinc.py` | the unlabelled set in, molecular graphs out |
| `fetch_bindingdb.py` | BindingDB's 9 GB table in, one target's compounds out |

## Looking at the output

| File | What it is |
|---|---|
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `audit.py` | what the agent built: motif counts and whether the scaffold survived |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |
| `plot_regressor.py` | predicted against measured, and whether disagreement predicts error |
