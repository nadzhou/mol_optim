# Module map

Start with `environment.py` and `train_dqn.py` — between them they hold the whole loop.
Everything is plain functions over `Chem.Mol` and tensors. There are no wrapper classes,
no config globals, and no training framework: hyperparameters travel as frozen
dataclasses passed explicitly, and the reward arrives as a function, so reading a call
site tells you what runs.

## The loop

| File | What it is |
|---|---|
| `determinism.py` | seeds every source of randomness the loop touches |
| `config.py` | every hyperparameter, one frozen dataclass, passed explicitly |
| `environment.py` | the MDP: candidate enumeration by RWMol edits, `step` |
| `graph_key.py` | the state key: a canonical name taken off the graph, no SMILES |
| `featurize.py` | molecular graph to network tensors: int8 codes, one-hot at the network |
| `encoder.py` | the GNN: message passing over atoms and bonds, mean pooled |
| `dqn.py` | the Q network: the encoder plus a head that reads steps remaining |
| `replay_buffer.py` | ours; ragged, because the target is a max over a candidate set |
| `train_dqn.py` | the training loop, flat |
| `baseline_random.py` | tier 0 of the ladder — the number DQN has to beat |
| `results.py` | what a run returns; shared by the DQN loop and the baseline |

## Rewards

| File | What it is |
|---|---|
| `rewards.py` | QED: RDKit's drug-likeness score, kept as the control |
| `reward_pic50.py` | the real reward: the pIC50 regressor, behind three guardrails |

## Data and the pIC50 model

| File | What it is |
|---|---|
| `bindingdb.py` | the EGFR dataset: pIC50 units, aggregation, loading |
| `splits.py` | scaffold split, and holding the seed scaffolds out of training |
| `seeds.py` | the chemotypes the RL run starts from, held out of the regressor |
| `vocabulary.py` | the fragment vocabulary: precedented decorations cut from the actives |
| `regressor.py` | the pIC50 network and the ensemble that predicts with a spread |
| `train_regressor.py` | the regressor training loop, ensemble, and test report |
| `pretrain.py` | masked-atom pretraining on ZINC; writes the shared encoder |
| `finetune_zinc.py` | is the pretrained checkpoint a better place to start? |

## Run once, then forget

| File | What it is |
|---|---|
| `zinc.py` | ZINC 250k in, molecular graphs out |
| `fetch_bindingdb.py` | BindingDB's 9 GB table in, one target's compounds out |

## Looking at the output

| File | What it is |
|---|---|
| `molio.py` | SDF in and out; the only place a graph becomes bytes |
| `report.py` | top-k molecules as a drawing and an SDF |
| `audit.py` | what the agent built: motif counts and whether the scaffold survived |
| `plot_run.py` | reward and loss curves from a run log |
| `plot_pretrain.py` | loss and accuracy curves from a pretraining log |
| `plot_regressor.py` | predicted against measured, and whether disagreement predicts error |
