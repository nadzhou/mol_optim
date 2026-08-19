# RL Molecule Optimization — Derivatization Workflow

Graph-based (molecules are graphs end to end, never SMILES), RL-driven molecular
derivatization, scored by a learned binding-affinity surrogate and constrained by
drug-likeness.

Coding style: see [CLAUDE.md](CLAUDE.md).

## Core loop

```
seed scaffold (known kinase chemotype)
  → enumerate valid fragment edits (attach group / remove group / no-op)
  → GNN scores each candidate
  → pick action, apply, repeat for N steps
  → terminal reward = predicted pIC50 (× drug-likeness terms)
```

The action space is a *set of candidate next-molecules*, enumerated by the
environment. The network scores each candidate; it does not emit a fixed-size
action distribution. That much is MolDQN's formulation and we keep it — it is the
right shape for derivatization.

What we change is the granularity: MolDQN's edits are atom-level, ours are
fragment-level, drawn from a vocabulary of precedented functional groups. See
[Action space](#action-space--fragment-edits-over-a-precedented-vocabulary).

## Scope — lead optimization, not de novo

We start from known kinase scaffolds and improve them. This is decided, and it
propagates everywhere:

- **The seed is a real chemotype**, drawn from the target's known inhibitors —
  several distinct scaffolds, not one. Coverage of chemical space comes from the
  spread of seeds, not from any single run wandering far.
- **The scaffold is immutable.** Every state in an episode contains the seed
  scaffold as a substructure. If it doesn't, the episode has drifted into de novo
  design and the run is invalid. This is a testable invariant, not a hope.
- **Evaluation is per-seed** — best analog of *this* lead, measured against the
  seed's own pIC50, not a global leaderboard. A method that gets one great
  molecule out of one lucky seed has not solved lead optimization.

---

# Part I — Design

## What we're starting from

Two repos are cloned here:

- `MolDQN-pytorch/` — PyTorch port of MolDQN. The working base.
- `google-research/mol_dqn/` — original TF1 implementation. Reference only, but
  `chemgraph/multi_obj_opt.py` and `target_sas.py` show how the authors handled
  multi-objective and similarity-constrained rewards.

### What MolDQN actually is (important)

It is **not** a GNN. Per `hyp.py` + `dqn.py`:

- State = Morgan fingerprint, radius 3, 2048 bits, **+ 1 scalar** (steps remaining)
- Network = MLP 2049 → 1024 → 512 → 128 → 32 → 1
- Reward = QED, discounted by `discount_factor ** steps_left`

So the GNN work is *replacing the state encoder*, not bolting onto it.

### Three things that must survive the swap

1. **Q(s,a) over a variable candidate set.** `agent.py:71-73` enumerates all valid
   next states, scores each, argmaxes. The GNN encodes a batched set of candidate
   graphs → one scalar per graph. No fixed output head.
2. **Steps-remaining.** The discount depends on steps left, so the MDP is
   non-stationary without it. Concatenate it to the **pooled graph embedding**,
   not to node features.
3. **The state is a graph, and so is its name.** `environment.py` does real graph
   edits on an `RWMol`. The original keeps the state as a SMILES string and uses it
   as the dedup key; we do not. `graph_key.canonical_hash` names a molecule from
   RDKit's canonical atom ranking, so nothing in the loop writes or parses a
   molecule as text. Molecules become text once, in `report.py`, for a person to
   read.

   Two reasons, both found the hard way. Writing a SMILES and reading it back
   re-perceives aromaticity, and the perception on an edited `RWMol` does not always
   match the perception on a fresh parse — one molecule, two names, a duplicated
   candidate and two replay-buffer keys. And the round trip costs a write plus a
   parse per candidate, against 14 us to hash the graph.

   The key is constitutional: it does not separate stereoisomers, because the
   atom-level action space cannot make one. That has to change at Step 4, where real
   inhibitors arrive with stereochemistry that matters.

### Dependency landmine — fix before anything else

`MolDQN-pytorch/requirements.txt` pins `tensorflow==1.14`, `torch==1.4.0`, and
`agent.py` does `from baselines.deepq import replay_buffer` (OpenAI baselines,
TF1-era). None of this installs on Apple Silicon in 2026.

Fix: write our own replay buffer (~40 lines), drop TF entirely, move to current
torch + PyTorch Geometric.

## Action space — fragment edits over a precedented vocabulary

MolDQN's action space is atom-level: add one atom from `atom_types`, add a bond,
remove a bond. We replace it with fragment-level edits.

| Action | Definition |
|---|---|
| **Attach** | Bond a vocabulary fragment to an atom with free valence |
| **Remove** | Delete a previously attached fragment — decorations only, never scaffold |
| **No-op** | Terminate early. Already present as `allow_no_modification` (`environment.py:112`) |

**The vocabulary is the "only already-studied molecules" constraint.** Fragments
come from real chemistry — BRICS decomposition of the target's known inhibitors,
plus common medicinal-chemistry groups, ranked by frequency and truncated. Every
product is a novel molecule assembled entirely from precedented pieces. This buys
synthesizability more directly than SA score does, and it is a constraint on
*construction*, so it cannot be gamed the way a scoring term can.

The applicability-domain penalty (see Reward) still applies on top. The vocabulary
constrains what gets built; the domain check constrains what scores well.

### The cost: candidate-set explosion

Atom-level, the per-step candidate set is tens to low hundreds. Fragment-level it
is `attachment_points × |vocab| × fragment_attachment_atoms` — a 30-atom molecule
against a 100-fragment vocabulary is thousands of candidates, and **every one is
encoded and scored by the GNN at every step of every episode**. This is the main
performance risk in the project and it is worth measuring on day one.

Levers, in the order to reach for them:

1. Cap the vocabulary — start near 50 fragments, not 500
2. Restrict attachment points to designated decoration sites, not every free valence
3. Dedup by graph key *before* scoring, not after
4. Only if still too slow: sample a fixed-size candidate subset per step — but this
   changes the MDP and gives up the argmax guarantee, so it is a last resort

Episodes get shorter too. Atom-level needs ~40 steps to grow a substituent that
fragment-level attaches in one, so `max_steps` starts around 5–8 — and the
discount has to be re-tuned to match, or the terminal reward is weighted wrong.

## Featurization

**Atom features (node)**

| Feature | Encoding |
|---|---|
| Atomic number | one-hot over allowed set (C, N, O, F, S, Cl, Br + other) |
| Degree | one-hot 0–5 |
| Formal charge | one-hot −2..+2 |
| Hybridization | one-hot sp, sp2, sp3, sp3d, sp3d2 |
| Aromatic | bool |
| Total num H | one-hot 0–4 |
| In ring | bool |
| Ring size membership | bools for 3,4,5,6,7 |
| Chirality tag | one-hot 4-way |

**Bond features (edge)**

| Feature | Encoding |
|---|---|
| Bond type | one-hot single/double/triple/aromatic |
| Conjugated | bool |
| In ring | bool |
| Stereo | one-hot 6-way |

**Graph-level** (concatenated after pooling): steps remaining (normalized) —
**required**; optionally heavy-atom count and Lipinski descriptors.

`hyp.py` restricts `atom_types = ["C", "O", "N"]`, where it serves double duty as
both the featurization alphabet and the atom-addition action space. With
fragment-level actions those two roles separate: `atom_types` becomes featurization
only, and the action space is sized by the fragment vocabulary instead. Widen the
featurization alphabet freely for drug-like chemistry (F, S, Cl, Br at minimum) —
it costs input width, not candidate count.

Ablate the feature set once the pipeline runs. Feature choice is worth a few
points; it is not where the project succeeds or fails.

## Reward — the surrogate

**The naive version of the original plan doesn't work: there is no IC50 to look up
for a molecule that doesn't exist yet.** What we build instead:

**pIC50 is the reward.** Drug-likeness terms are modifiers on it, never a
substitute for it — if the surrogate is untrustworthy the answer is to fix the
surrogate, not to lean harder on QED.

1. Pull BindingDB IC50 for the chosen target.
2. Clean: single target, standardize to nM, drop qualified values (`>`, `<`),
   convert to pIC50 = −log10(IC50 [M]), median-aggregate duplicate compounds.
3. Train a GNN regressor on pIC50. **Scaffold split**, not random — random split
   massively overstates performance here.
4. That regressor's prediction is the RL reward.

**Seed scaffolds are held out of the regressor's training set.** Scaffold split and
applicability domain otherwise pull opposite ways: the split that makes the reported
MAE honest is the one that puts our seeds out-of-domain. Holding them out costs
reward accuracy at step 0 — the one molecule whose real pIC50 we already know. To
test the alternative, move them back into train and re-run the Step 5 known-actives
ranking test; nothing else changes.

### The failure mode to design around

RL will find adversarial examples of our own regressor. It optimizes the
surrogate, not binding. Mitigations, all cheap, all needed from day one:

- **Applicability domain** — penalize or zero the reward when max Tanimoto
  similarity to the regressor's training set falls below a threshold.
- **Ensemble uncertainty** — 3–5 regressors, different seeds; subtract
  `λ · std(predictions)`. Pessimism under uncertainty.
- **Reward clipping** — cap predicted pIC50 near the max observed in training.
  There is no molecule 10 logs better than everything known.
- **Docking spot-check** — Vina/smina on final top hits. Not in the loop, too
  slow; an independent check that hits aren't nonsense.

## Target selection

Two targets, two jobs:

**Sanity target — GSK3β (or DRD2).** Ready-made oracle in
[Therapeutics Data Commons](https://tdcommons.ai) with published baselines. Its
only purpose is to answer *is my RL loop broken, or is my surrogate broken?*
Without it, that ambiguity eats weeks.

**Real target — EGFR or JAK2.** Dense IC50 coverage in BindingDB, extensively
characterized, large body of known inhibitors to validate against. Kinases have
well-defined pharmacophores, so a chemist can eyeball plausibility.

Check actual row counts before committing — pick whichever of EGFR / JAK2 / CDK2 /
ABL1 has the most clean, unqualified IC50 records.

**Seeds.** Several distinct chemotypes from that target's known inhibitors, chosen
to span the scaffold classes actually present in the data. Enough that per-seed
results can be compared against each other; few enough that each gets a real
compute budget. Cluster the actives by Murcko scaffold and take a representative
from each of the largest clusters.

## Lipinski — filter vs. penalty

A hard filter gives zero gradient signal and can starve the agent (every candidate
rejected → no learning).

- **Training:** soft penalty, e.g. reward × `exp(-α · num_violations)`.
- **Evaluation:** hard filter. Report raw and Lipinski-passing hit rates.

Add **SA score** (synthetic accessibility) alongside — Lipinski alone happily
admits molecules nobody can synthesize, and in practice that's the more common
failure than Ro5 violation.

## Algorithm ladder (simple → complex)

| Tier | Method | Purpose |
|---|---|---|
| 0 | Random / greedy over candidate set | Baseline. If RL doesn't beat this, something is wrong. |
| 1 | **DQN** (double DQN + bootstrapped heads) | What MolDQN is. Already works on QED. |
| 2 | **PPO** — policy = softmax over scored candidates | Same encoder, on-policy, usually better sample efficiency |
| 3 | Multi-objective: pIC50 + QED + SA + Lipinski | Scalarized, or Pareto |
| 4 | **GFlowNet** | Samples proportional to reward → *diverse batch* |

Tier 4 is arguably the correct endpoint: we want 50 diverse plausible analogs for
a chemist, not one argmax molecule. Argmax methods mode-collapse. Reach it only
after the reward is trustworthy.

## ZINC pretraining — masked attribute prediction

The encoder is pretrained on ZINC so it starts with a picture of the molecular
landscape rather than inferring one from ~5–10k labelled points.

**Objective: masked atom / attribute prediction (AttrMask).** Mask a fraction of
atoms' features and predict them from the surrounding graph. No labels needed, and
it forces the encoder to learn local chemical context — which is exactly the
judgement a fragment-level action space asks of it: *does this group belong at this
position?*

One checkpoint initializes both the pIC50 regressor (Step 4) and the RL encoder
(Step 2). That sharing is the point. Keep the checkpoint under version control
alongside the config and featurization that produced it — a pretrained encoder
fine-tuned on a *different* featurization is a silent, total waste.

**Ordering.** Pretraining sits at Step 3b, not Step 0. It has to exist before the
regressor is trained, but it comes after the pipeline is proven, because a
pretrained encoder that silently fails to load looks exactly like a hard research
problem. Build the thing that works, then improve its initialization.

**Measure the null.** Train the regressor from random init as well. With ~5–10k
clean points and a well-regularized GNN, from-scratch is often competitive; if
pretraining doesn't beat it, say so and drop it.

---

# Part II — Build order, with gating tests

**Premise: almost every bug in this project is silent.** Nothing crashes — the
reward curve is just lower than it should be, and that is indistinguishable from
"this is a hard research problem." Every test below converts a silent failure into
a loud one.

Note the reference tests (`google-research/mol_dqn/chemgraph/optimize_qed_test.py`
et al.) are smoke tests — 10 episodes, no assertions. They would pass on a
completely broken agent. `dqn/molecules_test.py` has real environment assertions
and is worth porting; the rest is not a model to copy.

Stack: `pytest` + `hypothesis`. Mark slow tests `@pytest.mark.slow`; keep the
default run under ~30s.

---

## Step 0 — Determinism

Everything below depends on this. Without it no integration test is trustworthy
and you cannot bisect a regression.

```python
def test_run_is_bitwise_reproducible():
    assert run_episodes(n=5, seed=0) == run_episodes(n=5, seed=0)
```

Seed `random`, `numpy`, `torch`; set `torch.use_deterministic_algorithms(True)`
and `PYTHONHASHSEED`. Candidate sets come back sorted by graph key rather than as a
Python set, because set iteration order over strings moves with `PYTHONHASHSEED` and
would make the argmax depend on the shell. PyG's scatter-based pooling is **not**
deterministic on GPU by default — this test catches that.

## Step 1 — Get MolDQN running on QED

Strip TF and `baselines`, write our own replay buffer, reproduce the published QED
curve. **Nothing else proceeds until this works.**

### Environment invariants

The action space is the foundation — if it emits an invalid molecule, everything
downstream is garbage. Property-based fuzzing over random *action sequences* (not
single steps) is where the real bugs are: run full random episodes from each seed
scaffold, assert invariants after every step, over a few thousand runs.

```python
@given(seed=kinase_seed_scaffolds(), vocab=fragment_vocabularies())
def test_all_actions_produce_valid_molecules(seed, vocab):
    for mol in valid_actions(seed, vocab, cfg):
        Chem.SanitizeMol(Chem.Mol(mol))                # raises on bad valence
        assert len(Chem.GetMolFrags(mol)) == 1         # removal never disconnects
        assert mol.HasSubstructMatch(scaffold_of(seed))
```

| Invariant | Catches |
|---|---|
| Every action sanitizes | valence violations |
| Actions deduplicated by **graph key** | inflated action space, skewed argmax |
| A candidate's key is the same however the edit built it | one molecule under two names |
| **Seed scaffold is a substructure of every state** | de novo drift — the whole premise |
| **Removal never disconnects** (`len(GetMolFrags) == 1`) | orphaned fragments scored as molecules |
| **Removal only ever deletes decorations** | scaffold erosion one atom at a time |
| **Every attached fragment is in the vocabulary** | vocab leak, wrong attachment chemistry |
| **Attach then remove the same fragment → the original graph key** | asymmetric edit logic |
| No-op present iff `allow_no_modification` | early-termination path |
| `allowed_ring_sizes` respected | ring-size off-by-one |
| `allow_bonds_between_rings=False` respected | flag not threaded through |
| Action set non-empty for any valid state | agent deadlock |
| Episode terminates in exactly `max_steps_per_episode` | discount misalignment |

The scaffold-substructure invariant is the highest-value one in this table.
Everything else catches a bug; that one catches the project quietly becoming a
different project.

### Replay buffer (ours, so test it properly)

- FIFO eviction at capacity
- Sampling when `len(buffer) < batch_size`
- Shapes/dtypes of sampled batches
- **No aliasing** — push a state, mutate the original `RWMol`, assert the stored
  copy is unchanged. Real bug, mutable RDKit objects.

### Discount arithmetic

```python
def test_discount_applied_from_steps_remaining():
    # reward * gamma ** (max_steps - steps_taken)
    assert env_reward(qed=0.8, max_steps=5, taken=3) == approx(0.8 * 0.9 ** 2)
```

Hand-compute a 3-step episode. Off-by-one here silently reweights all training —
and it matters more with 5–8 step episodes than with 40, because each step carries
a much larger share of the discount.

### Also
- Target network params have `requires_grad=False`; N polyak updates with the
  online net frozen move target measurably toward online
- Epsilon equals `epsilon_start` at step 0, ≈`epsilon_end` after `epsilon_decay`
  steps, monotonically decreasing

### Done when
```python
@pytest.mark.slow
def test_dqn_beats_random_on_qed():
    rl     = run(agent="dqn",    episodes=2000, seed=0)
    random = run(agent="random", episodes=2000, seed=0)
    assert rl.final_mean_reward > random.final_mean_reward + 0.1
```
**The random baseline is the test.** A DQN that ties random is broken, and a
mediocre-but-nonzero reward curve looks like progress without this comparison.
Then pin `final_mean_reward > 0.90` as a golden regression.

## Step 2 — Swap fingerprint+MLP → GNN encoder

Re-run QED. Should match or beat the fingerprint baseline; if not, the encoder is
wrong. These three bugs are all completely silent.

The GNN reads the same `Chem.Mol` the environment already carries, straight into the
atom and bond feature tables above — there is no intermediate representation to build,
because the state was never a string. When this lands, **delete `featurize.py`'s Morgan
path** rather than keeping it behind a flag: two encoders selected by config is exactly
the branch matrix CLAUDE.md rules out, and the fingerprint baseline stays reproducible
from the git history.

### Permutation invariance — highest-value encoder test

```python
def test_embedding_invariant_to_atom_ordering():
    mol  = NAMED["paracetamol"]          # from the SDF fixture, not a SMILES literal
    perm = Chem.RenumberAtoms(mol, shuffled_indices)
    assert torch.allclose(encode(mol), encode(perm), atol=1e-5)
```
If this fails the model is keying on atom index — trains fine, generalizes badly.

### Batch-vs-single equivalence

```python
def test_scoring_is_batch_invariant():
    solo    = torch.cat([score([c]) for c in candidates])
    batched = score(candidates)                    # 200 candidates, one batch
    assert torch.allclose(solo, batched, atol=1e-5)
```
Catches padding/masking bugs where one candidate's atoms leak into another's
pooled embedding. Use a ragged batch (mix 5-atom and 60-atom molecules). This path
runs every single step here, and fails invisibly.

### Steps-remaining is wired in

```python
def test_steps_remaining_changes_q_value():
    assert score(mol, steps_left=1) != score(mol, steps_left=39)
```
Trivial, but if this feature is dropped during the swap the MDP silently becomes
non-stationary.

### Candidate-set throughput

Not a correctness test, but measure it here or discover it in week four:

```python
@pytest.mark.slow
def test_step_latency_under_budget():
    # realistic worst case: largest seed, full vocabulary
    assert median_step_seconds(seed=largest_scaffold, vocab=full) < 0.5
```
Record the per-step candidate count alongside it. If either number is an order of
magnitude off, go back to the levers in Action space before training anything.

### Also
- Round-trip: `mol → graph → adjacency` matches `Chem.GetAdjacencyMatrix(mol)`
- One-hot blocks sum to exactly 1 (unknown atom type → "other" bucket, never
  all-zeros)
- Feature dimension stable across molecules
- Every parameter gets a non-`None`, non-zero gradient after one backward pass

## Step 3 — Plug in the TDC GSK3β oracle

Confirm the loop optimizes a real bioactivity signal. Reuses Step 1's
beats-random test with the oracle as reward.

## Step 3b — Pretrain the encoder on ZINC (AttrMask)

Here rather than at the end: the checkpoint initializes both the Step 4 regressor
and the RL encoder, so it must exist before the regressor is trained. And here
rather than at the start, because Step 3 is what proves the pipeline works — a
pretrained encoder that silently does nothing is indistinguishable from a hard
research problem.

### The masking actually masks

```python
def test_masked_atom_features_are_not_visible_to_the_head():
    # a lone masked atom with no neighbours carries no signal
    assert accuracy(predict_masked(isolated_atom_graph)) == approx(chance, abs=0.05)
```
If the masked atom's own features leak into its prediction, the task is free, the
loss curve looks great, and the encoder learns nothing. This is the AttrMask bug.

### Checkpoint round-trip

```python
def test_checkpoint_roundtrip_is_exact():
    save(encoder, path); loaded = load(path)
    assert torch.allclose(encoder(mol), loaded(mol), atol=1e-6)
```
A silently failing load is the single most common reason pretraining "doesn't
help." Assert it, and assert the featurization dimensions recorded in the
checkpoint match the RL encoder's exactly — pretraining on one featurization and
fine-tuning on another is a total, silent waste.

### Also
- Held-out ZINC loss decreases, and does not decrease when the graph structure is
  shuffled (control: the task must actually depend on the neighbourhood)
- A linear probe from frozen embeddings to logP beats the same probe on a
  randomly-initialized encoder. Cheap, and the first real evidence pretraining did
  anything.

## Step 4 — BindingDB pIC50 regressor

Standalone deliverable: scaffold split, ensemble, report test MAE / Spearman.

### Leakage tests — run on every dataset rebuild

```python
def test_no_duplicate_compounds_across_splits():
    # By graph key — and by then the key must carry stereochemistry, or two
    # enantiomers with different measured IC50 count as one compound.
    assert not (graph_keys(train) & graph_keys(test))

def test_scaffold_disjoint():
    assert not (scaffolds(train) & scaffolds(test))

def test_seed_scaffolds_held_out_of_training():
    assert not (scaffolds(train) & {scaffold_of(s) for s in seed_scaffolds})
```
BindingDB duplicates compounds heavily across assays. This is *the* leakage
source and it will hand you a beautiful, meaningless R².

### Random-label control

```python
@pytest.mark.slow
def test_shuffled_labels_give_chance_performance():
    assert spearman(train(X, shuffle(y)), test) < 0.15
```
If a model trained on scrambled labels still scores well, the split leaks or a
feature encodes the target. Nothing else catches this as cleanly.

### Overfit 20 molecules — best effort-to-value ratio in ML testing

```python
def test_can_overfit_20_molecules():
    assert train(X[:20], y[:20], epochs=500) < 0.01
```
Seconds to run. If the model can't memorize 20 points, the loss, optimizer, or
forward pass is broken — learn it now, not after a 12-hour run.

### Unit conversion

```python
@pytest.mark.parametrize("ic50_nm,expected", [(1.0, 9.0), (1000.0, 6.0), (0.1, 10.0)])
def test_pic50_conversion(ic50_nm, expected):
    assert to_pic50(ic50_nm) == pytest.approx(expected)
```
nM / µM / M confusion is endemic in BindingDB and shifts every reward by a
constant. Hand-compute these.

## Step 5 — Swap regressor in as reward; add guardrails

The reward is the specification. It deserves real assertions.

### Known-actives ranking — the one that means something

```python
def test_ranks_known_inhibitors_above_random_zinc():
    actives = reward(held_out_known_potent_inhibitors)   # never in training
    decoys  = reward(random_zinc_sample(500))
    assert actives.mean() > decoys.mean() + 1.0          # pIC50 units
    assert roc_auc(actives, decoys) > 0.8
```
Everything else checks plumbing; this checks that the reward means something
chemically.

### Guardrails actually fire
- Applicability domain: molecule identical to a training compound → similarity
  1.0, zero penalty; random exotic structure → large penalty
- Reward clipping: hand a molecule an artificially huge prediction, assert capped
- Determinism: same molecule twice → identical reward (catches dropout left on at
  eval, or nondeterministic pooling)

## Step 6 — Lipinski / SA soft penalties

```python
def test_penalty_decreases_with_violations():
    # hand-picked: 0, 1, and 4 violations respectively
    r = [reward_shaping(m) for m in [aspirin, atorvastatin, cyclosporine]]
    assert r[0] > r[1] > r[2]
```
Use real drugs with known descriptor values — they double as documentation.

## Step 7 — Climb the algorithm ladder

Each tier must clear the previous tier's golden regression before replacing it.

## Step 8 — Pretraining ablation and the final report

Retrain the Step 4 regressor from random init and compare against the pretrained
one on the same scaffold split. If pretraining doesn't win, say so and drop it — a
clean null result costs nothing to report and a lot to hide.

Then the actual deliverable, per seed scaffold: top-k analogs with predicted pIC50,
the seed's own pIC50 for reference, Lipinski and SA status, Tanimoto to seed, and a
docking spot-check on the best few. Diversity and novelty reported alongside top-k,
never instead of it.

---

## Testing conventions

**Do not test:** exact float outputs of the network; RDKit's own correctness;
anything needing a full training run in the default path; convergence *speed*
(too noisy across seeds to assert on).

```bash
pytest -m "not slow"          # every commit, ~30s
pytest                        # nightly, includes training-dependent tests
```

---

## Open questions

- ~~**Are the seed scaffolds in the regressor's training set?**~~ Decided: held out.
  See [Reward](#reward--the-surrogate). Revisit if step-0 reward proves too noisy to
  rank early actions.
- **Stereochemistry in the state key.** `graph_key.canonical_hash` is constitutional
  today: L- and D-alanine share a name. Nothing in the atom-level action space can
  create a stereocentre, and dropping chirality from the canonical ranking is what
  keeps the key stable across an SDF round trip. Step 4 brings in real inhibitors
  where the configuration is part of the compound, so the key needs a stereo layer by
  then — and the leakage tests above depend on it.
- **Fragment vocabulary source and size** — BRICS over the target's actives, a
  general med-chem group list, or both. Start near 50, measure the candidate-set
  cost from Step 2, grow only if the budget allows.
- **Similarity constraint to seed** — scaffold preservation is a hard floor. The
  question is whether we also want a Tanimoto ceiling on top of it, to stop the
  agent from hanging a large fragment off every position. `target_sas.py` has the
  machinery.
- **Evaluation protocol** — top-k reward is standard but rewards mode collapse.
  Report diversity + novelty alongside, per seed.
