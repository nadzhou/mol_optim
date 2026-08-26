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
torch. PyTorch Geometric was in this plan and is not used — see [Step 2](#step-2--swap-fingerprintmlp--gnn-encoder).

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

### Measured, before anything was built on it

**The vocabulary.** BRICS over the 5,392 EGFR compounds at pIC50 ≥ 7.0 gives 13,496
usable fragment occurrences across 879 distinct fragments, once two kinds of piece are
set aside: 21,753 occurrences with zero or two-plus open ends, which are linkers and
cores rather than decorations, and 379 single-attachment pieces larger than 12 heavy
atoms — the largest is 33, which is not a substituent, it is another molecule. **The top
50 cover 78%** of what is left, so the plan's "start near 50" is the right size; the top
25 cover 67% and the top 100 only reach 86%.

`data/egfr_fragments.sdf` is that vocabulary, under version control because it *is* the
action space and a run is not reproducible without it. It reads like a substituent list
a chemist would write: methoxy, acrylamide, phenyl, dimethylamino, chlorofluorophenyl,
tert-butyl, N-methylpiperazine, trifluoromethyl, morpholine, cyclopropyl, methylsulfonyl.
Fragments are hashed with the BRICS dummy still attached, so ortho- and para-fluorophenyl
are two entries rather than one — collapsing them would delete the regiochemistry the
vocabulary exists to carry.

**None of the 50 contains a non-aromatic N–N bond.** That is the Step 5 finding closed by
construction rather than by a penalty, and it is asserted in the test suite rather than
noted here.

**The cost, and lever 2.** Every free valence against 50 fragments:

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| heavy atoms | 19 | 44 | 27 | 21 | 35 |
| candidates, every free valence | 500 | 750 | 650 | 450 | 700 |
| per step: enumerate + dedup + score | 212 ms | 558 ms | 351 ms | 194 ms | 430 ms |
| candidates, aromatic C–H only | 400 | 250 | 450 | 350 | 300 |
| per step | 164 ms | 175 ms | 248 ms | 138 ms | 172 ms |

The unrestricted version **misses the 0.5 s budget** on the largest seed, before removal
actions exist and before the molecule grows over an episode. So lever 2 is taken:
attachment is restricted to aromatic C–H, one site per symmetry class of the state's
canonical ranking. That is the dominant real derivatization move, it holds every seed
under 250 ms, and it gives up decoration at aliphatic positions and at exocyclic N–H —
N-alkylating a piperazine is real chemistry this action space cannot do. Revisit if the
agent looks starved for moves.

One surprise worth recording: **deduplication costs more than scoring.** 68 to 124 ms
against 58 to 94 ms, because `graph_key.normalize` kekulizes and re-sanitizes every
candidate, and that runs 250 to 550 times a step. It buys little here — the raw and
deduplicated counts differ by 0 to 18% — but it is a correctness requirement, not an
optimization, so it stays until something cheaper is proven equivalent.

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
**required**; heavy-atom count, taken because mean pooling is size-invariant and QED
depends strongly on size; Lipinski descriptors, not taken. Both are divided by
`max_steps_per_episode`, which keeps the atom count near 1.0 as well — an episode adds
at most one atom per step.

Every categorical carries a trailing "other" bucket, including the ones the table above
writes as fixed-width. An atom the tables do not name then lands in a real column
instead of an all-zero row, which reaches the network as "no atom" and trains fine.

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
  slow; an independent check that hits aren't nonsense. **Measured, and it does not
  work on this target** — see below.

### The docking spot-check was built and does not rank EGFR compounds

Every number in this project traces back to one GNN trained on BindingDB, so an
orthogonal signal was worth having. AutoDock Vina 1.2.7 against 1M17 — the EGFR kinase
domain with erlotinib bound — receptor typed by Open Babel, ligands prepared by Meeko
straight from an RDKit `Mol`, box 22 Å on the co-crystal ligand's centroid.

**The setup is right.** Redocking erlotinib puts the top-scored pose **1.28 Å** from the
crystal pose, against a conventional 2.0 Å pass, and the correct pose is also the
best-scored one. 6 s a ligand at exhaustiveness 16.

**The score still carries no usable signal.** From
`python -m mol_optim.docking runs/pilot_pic50_seed0_top.sdf --controls`:

| | Vina score, kcal/mol | ligand efficiency | heavy atoms |
|---|---|---|---|
| Step 5 top 12 | **−8.84** | −0.374 | 23.8 |
| 15 EGFR compounds, pIC50 ≥ 9 | −7.69 | −0.353 | 22.0 |
| 15 EGFR compounds, pIC50 ≤ 5.5 | **−7.93** | −0.361 | 22.5 |
| 15 ZINC molecules | −7.72 | −0.317 | 24.9 |

**The weak binders score better than the potent ones.** Three and a half logs of measured
difference, and Vina puts the weaker half 0.24 kcal/mol ahead. Across all 30 compounds,
spanning pIC50 4.0 to 11.2, score against measured pIC50 is Spearman +0.26 — and the sign
is the wrong way round, because Vina is negative-is-better. Ligand efficiency, the
standard size correction, does not rescue it. Size explains part of the spread — score
against heavy-atom count is Spearman −0.41 — but not all. (Vina is stochastic; a separate
run of the same 30 gave +0.31. The qualitative result is stable, the second decimal is
not.)

So the check cannot say whether the Step 5 molecules are real — it cannot say that about
known inhibitors either. Note the Step 5 molecules score *best* of the four groups, which
would have read as vindication if the controls had been left out. That is what the
controls are for.

What this does and does not license. It is one rigid receptor, one structure, stock Vina,
30 compounds, protonation assigned crudely at pH 7.4 — not a claim that docking is
useless, and not a reason to skip it at Step 8. It is a measurement that the spot-check
as specified would have returned a confident number meaning nothing. If an orthogonal
signal is wanted, the next things to try are an ensemble of EGFR structures, a rescoring
function, or more compounds — each of which has to clear this same ranking test before
anything is built on it.

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

## Synthetic accessibility — a penalty, never a filter

A hard filter gives zero gradient signal and can starve the agent (every candidate
rejected → no learning). This is not hypothetical here: the applicability domain *is*
a hard filter, and at 40 atom-level edits it zeroed the reward on every rollout from
every seed. See [Step 5, Measured](#the-episode-budget-is-the-guardrail).

- **Training:** soft penalty on **SA score** (synthetic accessibility), applied to the
  predicted pIC50 the way the domain penalty is — reward × `exp(-α · max(0, sa - target))`,
  so a molecule that is already easy to make pays nothing.
- **Evaluation:** report the SA distribution alongside top-k, not a single number.

**Lipinski is out of scope.** Ro5 is a filter over four descriptors a chemist reads
off the final table anyway, and it is not the failure this project produces. Step 3
produced hemiaminals and N-hydroxylamines — structures that fall apart in water, and
that pass Ro5 comfortably. SA is aimed at that failure; Ro5 is not.

## Algorithm ladder (simple → complex)

| Tier | Method | Purpose |
|---|---|---|
| 0 | Random / greedy over candidate set | Baseline. If RL doesn't beat this, something is wrong. |
| 1 | **DQN** (double DQN + bootstrapped heads) | What MolDQN is. Already works on QED. |
| 2 | **PPO** — policy = softmax over scored candidates | Same encoder, on-policy, usually better sample efficiency |
| 3 | Multi-objective: pIC50 + QED + SA | Scalarized, or Pareto |
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
    rl     = run(agent="dqn",    episodes=5000, seed=0)
    random = run(agent="random", episodes=5000, seed=0)
    assert rl.final_mean_reward > random.final_mean_reward + 0.1
```
**The random baseline is the test.** A DQN that ties random is broken, and a
mediocre-but-nonzero reward curve looks like progress without this comparison.

**Measured**, 5000 episodes at seed 0, published `bootstrap_dqn.json` config:
DQN **0.895**, random **0.145**, best single molecule 0.943. The golden regression is
pinned at 0.85 — below the measured number to survive refactors, not to leave room for
a regression.

Two notes on getting there. The defaults in `MolDQN-pytorch/hyp.py` are not the
published ones and cost 0.19 of QED: gamma 0.95 where the environment already discounts
by steps remaining (double discounting), a gradient step every 20 environment steps
instead of every 4, 3- and 4-membered rings allowed, a 1M-transition replay buffer
instead of 5000, and no gradient clipping. And the published 0.94 *mean* uses double Q
over 12 bootstrapped heads, which this does not implement — a single-head, single-
estimator DQN lands at 0.895 with the curve still climbing at episode 5000.

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

### Measured

5000 episodes at seed 0, the same config as Step 1: GNN **0.896**, fingerprint baseline
**0.895** (`runs/dqn_graph_mse.csv`), random **0.146**. Best single molecule 0.928
against the fingerprint run's 0.943.

The means are a tie — 0.001 apart is noise. The difference is spread. Over the last 500
episodes the GNN's per-episode standard deviation is 0.047 against the fingerprint run's
0.087, and 0.4% of its episodes fall below 0.5 against 1.2%. It is also narrower: 302
distinct molecules in those 500 episodes against 353. Steadier and more mode-collapsed
are the same fact.

The encoder does that with 56k parameters against the MLP's 2.7M, a factor of 48.

Throughput: 5.2 ms median to featurize and score a 46-candidate set, against the 0.5 s
budget above. Over a run the rate falls from 70 to 23 steps/s as the agent builds larger
molecules and the candidate sets grow with them.

The molecules are still nonsense — fused strained heterocycles, N-hydroxyls, enol
ethers, one aminooxazole motif repeated across most of the top 12. That is QED, not the
encoder. QED scores descriptors and has no term for strain, stability, or
synthesizability, and an agent free to build any graph one atom at a time will find the
corner where it peaks. The fragment vocabulary is the fix, because it constrains
construction rather than scoring.

### Three decisions taken here

**Plain torch, not PyTorch Geometric.** Message passing is one `index_add_` for the
neighbour sum and another for pooling, roughly 40 lines in `encoder.py`. The repo
already had the ragged machinery — the segment max over candidate sets — so PyG would
have saved little, and its scatter kernels are nondeterministic on GPU, which is exactly
what Step 0 exists to prevent. Revisit if a later step needs an operator that is painful
to write by hand.

**Codes in the buffer, one-hot at the network.** Atom and bond features are stored as
int8 codes (13 bytes an atom against 48 for the one-hot) and expanded by one vectorized
scatter where a batch enters the network. The buffer holds a whole candidate set per
transition, roughly 200k graphs at capacity.

**Mean pooling, with the atom count as a graph feature.** Sum pooling makes the
embedding grow with the molecule and swamps the per-atom signal; mean pooling throws
size away entirely, and QED depends on size. So: mean pool, and hand the count to the
head separately.

## Step 3 — Plug in the TDC GSK3β oracle

Confirm the loop optimizes a real bioactivity signal. Reuses Step 1's
beats-random test with the oracle as reward.

The oracle is a random forest of 100 trees over a 2048-bit ECFP4 fingerprint, trained on
ExCAPE-DB GSK3β actives and published through Therapeutics Data Commons. Its only job is
to answer one question: when a reward curve is flat, is the loop broken or is the reward
broken? The oracle is fixed and published, so on this reward a flat curve is the loop.

Step 2 ended with QED at 0.896 and a top-12 of fused strained heterocycles — molecules
that score well and mean nothing. QED has no term for whether a structure can exist, so
that outcome is the reward's, not the agent's. This step swaps in a reward that at least
scores a real target.

### The reward is TDC's, to 2.3e-8 — the gate

`tests/fixtures/gsk3b_reference.sdf` holds 53 molecules with the score PyTDC 1.0.0 gives
them, computed in a throwaway venv on scikit-learn 1.2.2: 14 ZINC molecules chosen to
spread across the oracle's output range, 24 molecules from random rollouts of this
environment, and the drug fixtures. `test_matches_tdc_on_the_reference_set` re-scores
all 53 with our own forest walk.

This is the gate because every way of getting it wrong returns a number rather than an
error. Radius 3 instead of 2, count bits instead of binary, a walk that stops one level
above the leaf and reads an internal node's class ratio — each of those hands the
training loop a plausible float in [0, 1] and the run quietly optimizes something else.

### The oracle is vendored, not imported

`fetch_gsk3b.py` downloads TDC's 28 MB pickle once, verifies it against a pinned SHA-256,
and writes `models/gsk3b_forest.npz` — 385,384 nodes as five flat arrays, 2.0 MB.
`oracle_gsk3b.py` walks that by hand: 100 trees stepping together as one array of node
indices, 78 rounds deep, 183 µs a molecule against the 15-40 ms a training step already
costs.

Three reasons not to call PyTDC at runtime. Its oracle takes a SMILES string, and
nothing in this loop writes a molecule as text. It pins `scikit-learn==1.2.2` and
`numpy<2` against this venv's numpy 2.5, because the published pickle only unpickles
under the scikit-learn that wrote it. And the pickle stays a black box that has to be
re-executed on every run, where the arrays are inspectable and version-free.

The conversion never imports sklearn: the unpickler maps every sklearn class to an inert
stub that collects the state dict it is handed, so what comes back is numpy arrays. That
is what makes unpickling a 28 MB download safe — a pickle can only construct what
`find_class` returns.

Every split in the forest turned out to be a bit test — threshold exactly 0.5 over a 0/1
feature — which is why the walk is three lines. `fetch_gsk3b.py` asserts it rather than
assuming it.

### Also
- Every walk reaches a leaf: after `depth` rounds from any root, on random bit vectors,
  the node must be one whose children are itself. Catches a `depth` that is one short.
- Atom numbering does not change the score.
- A molecule the environment built scores the same after an SDF round trip. The
  reference fixture reaches the test through a file; training scores the RWMol directly,
  and if perception differed between those two the fixture would pin the wrong numbers.

### Measured

5000 episodes at seed 0, the same config as Step 2 with only the reward changed: DQN
**0.610** final mean against a random floor of **0.077**, best single molecule 0.66 at
28 heavy atoms, in 7653 s. For scale, 3000 ZINC molecules average 0.029 on this oracle
and the best of those 3000 scores 0.51.

The curve climbs the whole way: 0.090 over the first 500 episodes, 0.413 over episodes
2000-2500, 0.584 over the last 500. Under 0.1 is where 60% of the first 500 episodes
land and none of the last 500.

The loop optimizes a real bioactivity model, which is what this step existed to
establish. The next paragraph is the part worth reading.

### The oracle gets gamed, and how it gets gamed is the finding

All 12 top molecules share one core: two or three aminopyrazoles joined by NH bridges.
That much is a real motif — aminopyrazoles hinge-bind and run through published kinase
inhibitor series. What hangs off it is not: N-hydroxylamines in 12 of 12 molecules, 23 in total;
hemiaminals in 11 of 12, 36 in total; aminals in 11 of 12, 15 in total. A hemiaminal
falls apart in water. The agent found the fingerprint bits the forest votes on and built
the cheapest structure that sets them.

The two rewards are close to orthogonal at their own optima. The GSK3β top 12 score
0.05-0.14 on QED; the Step 2 QED top 12 score 0.02-0.10 on the oracle. Neither reward
constrains what the other measures, and the molecules that maximize either one are
molecules no chemist would order.

This is the failure mode [Reward](#the-failure-mode-to-design-around) predicts for the
Step 4 pIC50 regressor, arriving two steps early against a published model that we did
not fit. It carries two consequences. The guardrails listed there — applicability
domain, ensemble standard deviation, reward clipping — are not optional extras for Step
5. And the fragment vocabulary is the load-bearing fix, because a reward term can be
gamed by construction while a vocabulary constrains construction itself: no edit can
produce a hemiaminal if no fragment contains one.

Diversity fell as the reward rose: 82 distinct molecules over the last 500 episodes,
against 302 for the Step 2 QED run. An argmax policy at epsilon 0.01 collapses onto one
motif, which is the argument for tier 4 (GFlowNet) in the ladder above.

One difference to note in the run log: three RDKit warnings, "could not find number of
expected rings", where neither QED run produced any. RDKit fell back to an approximate
ring finder on three molecules, so the ring-size features on those three are
approximate. Which molecules, and why this reward reaches them, is not chased down —
the top 12 here carry 2 to 3 rings against the QED top 12's 2 to 12, so it is not simply
that these are more fused.

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

### Three decisions taken here

**The mask is an all-zero atom feature row, not a mask column.** A dedicated "masked"
column would widen the encoder's input by one, and an encoder the RL network cannot load
is the exact failure this step exists to avoid. The all-zero row also turns the leak test
from a statistical one into an exact one: two lone masked atoms of different elements
have no neighbours and identical inputs, so their logits are equal bit for bit, and the
assertion is `torch.equal` rather than an accuracy near chance.

**The target is the element, and the head is one linear layer.** A deeper head can name
a masked atom from a representation the encoder never had to make chemical sense of, and
it is the encoder that gets kept.

**Masking does not hide the atom's degree.** Its bonds stay in the edge list, so the
number of them and their types still reach it. That is the task working as intended —
degree is context — and it is why the shuffled-context control below lands above the
prior on accuracy: knowing an atom has four single bonds says carbon without knowing
anything else about the molecule.

### Measured

10 epochs over 249,455 ZINC molecules, 5000 held out, seed 0, 409 s
(`runs/pretrain_zinc.csv`, drawn in `runs/pretrain_curve.png`). Masked-element
cross-entropy on held-out molecules **0.181** against a marginal prior of **0.896**,
accuracy **0.929** against the prior's 0.736. The checkpoint is
`models/zinc_encoder.pt` — 43,456 parameters, 183 KB, with the featurization hash
inside it.

One pass does most of the work: held-out loss is 0.226 after epoch 1 and 0.181 after
epoch 10, and accuracy moves 0.914 to 0.929 over those nine further epochs.

The control behaves: the same molecules with the same atoms masked, atom features dealt
out to random positions, score 0.894 at epoch 10 against the prior's 0.896, and 0.78 to
0.89 across the ten epochs. The task depends on the neighbourhood. Its accuracy stays
at 0.81 against the prior's 0.736, which is the degree leak named above, visible.

**Frozen probes say the representation got worse. Fine-tuning says the initialization
got better.** Both were measured, and they disagree, so both are here.

Linear maps from frozen mean-pooled embeddings to seven RDKit properties, fitted on 2500
held-out molecules and scored on 2500 more, pretrained against randomly initialized:
Crippen logP 0.605 against 0.574 — and then six losses. Rotatable bonds 0.298 against
0.786. Molecular weight 0.300 against 0.626. TPSA 0.544 against 0.746. Aromatic rings
0.822 against 0.921. QED 0.168 against 0.270. Fraction sp3 0.941 against 0.976.

Fine-tuned, the order reverses. A stand-in for the Step 4 regressor (`finetune_zinc.py`)
— the encoder plus a 64-unit head, all weights trainable, 3500 held-out molecules to
train on and 1500 to score, everything identical between the two runs except the
encoder's starting weights — gives the pretrained encoder every one of four comparisons. Crippen logP test MAE 0.239
against 0.360 at seed 0 and 0.302 against 0.506 at seed 1; QED 0.0753 against 0.0848 and
0.0795 against 0.0845.

The reading: AttrMask pushes node embeddings toward what its own task needs, which is
local chemical identity, and a mean pool of that carries less linearly readable size and
polarity than a random projection of the raw atom features does. A frozen linear probe
measures the pooled representation. What this step produces is a starting point for
gradient descent. Those are different questions, and only the second one is the one the
project asks — so the probe stays in the code as a cheap sanity check and stops being
the evidence.

What is still not established is the part that matters. Those proxies are ZINC molecules
with computed labels, one distribution, noise-free targets. Step 4's regressor is
BindingDB compounds with measured pIC50 — 10,850 points, a different distribution, a
target with real assay noise — and it re-ran this comparison against the from-scratch
null. It came out the same way: test MAE 0.806 against 0.868, Spearman 0.642 against
0.582. See [Step 4, Measured](#measured-2). Step 8 runs it for the RL agent, where
`train_dqn.py --pretrained-encoder` is already wired and tested.

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

### Decisions taken here

**BindingDB's own dated snapshot, not TDC's processed copy.** TDC publishes a
`bindingdb_ic50.csv` on the same Harvard dataverse as the GSK3β oracle and ZINC, which
would have been the consistent choice, and it hands back a bare nanomolar number. The
qualifier is gone. A row recorded as ">10000 nM" — the assay ran off the end of its
range and nobody found the number — arrives as 10000, becomes a pIC50 of 5.0, and is
indistinguishable from a real measurement of 10 µM. There are 5,720 such rows for EGFR
against 23,461 usable ones. So: `BindingDB_All_202608_tsv.zip`, 593 MB, pinned by the
MD5 BindingDB publishes beside it, and 9 GB of TSV streamed once.

**One construct, not one UniProt id.** P00533 covers 51 distinct EGFR constructs in this
snapshot. Wild type is 16,512 of the rows; the rest are T790M, L858R, C797S and
truncations. Pooling them puts one compound's wild-type and T790M numbers under one
label, and telling those apart is the entire purpose of a third-generation EGFR
inhibitor.

**A second, stereo-aware key.** `graph_key.canonical_hash` stays constitutional and
stays the RL loop's name for a state. `stereo_hash` separates configurations and is what
the dataset deduplicates and splits on, because two enantiomers here are two compounds
with two measured IC50 values. It cost one real bug to get right: a double bond with
undefined geometry reads back from a molblock as STEREOANY where a SMILES said
STEREONONE, and both mean "nobody said", so 185 of 10,855 compounds changed name on the
trip to disk. Five macrocycles change name even with that fixed — their geometry does
not survive being drawn in two dimensions — and the ingest reads its own output back,
finds them, and drops them rather than leaving the leakage tests unable to reason.

**The split is three ways, and validation comes out of training by scaffold.** Train and
test by scaffold; the training half scaffold-split again for the epoch to stop at.
Choosing that epoch on the test set is the oldest way to report a number that does not
survive contact with new data.

**Seeds are chosen here, not in the RL run, because the split has to know about them.**
Largest active scaffold clusters first, then two filters that came from reading what it
picked without them: the representative must have more than one measurement agreeing
within a log — the first pick otherwise had a 2.73-log spread across five measurements,
and the final report quotes each seed's own pIC50 — and a cluster is skipped if its
representative is within 0.6 Tanimoto of a seed already taken, because two of the first
five differed only in where a ring nitrogen sat.

### Measured

**The dataset.** 3,234,499 rows scanned, 23,209 of them EGFR wild type on a single
chain. 4,465 qualified values dropped, 1 unusable, 1 unreadable, 5 dropped for changing
name on the way to disk. What comes out is **10,850 compounds**, pIC50 1.60 to 11.52,
2,709 of them measured more than once, across 3,861 Bemis-Murcko scaffolds — the largest
holding 576 compounds and 2,527 holding one.

**The labels argue with each other.** Among compounds measured more than once the median
spread is 0.56 log units, and 822 of the 2,709 disagree by more than a log. The worst is
8.54 — the same compound, the same target, eight orders of magnitude apart. That number
is the floor the regressor is working against, and it is why the error below should be
read next to 0.28, the half-width of a typical repeated label, rather than next to zero.

**The regressor.** Five networks, scaffold split 7,378 / 1,302 / 2,170 with the five seed
scaffolds held out of training, early stopping on validation, 11 minutes a run
(`runs/regressor.png`).

| | pretrained on ZINC | random init | training mean |
|---|---|---|---|
| test MAE | **0.806** | 0.868 | 1.138 |
| test RMSE | **1.052** | 1.117 | — |
| test Spearman | **0.642** | 0.582 | — |
| single models, MAE | 0.828 ± 0.017 | 0.883 ± 0.004 | — |

**Step 3b's question, answered on measured data: pretraining helps.** 0.806 against 0.868
MAE and 0.642 against 0.582 Spearman, and every member of the pretrained ensemble beat
every member of the other on validation. This is the comparison the ZINC probes could
not settle — different distribution, real assay noise, a target nobody computed from the
graph — and it lands the same way the ZINC fine-tuning proxy did.

Model error against label noise: MAE is 0.777 on the 489 test compounds with repeated
measurements and 0.814 on the 1,681 with one. On the 233 whose repeats agree within 0.5
it is 0.760. So the noisy labels are not where the error is; the model is about 2.5x the
label half-width across the board, and the scatter shows why — predictions compress into
6 to 8.5 while measurements run 2 to 11.

### The guardrails Step 5 assumes, measured before Step 5 leans on them

The right two panels of `runs/regressor.png` test the two mitigations
[Reward](#the-failure-mode-to-design-around) specifies, and they do not come out equal.

**Ensemble disagreement barely predicts error.** Rank correlation 0.08. Mean absolute
error is 0.70 to 0.86 across the bottom eight deciles of disagreement and only rises in
the top two, to 0.97. `reward - λ·std` as written would spend most of its effect on
noise. As a gate on the top decile it would do something; as a linear penalty it is
mostly decoration.

**Distance from the training set does predict it.** Rank correlation -0.21, and the
decile curve falls monotonically: mean error 1.146 for the tenth of test compounds whose
nearest training neighbour is at Tanimoto 0.41, against 0.653 for the tenth sitting at
0.94. The applicability domain is the guardrail with support behind it, and Step 5
should lean on it rather than on the ensemble spread.

Neither result excuses the third mitigation. Reward clipping near the best measured
affinity costs nothing and needs no evidence to justify: there is no molecule ten logs
better than everything known.

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

### The episode budget is the guardrail

The applicability domain is a hard filter, and this plan says a hard filter starves the
agent. At `max_steps_per_episode = 40` it does. Atom-level edits are small, but forty of
them are not: one random edit takes seed 0 from Tanimoto 0.72 to the training set down
to 0.59, five reach the 0.3 floor, and forty land at 0.16.

| random edits from seed 0 | 1 | 2 | 4 | 6 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|
| median Tanimoto to training set | 0.59 | 0.45 | 0.37 | 0.28 | 0.24 | 0.18 | 0.16 |
| mean reward, pIC50 units | 7.28 | 6.98 | 5.59 | 3.21 | 1.07 | 0.00 | 0.00 |

Over 360 random rollouts at 40 edits — 60 from each of the five seeds and 60 from
scratch — **not one terminal molecule cleared the floor**. Every reward was exactly 0.0.
A 5000-episode run at that budget would have trained on a flat line, and the log would
have looked like a hard research problem rather than a misconfigured one.

So `max_steps_per_episode` is **6** for this reward. That is the number
[Action space](#the-cost-candidate-set-explosion) predicted the fragment vocabulary
would need anyway, arriving early and for a different reason: not that fragment edits
are large, but that the applicability domain is only a few atom-level edits wide.

The discount is unchanged at 0.9, which now weights the first step's reward by 0.9^5 =
0.59 rather than 0.9^39 = 0.02. Intermediate rewards matter under this budget where
they were rounding error under the old one.

### The reward means something chemically

`test_ranks_known_inhibitors_above_random_zinc`, the one test here that is about
chemistry rather than plumbing: 628 held-out actives at pIC50 ≥ 8.0 score a mean reward
of **7.516** against **1.508** for 500 ZINC molecules — a 6.01 log gap and ROC AUC
**0.990**, against thresholds of 1.0 and 0.8.

That number conflates two things, so both were measured. The domain filter zeroes 71.6%
of the ZINC sample and 0.5% of the actives, which is most of the gap. The regressor on
its own, guardrails off, still separates them: actives 7.650, decoys 5.619, a 2.03 log
gap at ROC AUC **0.963**. The regressor does the ranking and the domain filter sharpens
it. Neither is carrying the test alone, which is what had to be established before the
agent was allowed to optimize against it.

### Measured

A pilot, not the deliverable: one seed, 1000 episodes, 6 edits, 292 s
(`runs/pilot_pic50_seed0.csv`, drawn in `runs/pilot_pic50_seed0.png`). Seed 0 is a
4-anilinoquinazoline — the gefitinib chemotype — 19 heavy atoms, measured pIC50 10.00,
predicted 7.38, Tanimoto 0.72 to the nearest training compound.

Three lines on that plot, and the order matters. Random at the same budget is **0.331**.
The seed handed back untouched is **0.738**, which is what the agent collects for taking
the no-op every step. The DQN reaches **0.859** over the last 100 episodes and peaks at
0.925 near episode 800. It crosses the no-op line around episode 510 and stays above it,
so it is finding edits the regressor scores above the lead rather than learning to sit
still. Best single molecule **0.995** — predicted pIC50 9.95 at 24 heavy atoms.

Two things in the curve worth naming. The first 180 episodes run *below* random, down to
0.15: epsilon is near 1.0 and the agent has not yet learned to stay inside the domain,
so most episodes end at zero. And the MSE loss climbs from 0.07 to 0.30 across the run
rather than falling, because the reward the agent reaches keeps growing and the TD
targets grow with it. It flattens after episode 800; a loss that kept climbing would be
the thing to chase.

Re-running the same command reproduced the run exactly — same final mean, same best
molecule, same graph hash — which is Step 0 still doing its job five steps later.

### The agent games the regressor again, and this time the molecule looks plausible

Step 3 found hemiaminals. The audit here started by looking for those, and none of them
are present: hemiaminal 0/12, N-hydroxylamine 0/12, aminal 0/12, gem-diol 0/12. The
drawing is the reason not to stop there.

**The scaffold survives.** All 12 of the top molecules still contain the seed's Murcko
frame, against 66% for 200 random 6-edit rollouts. Part of that is the budget and part is
the agent, and the two numbers say which is which.

**The nitrogen is the tell.** The seed has 4 nitrogens and no N–N bond. Every one of the
top 12 has 7 to 10 nitrogens and **3 to 6 N–N bonds** — aryl hydrazines, chains of three
and four catenated NH, and fused rings carrying three contiguous NH. Random rollouts at
the same budget put an N–N bond in 40% of molecules, so this is a preference the agent
learned, not something the action space hands out. Polyazanes of that kind are not
stable and no chemist would order one.

So the failure mode is the same as Step 3 and the disguise is better. The quinazoline
core is real medicinal chemistry, the decorations are not, and a drawing that looks like
an EGFR inhibitor at a glance is harder to throw out than an obvious hemiaminal.

**The guardrails did not fire on any of it**, and the numbers say why:

| guardrail | setting | what the top 12 did |
|---|---|---|
| applicability domain | zero below Tanimoto 0.3 | 0.42 to 0.70 — comfortably inside |
| ensemble pessimism | −0.5 × spread | spread 0.84 to 1.23, so it removed about 0.45 of a 10.4 prediction |
| reward clipping | 11.10, the best training pIC50 | never engaged |

The domain check works on molecules that are far away, which is what
[Step 4](#the-guardrails-step-5-assumes-measured-before-step-5-leans-on-them) measured it
for. It does nothing about a molecule one hydrazine away from a real inhibitor. And note
where these predictions sit: Step 4 reported that the regressor's test predictions
compress into 6 to 8.5, and the agent is being paid 9.66 to 10.44. It has found the
place where the model extrapolates upward, which is the definition of the adversarial
example [Reward](#the-failure-mode-to-design-around) predicts.

### How much of the gain is the nitrogen?

The top 12 are the 12 best episodes of 1000, so they need not describe the policy. They
do. Two hundred greedy episodes from the trained checkpoint, epsilon at the 0.01 the run
ended on:

- **100% carry at least one N–N bond**, mean 3.0 per molecule
- 198 of 200 carry three or four, and score a mean predicted pIC50 of **8.51**
- the 2 that carry fewer score **7.72**, against the seed's own 7.38

So the policy has collapsed onto one motif, and the whole distance from the no-op line
to 8.50 arrives with nitrogen catenation attached. Step 5's headline — 0.859 against
random's 0.331 — is honest evidence that the loop optimizes its reward. It is not
evidence of lead optimization. Read the two claims separately from here on.

This also settles what to build next. A structural-alert penalty would block the
hydrazine and the agent would price the next motif; the diagnostic that would have
justified building it is the paragraph above, already measured. **The fragment
vocabulary is the fix**, for the reason it was the fix at Step 3 and is now the fix
twice: an alert is a term the agent trades against predicted pIC50, and a vocabulary has
no exchange rate. No fragment drawn from measured inhibitors is a pentazane, so no
sequence of attachments builds one.

SA stays in the plan for what it does measure, and out of the critical path.

## Step 6 — SA soft penalty

```python
def test_penalty_decreases_with_synthetic_difficulty():
    # hand-picked: SA roughly 1.4, 3.0 and 7.9 — one ring, a fused lead, a macrocycle
    r = [reward_shaping(m) for m in [aspirin, atorvastatin, cyclosporine]]
    assert r[0] > r[1] > r[2]

def test_an_easy_molecule_pays_nothing():
    assert reward_shaping(aspirin) == pytest.approx(reward(aspirin))
```
Use real drugs with known descriptor values — they double as documentation. The
second test is the one that matters: a penalty that fires below the target charges
the agent for chemistry it has not done yet, and the reward stops meaning pIC50.

### Measured before building it: SA does not catch what Step 5 found

Run on the Step 5 top 12 before writing the penalty, because a guardrail aimed at the
wrong failure is worse than none — it costs reward and buys nothing.

| | SA score |
|---|---|
| seed 0 | 2.05 |
| Step 5 top 12 | 3.05 to 3.49, mean 3.26 |
| 400 BindingDB EGFR compounds | mean 2.38, 90th percentile 2.66, max 4.24 |
| 400 ZINC molecules | mean 2.98, 90th percentile 4.07, max 5.95 |

**The tetrazane chains score as easier to make than a tenth of the ZINC catalogue.** SA
is a fragment-frequency and complexity score, and a short acyclic N–N–N–N chain is
neither rare-by-fragment nor complex. It does not rank within the group either: the
molecule carrying six N–N bonds scores 3.10 and one carrying three scores 3.35. A
threshold low enough to fire on 3.26 would also fire on real EGFR inhibitors, whose
own scores reach 4.24.

A curated structural alert does catch it. RDKit's Brenk catalogue flags 11 of the 12,
and its `hydrazine` alert alone fires on exactly the seven molecules with a
non-aromatic N–N bond. But the whole catalogue is too blunt to use as a reward term —
it flags 32% of the BindingDB EGFR compounds, and flags the seed itself, for `aniline`.
So: specific alerts, chosen and justified one at a time, not a catalogue.

SA stays worth having for the failure it does measure. It is not the answer to Step 5's.

## Step 7 — Climb the algorithm ladder

Each tier must clear the previous tier's golden regression before replacing it.

## Step 8 — Pretraining ablation and the final report

Retrain the Step 4 regressor from random init and compare against the pretrained
one on the same scaffold split. If pretraining doesn't win, say so and drop it — a
clean null result costs nothing to report and a lot to hide.

Then the actual deliverable, per seed scaffold: top-k analogs with predicted pIC50,
the seed's own pIC50 for reference, SA score, Tanimoto to seed, and a
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
