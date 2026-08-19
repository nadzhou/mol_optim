# CLAUDE.md

Instructions for writing code in this repo. See [plan.md](plan.md) for the project
design, build order, and the tests gating each step.

## Responses: keep them short and plain

Default to a few sentences. Long answers are the exception and need a reason.

- **Lead with the answer.** No preamble, no restating the question.
- **Don't summarize a file you just wrote.** The file is the deliverable; the chat
  message is not a second copy of it. Say what changed and why, in one or two
  lines, and stop.
- **No closing recap** of what was done or which files were touched.
- Skip caveats and alternatives unless they change the decision.
- Prose over bullets for short answers. Don't bullet three sentences.

### Plain words

Write so the sentence can only be read one way. Short words, short sentences, and
the plain name for the thing.

- **Say it straight.** "The split puts our seeds outside the training data" beats
  "the split and the domain check pull opposite ways." If a sentence sounds clever,
  rewrite it.
- **No metaphor for technical facts.** Things do not "pull," "eat," "buy," or
  "starve." Say what happens: what breaks, what gets slower, what gets a wrong
  number.
- **No vague size words.** Not "much better" or "massively overstates" — give the
  number, or say you don't have one.
- **Name the thing, every time.** Repeat "the pIC50 regressor" instead of switching
  to "the surrogate," "the model," "the oracle" in the same paragraph. Synonyms
  read as different things.
- **Don't hedge, and don't oversell.** If a test passed, say it passed. If you did
  not run it, say that. No "should be fine."
- **Spell out a term the first time** you use it, then use it consistently.

Detail belongs in code and in the docs. Chat carries the decision, not the
reasoning behind every part of it. If a full explanation is genuinely needed, ask
whether to expand rather than defaulting to the long version.

## The through-line

Three sources, one claim: **indirection that hides control flow is the main cost
in a codebase.**

- **Locality of Behavior** (Carson Gross) — you should understand what a piece of
  code does by reading that piece of code, without opening other files.
- **Primitives** (Eskil Steenberg) — build a small set of simple, fully understood
  operations and compose everything from them. Optimize for reading and deleting,
  not for writing or extending.
- **Inlining** (John Carmack) — a function called from exactly one place usually
  shouldn't be a function. Sequential operations should read sequentially. A page
  limit on function length is not a real constraint.

This matters more than usual here, because ML bugs are silent (see plan.md Part
II). When a reward curve is wrong, you read the code to find out why. Every layer
of indirection is a place the bug can hide from that read.

---

## Rules

### 1. No global config bags
No `hyp.py` pattern — module-level globals imported and referenced at use sites.
It is action at a distance: changing one value silently alters behavior in files
that never mention it.

Use a frozen dataclass, passed explicitly:

```python
@dataclass(frozen=True)
class Config:
    gamma: float = 0.95
    max_steps: int = 40
    atom_types: tuple[str, ...] = ("C", "N", "O", "F", "S", "Cl")

def run_episode(env, agent, cfg: Config): ...
```

A value used in exactly one place is a literal at that place, not a config field.

### 2. Inline single-call functions
If it's called once, inline it. Especially in the training step — `sample_batch`,
`compute_td_error`, `apply_grads` as three one-call helpers fragment a single
logical operation across three scopes for no benefit. Write the step flat, in
order, so the read matches the execution.

A 150-line training loop that reads top-to-bottom is correct. Do not break it up
to satisfy a length rule.

### 3. Split on the purity boundary, not the length boundary
This is the resolution to "but then how do I test it?"

**Extract** when a function is pure and called from more than one place:
`featurize(mol)`, `reward(mol, cfg)`, `valid_actions(mol, cfg)`, `to_pic50(nm)`.
These are the primitives. They take data, return data, touch no global state, and
carry the project's heavy test coverage.

**Inline** when the code is a sequence of steps in one operation, called once, and
mutates state as it goes. Training loops, data-loading scripts, eval harnesses.

Purity is what makes a boundary worth paying for. Length is not.

### 4. No inheritance for behavior
Do not put the reward in a subclass override that the base class calls. That
inverts control flow — reading `Molecule.step()` doesn't tell you what the reward
is, and reading the subclass doesn't tell you when it fires.

Pass the reward in as a function:

```python
def step(mol, action, reward_fn, cfg): ...
```

Composition over hierarchy, always. Subclassing is acceptable only to satisfy an
external framework's API (e.g. `torch.nn.Module`, `Dataset`).

### 5. No hidden control flow
Banned: metaclasses, `__getattr__` tricks, monkeypatching, decorators that alter
when or whether a function runs, import-time side effects, framework callback
registries. If reading the call site doesn't tell you what executes, it's wrong.

Concretely: **no Lightning, no Hydra, no Ignite.** They invert the training loop
so you can no longer see where the optimizer steps — which is exactly the line you
need to see when debugging a silent ML bug. Plain `torch` + an explicit loop.

### 6. Plain data
Pass `Chem.Mol`, tensors, dataclasses, dicts. Do not write a `Molecule` wrapper
class holding a mol plus cached properties plus helper methods. RDKit's object is
the primitive; wrapping it adds a layer that must be read and kept in sync.

### 7. Justify every dependency
Each one is a black box you cannot debug at 2am. `rdkit`, `torch`,
`torch_geometric` are load-bearing — fine. A library that saves 20 lines is not
worth the opacity. Prefer writing the 40-line replay buffer over pulling in a
TF1-era RL framework (this is already a live issue; see plan.md).

### 8. Name things fully, qualify at call sites
`featurize.atom_features(mol)`, not `from featurize import *` then
`atom_features(mol)`. The prefix tells the reader where to look. Long descriptive
names beat short ones — `predicted_pic50` not `y`, `candidate_mols` not `cs`.

Exception: standard math/tensor idiom where short names are the convention
(`x`, `h`, `logits`, `i`/`j` in a tight loop).

### 9. Comment tensor shapes inline
The shape *is* behavior, and it's invisible in Python.

```python
h = self.gnn(batch)            # [num_atoms_total, hidden]
g = global_mean_pool(h, batch.batch)   # [num_candidates, hidden]
q = self.head(torch.cat([g, steps_left], dim=-1))  # [num_candidates, 1]
```

Comment *why*, never *what*. A comment restating the code is a maintenance
liability.

---

## Python/ML-specific caveats

These principles come from C and web development. Three places they need adjusting:

**Flat does not mean scalar loops.** Carmack's inlining is about control flow in a
compiled language. In Python, a per-element loop over a batch is a 10–100x
slowdown (the reference `agent.py` does exactly this — a Python `for` over
`batch_size` calling the network once per item). Write flat, vectorized code:
one batched tensor op, not a loop that reads sequentially.

**Duplicate experiment variants; don't parameterize them.** Research code grows
variants. The failure mode is one training loop with twelve `if cfg.use_double_q:`
branches — nobody can read any single path through it. Two 120-line loops that
each read cleanly beat one 200-line loop with a flag matrix. Copy the file, change
it, delete it when the experiment dies. Code should be easy to delete.

**There is still a ceiling.** ~150–200 lines is where a Python function stops
fitting in working memory. Past that, look for a genuine pure primitive to extract
— not an arbitrary chop.

---

## Anti-examples in this repo

`MolDQN-pytorch/` is the working base but is not the style model:

| Pattern | File | Problem |
|---|---|---|
| Global config bag | `hyp.py` + `import hyp` throughout | Action at a distance (rule 1) |
| Reward via subclass override | `agent.py` `QEDRewardMolecule._reward()` | Inverted control flow; reward requires reading two files (rule 4) |
| Per-item Python loop over batch | `agent.py` `update_params()` | Correct but ~50x slower than batched |
| TF1-era framework dep | `from baselines.deepq import ...` | Unjustified dependency, and doesn't install (rule 7) |

`dqn.py` is fine as-is — five explicit `nn.Linear` layers, no cleverness, reads in
one pass. Keep that spirit.

When porting code from these repos, restyle it rather than preserving it.

---

## What this style is not

- Not an argument against functions. It's an argument against functions that exist
  only to make a file look shorter.
- Not an argument against testing. The pure primitives (rule 3) are precisely the
  testable surface, and they carry the assertions in plan.md Part II.
- Not permission to write clever, dense code. Explicit and boring beats short.
  Optimize for the reader debugging at 2am, who is you.
