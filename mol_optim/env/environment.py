"""The molecule MDP: substituent-level graph edits.

Identity comes from graph_key.canonical_hash, off the graph, never a SMILES round-trip.
"""

from dataclasses import dataclass
from typing import Callable

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments, graph_key

# Zero occurrences in both ZINC and the measured EGFR set, so no target contains one.
IMPLAUSIBLE = tuple(
    Chem.MolFromSmarts(smarts)
    for smarts in (
        "[CX2;r3,r4,r5,r6,r7]#[CX2]",
        "[CX2;r3,r4,r5,r6,r7](=*)=*",
        "[OX2]-[OX2]",
        "[OX2]-[CX2]#[NX1]",
    )
)


def is_plausible(mol: Chem.Mol) -> bool:
    return not any(mol.HasSubstructMatch(motif) for motif in IMPLAUSIBLE)


@dataclass(frozen=True)
class Result:
    state: Chem.Mol
    reward: float
    terminated: bool


@dataclass
class Episode:
    """Mutable MDP state for one episode. Plain data; step() advances it in place."""

    state: Chem.Mol
    num_steps_taken: int
    valid_actions: tuple[Chem.Mol, ...]


def valid_actions(
    state: Chem.Mol, library: tuple[fragments.Fragment, ...]
) -> tuple[Chem.Mol, ...]:
    """Every molecule one substituent edit away, deduplicated and ordered by hash.

    An action *is* the resulting molecule: the agent scores next states.
    """
    return _deduplicated(fragments.substitutions(state, library))


def _deduplicated(candidates: list[Chem.Mol]) -> tuple[Chem.Mol, ...]:
    """One molecule per graph, ordered by canonical hash so the order is reproducible.

    Normalized first, or one graph reaches the network as two fingerprints.
    """
    by_hash: dict[str, Chem.Mol] = {}
    for candidate in candidates:
        normalized = graph_key.normalize(candidate)
        # After normalize: is_plausible reads ring membership, which sanitize sets up.
        if not is_plausible(normalized):
            continue
        by_hash.setdefault(graph_key.canonical_hash(normalized), normalized)
    return tuple(by_hash[key] for key in sorted(by_hash))


def reset(cfg: config.Config, library: tuple[fragments.Fragment, ...]) -> Episode:
    if cfg.init_mol is None:
        raise ValueError("a substituent action space needs a molecule to start from")
    return Episode(
        state=cfg.init_mol,
        num_steps_taken=0,
        valid_actions=valid_actions(cfg.init_mol, library),
    )


def step(
    episode: Episode,
    action_index: int,
    reward_fn: Callable[[Chem.Mol], float],
    cfg: config.Config,
    library: tuple[fragments.Fragment, ...],
) -> Result:
    """Moves to the candidate at `action_index` and scores the graph it lands on."""
    if episode.num_steps_taken >= cfg.max_steps_per_episode:
        raise ValueError("This episode is terminated.")
    if not 0 <= action_index < len(episode.valid_actions):
        raise ValueError(
            f"No candidate {action_index}; there are {len(episode.valid_actions)}"
        )

    episode.state = episode.valid_actions[action_index]
    episode.num_steps_taken += 1
    episode.valid_actions = valid_actions(episode.state, library)

    steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken
    # Only the terminal molecule counts; earlier ones are discounted by steps left.
    reward = reward_fn(episode.state) * cfg.discount_factor**steps_remaining
    return Result(
        state=episode.state, reward=reward, terminated=steps_remaining == 0
    )
