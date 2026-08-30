"""The molecule MDP: atom-level graph edits, ported from MolDQN.

Identity comes from graph_key.canonical_hash, off the graph, never a SMILES round-trip.
"""

import itertools
from dataclasses import dataclass
from typing import Callable

from rdkit import Chem

from mol_optim import config
from mol_optim.chem import fragments, graph_key

ActionSpace = Callable[[Chem.Mol | None, config.Config], tuple[Chem.Mol, ...]]


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

    state: Chem.Mol | None
    num_steps_taken: int
    valid_actions: tuple[Chem.Mol, ...]


def valid_actions(state: Chem.Mol | None, cfg: config.Config) -> tuple[Chem.Mol, ...]:
    """Every graph reachable from `state` in one edit, deduplicated and ordered.

    An action *is* the resulting molecule: the agent scores next states.
    """
    if state is None:
        candidates = []
        for element in cfg.atom_types:
            single_atom = Chem.RWMol()
            single_atom.AddAtom(Chem.Atom(element))
            Chem.SanitizeMol(single_atom)
            candidates.append(single_atom.GetMol())
        return _deduplicated(candidates)

    mol = state
    periodic_table = Chem.GetPeriodicTable()
    max_bonds = {
        element: max(periodic_table.GetValenceList(element))
        for element in cfg.atom_types
    }
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]
    # Bounded by the bond orders that exist, not by the largest valence in atom_types:
    # sulfur's is 6, and a methane state then asks for bond_orders[4].
    atoms_with_free_valence = {
        order: [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= order
        ]
        for order in range(1, len(bond_orders))
    }

    candidates: list[Chem.Mol] = []

    for order, atom_indices in atoms_with_free_valence.items():
        for atom_idx in atom_indices:
            for element in cfg.atom_types:
                if max_bonds[element] < order:
                    continue
                candidate = Chem.RWMol(mol)
                new_atom_idx = candidate.AddAtom(Chem.Atom(element))
                candidate.AddBond(atom_idx, new_atom_idx, bond_orders[order])
                if Chem.SanitizeMol(candidate, catchErrors=True):
                    continue
                candidates.append(candidate.GetMol())

    for order, atom_indices in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atom_indices, 2):
            # Off a copy, so the SetBondType below cannot reach `mol`.
            bond = Chem.Mol(mol).GetBondBetweenAtoms(atom1, atom2)
            candidate = Chem.RWMol(mol)
            Chem.Kekulize(candidate, clearAromaticFlags=True)
            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue
                upgraded = bond_orders.index(bond.GetBondType()) + order
                if upgraded >= len(bond_orders):
                    continue
                bond.SetBondType(bond_orders[upgraded])
                candidate.ReplaceBond(bond.GetIdx(), bond)
            elif not cfg.allow_bonds_between_rings and (
                mol.GetAtomWithIdx(atom1).IsInRing()
                and mol.GetAtomWithIdx(atom2).IsInRing()
            ):
                continue
            elif (
                cfg.allowed_ring_sizes is not None
                and len(Chem.rdmolops.GetShortestPath(mol, atom1, atom2))
                not in cfg.allowed_ring_sizes
            ):
                continue
            else:
                candidate.AddBond(atom1, atom2, bond_orders[order])
            if Chem.SanitizeMol(candidate, catchErrors=True):
                continue
            candidates.append(candidate.GetMol())

    if cfg.allow_removal:
        for order in (1, 2, 3):
            for existing_bond in mol.GetBonds():
                bond = Chem.Mol(mol).GetBondBetweenAtoms(
                    existing_bond.GetBeginAtomIdx(), existing_bond.GetEndAtomIdx()
                )
                if bond.GetBondType() not in bond_orders:
                    continue
                candidate = Chem.RWMol(mol)
                Chem.Kekulize(candidate, clearAromaticFlags=True)
                downgraded = bond_orders.index(bond.GetBondType()) - order
                if downgraded > 0:
                    bond.SetBondType(bond_orders[downgraded])
                    candidate.ReplaceBond(bond.GetIdx(), bond)
                    if Chem.SanitizeMol(candidate, catchErrors=True):
                        continue
                    candidates.append(candidate.GetMol())
                elif downgraded == 0:
                    candidate.RemoveBond(
                        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    )
                    if Chem.SanitizeMol(candidate, catchErrors=True):
                        continue
                    # Never score a pair of orphaned fragments as one molecule.
                    pieces = sorted(
                        Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=False),
                        key=lambda piece: piece.GetNumAtoms(),
                    )
                    if len(pieces) == 1 or pieces[0].GetNumAtoms() == 1:
                        candidates.append(pieces[-1])

    if cfg.allow_no_modification:
        candidates.append(Chem.Mol(mol))

    return _deduplicated(candidates)


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


def fragment_actions(library: tuple[fragments.Fragment, ...]) -> ActionSpace:
    """Substituent-level action space, bound to one library. See chem/fragments.py."""

    def actions(state: Chem.Mol | None, cfg: config.Config) -> tuple[Chem.Mol, ...]:
        if state is None:
            return valid_actions(None, cfg)
        return _deduplicated(fragments.substitutions(state, library))

    return actions


def reset(cfg: config.Config, action_space: ActionSpace = valid_actions) -> Episode:
    return Episode(
        state=cfg.init_mol,
        num_steps_taken=0,
        valid_actions=action_space(cfg.init_mol, cfg),
    )


def step(
    episode: Episode,
    action_index: int,
    reward_fn: Callable[[Chem.Mol], float],
    cfg: config.Config,
    action_space: ActionSpace = valid_actions,
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
    episode.valid_actions = action_space(episode.state, cfg)

    steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken
    # Only the terminal molecule counts; earlier ones are discounted by steps left.
    reward = reward_fn(episode.state) * cfg.discount_factor**steps_remaining
    return Result(
        state=episode.state, reward=reward, terminated=steps_remaining == 0
    )
