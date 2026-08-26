"""The molecule MDP: atom-level graph edits, ported from MolDQN.

The state is an RDKit molecular graph and stays one for the whole episode. Nothing here
writes or parses a molecule as text — candidates are built by editing an RWMol, and
identity comes from graph_key.canonical_hash, taken off the graph itself. Molecules
become text only at the boundary, in report.py, where a person looks at them.

Restyled from the reference implementation: the reward arrives as a function rather
than a subclass override, and the three single-call action generators are inlined into
valid_actions.
"""

import itertools
from dataclasses import dataclass
from typing import Callable

from rdkit import Chem

from mol_optim import config, graph_key


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
    """Every graph reachable from `state` in one edit, deduplicated and in a fixed order.

    An action *is* the resulting molecule — the agent scores next states, it does not
    emit a fixed-size action distribution. Two edits landing on the same graph appear
    once, and the order is by canonical hash so a run is reproducible.
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
    # atoms_with_free_valence[order] = atoms that can accept a new bond of that order.
    atoms_with_free_valence = {
        order: [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= order
        ]
        for order in range(1, max(max_bonds.values()))
    }
    # Index i of this list is the bond of order i; index 0 (None) means "no bond", which
    # makes the bond order arithmetic below a plain list lookup.
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    candidates: list[Chem.Mol] = []

    # Atom addition: hang one new atom off every atom with room for that bond order.
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

    # Bond addition: a new bond between two existing atoms, or an upgrade of one that is
    # already there (single -> double -> triple). Aromatic bonds are never modified.
    for order, atom_indices in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atom_indices, 2):
            # Read the bond off a copy, so the SetBondType below cannot reach `mol`.
            bond = Chem.Mol(mol).GetBondBetweenAtoms(atom1, atom2)
            candidate = Chem.RWMol(mol)
            # Kekulize so sanitization does not trip over aromatic flags. Bonds that are
            # aromatic in `mol` are skipped outright, not rewritten.
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

    # Bond removal: downgrade a bond, or delete it outright.
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
                    # Deleting a bond can split the graph. Keep the action only if what
                    # is left is one fragment, or one fragment plus a lone atom; never
                    # score a pair of orphaned fragments as a molecule.
                    fragments = sorted(
                        Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=False),
                        key=lambda fragment: fragment.GetNumAtoms(),
                    )
                    if len(fragments) == 1 or fragments[0].GetNumAtoms() == 1:
                        candidates.append(fragments[-1])

    if cfg.allow_no_modification:
        candidates.append(Chem.Mol(mol))

    return _deduplicated(candidates)


def _deduplicated(candidates: list[Chem.Mol]) -> tuple[Chem.Mol, ...]:
    """One molecule per graph, ordered by canonical hash so the order is reproducible.

    Perception is normalized first (see graph_key.normalize): the stored molecule, not
    just its key, has to be the normalized one, or the same state reaches the network as
    two different fingerprints depending on which edit built it.
    """
    by_hash: dict[str, Chem.Mol] = {}
    for candidate in candidates:
        normalized = graph_key.normalize(candidate)
        by_hash.setdefault(graph_key.canonical_hash(normalized), normalized)
    return tuple(by_hash[key] for key in sorted(by_hash))


def reset(cfg: config.Config) -> Episode:
    return Episode(
        state=cfg.init_mol,
        num_steps_taken=0,
        valid_actions=valid_actions(cfg.init_mol, cfg),
    )


def step(
    episode: Episode,
    action_index: int,
    reward_fn: Callable[[Chem.Mol], float],
    cfg: config.Config,
) -> Result:
    """Moves to the candidate at `action_index` and scores the graph it lands on.

    The action is an index into episode.valid_actions rather than a molecule, so a step
    to a graph that is not a legal successor cannot be expressed.
    """
    if episode.num_steps_taken >= cfg.max_steps_per_episode:
        raise ValueError("This episode is terminated.")
    if not 0 <= action_index < len(episode.valid_actions):
        raise ValueError(
            f"No candidate {action_index}; there are {len(episode.valid_actions)}"
        )

    episode.state = episode.valid_actions[action_index]
    episode.num_steps_taken += 1
    episode.valid_actions = valid_actions(episode.state, cfg)

    steps_remaining = cfg.max_steps_per_episode - episode.num_steps_taken
    # Only the molecule at the end of an episode really counts, so a reward collected
    # with steps still to come is discounted by that many steps.
    reward = reward_fn(episode.state) * cfg.discount_factor**steps_remaining
    return Result(
        state=episode.state, reward=reward, terminated=steps_remaining == 0
    )
