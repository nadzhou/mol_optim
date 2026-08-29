"""Splitting the EGFR dataset by scaffold.

BindingDB is full of series — one paper, forty analogs of one frame — so a random split
asks "have I seen this molecule's cousin" and answers beautifully. Whole scaffolds fall
on one side here, largest groups to training, which leaves the test set made of frames
with the least support. Seed scaffolds are held out of training on top of that.
"""

from collections import defaultdict
from typing import Sequence

from mol_optim.datasets import bindingdb


def by_scaffold(
    compounds: Sequence[bindingdb.Compound],
) -> dict[str, list[bindingdb.Compound]]:
    """Grouped by scaffold key, largest group first."""
    groups: dict[str, list[bindingdb.Compound]] = defaultdict(list)
    for compound in compounds:
        groups[compound.scaffold].append(compound)
    # Size then name: equal-sized groups must not swap places between runs.
    return dict(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    )


def scaffold_split(
    compounds: Sequence[bindingdb.Compound],
    test_fraction: float = 0.2,
    held_out_scaffolds: frozenset[str] = frozenset(),
) -> tuple[tuple[bindingdb.Compound, ...], tuple[bindingdb.Compound, ...]]:
    """(train, test), with no scaffold on both sides and no seed scaffold in train."""
    train: list[bindingdb.Compound] = []
    test: list[bindingdb.Compound] = []
    room_in_train = (1.0 - test_fraction) * len(compounds)

    for scaffold, group in by_scaffold(compounds).items():
        if scaffold in held_out_scaffolds or len(train) + len(group) > room_in_train:
            test.extend(group)
        else:
            train.extend(group)
    return tuple(train), tuple(test)
