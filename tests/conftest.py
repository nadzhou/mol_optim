"""Fixtures shared across test modules.

The EGFR dataset is 10,850 molecules read off disk with a scaffold key derived per
molecule — three seconds. Two test modules want it, so it is loaded once per session
rather than once per module.
"""

from pathlib import Path

import pytest

from mol_optim import bindingdb, vocabulary


@pytest.fixture(scope="session")
def compounds() -> tuple[bindingdb.Compound, ...]:
    return bindingdb.load()


@pytest.fixture(scope="session")
def fragments() -> tuple[vocabulary.Fragment, ...]:
    """The committed vocabulary, read off disk in a millisecond. Building it is 15 s."""
    return vocabulary.load(Path("data/egfr_fragments.sdf"))
