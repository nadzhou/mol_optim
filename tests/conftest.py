"""Fixtures shared across test modules.

The EGFR dataset is 10,850 molecules read off disk with a scaffold key derived per
molecule — three seconds. Two test modules want it, so it is loaded once per session
rather than once per module.
"""

import pytest

from mol_optim import bindingdb


@pytest.fixture(scope="session")
def compounds() -> tuple[bindingdb.Compound, ...]:
    return bindingdb.load()
