"""Fixtures shared across test modules.

The EGFR dataset is 10,850 molecules read off disk with a scaffold key derived per
molecule — three seconds. Two test modules want it, so it is loaded once per session
rather than once per module.

Neither downloaded dataset is in version control, so a fresh checkout has neither. A
test that needs one skips with the command that builds it rather than erroring, which is
what lets CI run the rest of the suite on a clean clone.

The paths come from the config dataclasses' own defaults, so the tests look where the
default config file writes.
"""

from pathlib import Path

import pytest

from mol_optim import bindingdb, config

BINDINGDB_PATH = config.BindingDBSpec().path
ZINC_PATH = config.ZincSpec().path
BUILD_IT = "mol-optim configs/config.toml"


def require(path: Path, command: str) -> None:
    """Skips the calling test if `path` has not been built yet."""
    if not path.exists():
        pytest.skip(f"{path} is missing; build it with: {command}")


@pytest.fixture(scope="session")
def compounds() -> tuple[bindingdb.Compound, ...]:
    require(BINDINGDB_PATH, BUILD_IT)
    return bindingdb.load(BINDINGDB_PATH)
