from pathlib import Path

import pytest

from mol_optim import config
from mol_optim.datasets import bindingdb

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

