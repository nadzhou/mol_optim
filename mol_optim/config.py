"""Every knob, as frozen dataclasses built from a TOML file. Never a global config bag.

`load` is the only thing that reads the file; everything downstream is passed a
dataclass explicitly. A value used in exactly one place is a literal at that place.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin


@dataclass(frozen=True)
class Config:
    # Defaults follow Google's configs/bootstrap_dqn.json, not MolDQN-pytorch/hyp.py.
    # The PyTorch port deviates on gamma, ring sizes, buffer size, update interval and
    # gradient clipping, and reproduces a visibly worse reward curve because of it.
    seed: int = 0

    init_mol: str | None = None
    atom_types: tuple[str, ...] = ("C", "O", "N")
    allow_removal: bool = True
    allow_no_modification: bool = True
    allow_bonds_between_rings: bool = False
    # 3- and 4-rings are strained and rarely make it into a real compound; the published
    # config leaves them out.
    allowed_ring_sizes: tuple[int, ...] = (5, 6)
    max_steps_per_episode: int = 40
    # Only the terminal molecule counts, so an intermediate reward is discounted by
    # discount_factor ** steps_remaining. Separate from gamma below.
    discount_factor: float = 0.9

    # The published MolDQN numbers were tuned against a 2049 -> 1024 -> 512 -> 128 -> 32
    # -> 1 MLP over a Morgan fingerprint: 2.7M parameters against this network's 56k.
    hidden_dim: int = 64
    num_message_passing_layers: int = 3

    # gamma is 1.0 because the environment already discounts by steps remaining; a second
    # discount here would charge the agent twice for taking its time.
    gamma: float = 1.0
    learning_rate: float = 1e-4
    grad_clip_norm: float = 10.0
    polyak: float = 0.995
    batch_size: int = 128
    replay_buffer_size: int = 5000
    update_interval: int = 4  # gradient steps per environment step, published: every 4
    updates_per_interval: int = 1

    # Piecewise linear: start -> mid at half the run -> end at the end.
    epsilon_start: float = 1.0
    epsilon_mid: float = 0.1
    epsilon_end: float = 0.01

    episodes: int = 5000


@dataclass(frozen=True)
class PretrainConfig:
    # The encoder's shape stays in Config, so the pretrained encoder, the RL encoder and
    # the regressor are built from one set of numbers and one checkpoint loads into all
    # three. A hidden_dim here too would drift.
    seed: int = 0
    num_molecules: int | None = None  # None means every molecule in the file
    num_holdout: int = 5000
    mask_fraction: float = 0.15  # Hu et al. 2020
    epochs: int = 10
    batch_size: int = 128
    # 1e-3, not the DQN's 1e-4: supervised training with a fixed target, not a moving Q
    # target that a large step can push away from itself.
    learning_rate: float = 1e-3
    grad_clip_norm: float = 10.0


@dataclass(frozen=True)
class RegressorConfig:
    seed: int = 0
    test_fraction: float = 0.2
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip_norm: float = 10.0
    # Five networks, different seeds, same data. The mean is the prediction; the spread is
    # what the reward subtracts to stay pessimistic where the model is guessing.
    ensemble_size: int = 5


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    # Short episodes (6 edits on pIC50) make a single episode far too small a batch to
    # estimate an advantage from.
    rollout_episodes: int = 16
    update_epochs: int = 4
    minibatch_steps: int = 32
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    # The candidate set is large and the policy is a softmax over it, so collapse is the
    # failure mode. Small, but not zero.
    entropy_coefficient: float = 0.01
    # 3e-4, not the DQN's 1e-4: PPO's clipped objective bounds its own step size.
    learning_rate: float = 3e-4
    grad_clip_norm: float = 10.0
    num_updates: int = 60


@dataclass(frozen=True)
class ZincSpec:
    url: str = "https://dataverse.harvard.edu/api/access/datafile/4170963"
    sha256: str = "b65ee88f1838586571fc41200ee60fb7b97da55da72823bed72dc315af2fb48b"
    path: Path = Path("data/zinc.tab")


@dataclass(frozen=True)
class BindingDBSpec:
    url: str = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
    # MD5 because that is the digest BindingDB publishes beside the file.
    md5: str = "dac667f2d194ae6744104a7f87549db5"
    archive: Path = Path("data/BindingDB_All_202608_tsv.zip")
    table: str = "BindingDB_All.tsv"
    path: Path = Path("data/egfr_ic50.sdf")
    uniprot: str = "P00533"
    # One construct, not one UniProt id: P00533 covers 51 EGFR constructs, and pooling
    # wild type with T790M puts one compound's two very different numbers under one label.
    construct: str = "Epidermal growth factor receptor"


@dataclass(frozen=True)
class PretrainSpec:
    cfg: PretrainConfig = PretrainConfig()
    checkpoint: Path | None = None
    log: Path | None = None
    report_every: int = 1


@dataclass(frozen=True)
class RegressorSpec:
    cfg: RegressorConfig = RegressorConfig()
    checkpoint: Path | None = None
    pretrained_encoder: Path | None = None  # omitted means random init, the null
    report_every: int = 0


@dataclass(frozen=True)
class AgentSpec:
    kind: str = "dqn"  # a key of cli.AGENTS
    name: str = "run"  # names the CSV, the checkpoint and the top-k under Settings.runs
    seed_molecule: int | None = None  # index into seeds.choose()
    regressor: Path = Path("models/egfr_regressor.pt")
    pretrained_encoder: Path | None = None
    report_every: int = 25
    top_k: int = 12
    cfg: Config = Config()
    ppo: PPOConfig = PPOConfig()


@dataclass(frozen=True)
class AuditSpec:
    sdf: tuple[Path, ...] = ()
    seed_molecule: int | None = None


@dataclass(frozen=True)
class RecoverySpec:
    logs: tuple[Path, ...] = ()
    seed_molecule: int | None = None


@dataclass(frozen=True)
class PlotSpec:
    kind: str = "run"  # a key of cli.PLOTS
    out: Path = Path("results/plot.png")
    inputs: tuple[Path, ...] = ()
    window: int = 100
    ylabel: str = "terminal reward"
    random_baseline: float | None = None
    seed_reward: float | None = None


@dataclass(frozen=True)
class Settings:
    steps: tuple[str, ...]
    runs: Path
    zinc: ZincSpec
    bindingdb: BindingDBSpec
    pretrain: PretrainSpec
    regressor: RegressorSpec
    agents: tuple[AgentSpec, ...]
    audit: AuditSpec
    recovery: RecoverySpec
    plots: tuple[PlotSpec, ...]


def build(cls, table: dict[str, Any], **nested):
    """A frozen dataclass from a TOML table, falling back to the dataclass's defaults.

    Coerces by annotation — str to Path, list to tuple — and refuses a key the dataclass
    does not have, so a typo in the config file is an error rather than a setting that
    silently does nothing. `nested` supplies fields that are themselves dataclasses.
    """
    known = {field.name: field for field in fields(cls)}
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no {', '.join(unknown)}. "
            f"It takes: {', '.join(sorted(set(known) - set(nested)))}"
        )
    values = dict(nested)
    for name, value in table.items():
        if name in nested:
            continue
        values[name] = _coerce(value, known[name].type)
    return cls(**values)


def _coerce(value: Any, annotation: Any) -> Any:
    # `X | None` arrives as a UnionType; the interesting half is the arm that is not None.
    if isinstance(annotation, UnionType):
        annotation = next(arm for arm in get_args(annotation) if arm is not type(None))
    if annotation is Path:
        return Path(value)
    if get_origin(annotation) is tuple:
        element = get_args(annotation)[0]
        return tuple(_coerce(item, element) for item in value)
    return value


def _split(table: dict[str, Any], spec_class) -> tuple[dict, dict]:
    """A table into (the spec's own keys, everything else), so one flat TOML section can
    fill a spec and the algorithm config nested inside it."""
    names = {field.name for field in fields(spec_class)}
    return (
        {k: v for k, v in table.items() if k in names},
        {k: v for k, v in table.items() if k not in names},
    )


def load(path: Path) -> Settings:
    """The TOML file as one Settings. Paths in it are relative to the working directory."""
    with open(path, "rb") as config_file:
        table = tomllib.load(config_file)

    agents = []
    ppo_names = {field.name for field in fields(PPOConfig)}
    for agent_table in table.pop("agents", []):
        mine, rest = _split(agent_table, AgentSpec)
        agents.append(
            build(
                AgentSpec,
                mine,
                cfg=build(Config, {k: v for k, v in rest.items() if k not in ppo_names}),
                ppo=build(PPOConfig, {k: v for k, v in rest.items() if k in ppo_names}),
            )
        )

    pretrain_spec, pretrain_cfg = _split(table.pop("pretrain", {}), PretrainSpec)
    regressor_spec, regressor_cfg = _split(table.pop("regressor", {}), RegressorSpec)

    steps = tuple(table.pop("steps", ()))
    runs = Path(table.pop("runs", "runs"))
    settings = Settings(
        steps=steps,
        runs=runs,
        zinc=build(ZincSpec, table.pop("zinc", {})),
        bindingdb=build(BindingDBSpec, table.pop("bindingdb", {})),
        pretrain=build(
            PretrainSpec, pretrain_spec, cfg=build(PretrainConfig, pretrain_cfg)
        ),
        regressor=build(
            RegressorSpec, regressor_spec, cfg=build(RegressorConfig, regressor_cfg)
        ),
        agents=tuple(agents),
        audit=build(AuditSpec, table.pop("audit", {})),
        recovery=build(RecoverySpec, table.pop("recovery", {})),
        plots=tuple(build(PlotSpec, plot) for plot in table.pop("plots", [])),
    )
    if table:
        raise ValueError(f"{path} has sections nothing reads: {', '.join(sorted(table))}")
    return settings
