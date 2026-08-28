"""Hyperparameters as frozen dataclasses, passed explicitly — never a global config bag."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Every knob for the atom-level MolDQN on QED, now with a GNN state encoder.

    Defaults follow Google's published configs/bootstrap_dqn.json, not
    MolDQN-pytorch/hyp.py. The PyTorch port quietly deviates from the published setup
    on five knobs — gamma, ring sizes, buffer size, update interval, gradient clipping
    — and reproduces a visibly worse QED curve because of it.
    """

    seed: int = 0

    # Environment / MDP
    init_mol: str | None = None
    atom_types: tuple[str, ...] = ("C", "O", "N")
    allow_removal: bool = True
    allow_no_modification: bool = True
    allow_bonds_between_rings: bool = False
    # 3- and 4-rings are strained and drag QED down; the published config leaves them out.
    allowed_ring_sizes: tuple[int, ...] = (5, 6)
    # Precedented decorations from vocabulary.load(), attachable at any free valence.
    # Empty means atom-level edits only, which is what every number in results/ was
    # measured on. Typed loosely to keep config.py free of package imports.
    fragments: tuple = ()
    # Reject candidates carrying a nitrogen-nitrogen bond outside a ring. Costs 1.2% of
    # measured EGFR actives and removes the polyazane chains the agent reaches for --
    # a constraint it cannot price, unlike a reward penalty.
    forbid_acyclic_nn: bool = False
    max_steps_per_episode: int = 40
    # Only the molecule at the end of an episode counts, so an intermediate reward is
    # discounted by discount_factor ** steps_remaining. This is the *environment's*
    # discount and is separate from gamma below.
    discount_factor: float = 0.9

    # State encoder: a GNN over the molecular graph. The published MolDQN
    # numbers below were tuned against a 2049 -> 1024 -> 512 -> 128 -> 32 -> 1 MLP over
    # a Morgan fingerprint, 2.7M parameters against this network's 56k.
    hidden_dim: int = 64
    num_message_passing_layers: int = 3

    # Agent. gamma is 1.0 because the environment already discounts by steps remaining;
    # a second discount here would charge the agent twice for taking its time.
    gamma: float = 1.0
    learning_rate: float = 1e-4
    grad_clip_norm: float = 10.0
    polyak: float = 0.995
    batch_size: int = 128
    replay_buffer_size: int = 5000
    update_interval: int = 4  # gradient steps per environment step, published: every 4
    updates_per_interval: int = 1

    # Exploration: piecewise linear, epsilon_start -> epsilon_mid at half the run ->
    # epsilon_end at the end.
    epsilon_start: float = 1.0
    epsilon_mid: float = 0.1
    epsilon_end: float = 0.01

    episodes: int = 5000


@dataclass(frozen=True)
class PretrainConfig:
    """Every knob for the ZINC AttrMask pretraining.

    Separate from Config, and passed alongside it: the encoder's shape stays in Config
    so the pretrained encoder and the RL encoder are built from one set of numbers.
    Two configs that each carried a hidden_dim would drift apart, and the checkpoint
    would load into a network of the wrong width.
    """

    seed: int = 0
    # ZINC 250k, minus the held-out tail. None means every molecule in the file.
    num_molecules: int | None = None
    num_holdout: int = 5000
    # Fraction of atoms in a batch whose feature row is zeroed. 0.15 is Hu et al. 2020.
    mask_fraction: float = 0.15
    epochs: int = 10
    batch_size: int = 128
    # 1e-3, not the DQN's 1e-4: this is plain supervised training with a fixed target,
    # not a moving Q target that a large step can push away from itself.
    learning_rate: float = 1e-3
    grad_clip_norm: float = 10.0


@dataclass(frozen=True)
class RegressorConfig:
    """Every knob for the BindingDB pIC50 regressor.

    Passed alongside Config for the same reason PretrainConfig is: the encoder's shape
    lives in one place, so the ZINC checkpoint, the RL encoder and this regressor are
    built from one set of numbers and the checkpoint loads into all three.
    """

    seed: int = 0
    test_fraction: float = 0.2
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip_norm: float = 10.0
    # Five networks, different seeds, same data. The mean is the prediction and the
    # spread is what the reward subtracts to stay pessimistic where the model guesses.
    ensemble_size: int = 5


@dataclass(frozen=True)
class PPOConfig:
    """Every knob for PPO on the molecule MDP.

    Alongside Config for the same reason the others are: the encoder's shape and the
    MDP's discounting stay in Config, so DQN and PPO run the same environment and load
    the same ZINC checkpoint. Only what is specific to the algorithm lives here.
    """

    seed: int = 0
    # Episodes per policy update. Short episodes (6 edits on pIC50) make a single
    # episode far too small a batch to estimate an advantage from.
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
