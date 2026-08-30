import pytest

from mol_optim import config


def _write(tmp_path, body: str):
    path = tmp_path / "experiment.toml"
    path.write_text(body)
    return config.load(path)


AGENT = '''
steps = ["agents"]

[[agents]]
kind = "dqn"
name = "run"
seed = 7
learning_rate = 0.5
grad_clip_norm = 3.0
episodes = 11
'''


def test_a_key_both_configs_have_reaches_both(tmp_path):
    spec = _write(tmp_path, AGENT).agents[0]
    assert (spec.cfg.seed, spec.ppo.seed) == (7, 7)
    assert (spec.cfg.learning_rate, spec.ppo.learning_rate) == (0.5, 0.5)
    assert (spec.cfg.grad_clip_norm, spec.ppo.grad_clip_norm) == (3.0, 3.0)


def test_a_key_only_one_config_has_reaches_that_one(tmp_path):
    spec = _write(tmp_path, AGENT).agents[0]
    assert spec.cfg.episodes == 11  # Config only
    assert spec.ppo.num_updates == config.PPOConfig().num_updates


def test_a_key_no_config_has_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="learnign_rate"):
        _write(tmp_path, AGENT.replace("learning_rate", "learnign_rate"))
