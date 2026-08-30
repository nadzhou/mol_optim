import numpy as np
import pytest

from mol_optim.report import plot_run


def slow(values, window):
    """The definition, written out: trailing mean over what exists so far."""
    return np.array(
        [np.mean(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]
    )


@pytest.mark.parametrize("window", [1, 3, 100])
def test_it_agrees_with_the_mean_computed_the_obvious_way(window):
    values = np.random.default_rng(0).normal(size=250)
    assert np.allclose(plot_run.rolling_mean(values, window), slow(values, window))


def test_a_window_longer_than_the_run_is_the_running_mean_throughout():
    values = np.array([0.0, 1.0, 2.0])
    assert np.allclose(plot_run.rolling_mean(values, 100), [0.0, 0.5, 1.0])
