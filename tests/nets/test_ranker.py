import numpy as np
import torch

from mol_optim import config
from mol_optim.nets import ranker
from tests.molecules import START_MOLECULES

CFG = config.Config()


def test_the_predicted_difference_is_exactly_antisymmetric():
    torch.manual_seed(0)
    model = ranker.Ranker(CFG)
    a, b = START_MOLECULES[0], START_MOLECULES[1]
    forward = ranker.score([model], [a, b], CFG)
    backward = ranker.score([model], [b, a], CFG)
    assert np.isclose(forward[0] - forward[1], -(backward[0] - backward[1]), atol=1e-6)


def test_an_added_constant_per_series_does_not_change_the_ranking():
    """Why the RL loop can read the score directly, with no reference molecule."""
    predicted = [np.array([1.0, 2.0, 3.0, 0.5, 4.0])]
    measured = [np.array([5.0, 6.0, 7.0, 4.5, 8.0])]
    shifted = [predicted[0] + 17.0]
    assert np.isclose(ranker.within_series_spearman(predicted, measured), 1.0)
    assert np.isclose(ranker.within_series_spearman(shifted, measured), 1.0)


def test_the_metric_is_a_median_over_series_not_one_pooled_correlation():
    # Two series, each ranked backwards inside itself, but the second sits entirely above
    # the first. Pooled they correlate positively; per series they are -1.
    predicted = [np.array([5.0, 4.0, 3.0, 2.0, 1.0]), np.array([50.0, 40.0, 30.0, 20.0, 10.0])]
    measured = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 20.0, 30.0, 40.0, 50.0])]
    assert np.isclose(ranker.within_series_spearman(predicted, measured), -1.0)


def test_series_with_no_spread_are_dropped():
    """A series whose compounds all measure the same has no ranking to get right."""
    flat = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    same = [np.array([9.0, 9.0, 9.0, 9.0, 9.0])]
    assert np.isnan(ranker.within_series_spearman(flat, same))
