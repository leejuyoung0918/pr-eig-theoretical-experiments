import numpy as np

from figure2_experiments.exp_b_long.analyze import _slope_samples
from figure2_experiments.exp_b_long.run import REQUIRED_LONG_CHECKPOINTS, long_checkpoints
from figure2_experiments.plotting.plot_figure2 import _select_b_slopes


def test_long_checkpoints_include_pre_registered_values():
    checkpoints = set(long_checkpoints())
    assert set(REQUIRED_LONG_CHECKPOINTS).issubset(checkpoints)
    assert min(checkpoints) == 8192
    assert max(checkpoints) == 65536


def test_two_checkpoint_half_window_slope_is_defined():
    n = np.array([4096.0, 8192.0])
    epsilon = np.vstack([n ** -0.25, 2.0 * n ** -0.25])
    slopes = _slope_samples(n, epsilon, 4096, 8192)
    np.testing.assert_allclose(slopes, -0.25, atol=1e-14)


def test_publication_plot_selects_primary_or_terminal_half_window():
    import pandas as pd

    current = pd.DataFrame(
        {
            "method": ["PR-EIG", "Exact-RC", "Fisher-RC"],
            "fit_min_n": [512, 512, 512],
            "fit_max_n": [8192, 8192, 8192],
            "primary": [True, True, True],
        }
    )
    selected, title = _select_b_slopes(current, 8192)
    assert len(selected) == 3
    assert title == "Late-stage slopes"

    long = pd.DataFrame(
        {
            "method": ["PR-EIG", "Exact-RC", "Fisher-RC"],
            "analysis": ["half_window_local"] * 3,
            "fit_min_n": [32768] * 3,
            "fit_max_n": [65536] * 3,
        }
    )
    selected, title = _select_b_slopes(long, 65536)
    assert len(selected) == 3
    assert title == "Extended slopes"
