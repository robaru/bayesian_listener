import numpy as np
import pyfar as pf
import pytest
from bayesian_listener.resample import resample_two_step, resample, find_max_order


def make_grid():
    """Small synthetic upper-hemisphere grid for fast tests."""
    az = np.deg2rad(np.arange(0, 360, 30))
    el = np.deg2rad(np.array([0, 30, 60]))
    az_grid, el_grid = np.meshgrid(az, el)
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    coords = pf.Coordinates(az_flat, el_flat, np.ones(len(az_flat)),
                            domain='sph', convention='top_elev')
    cues = np.random.default_rng(0).standard_normal((coords.csize, 10))
    return cues, coords


def test_resample_two_step_default_regularisation_coefficient():
    """resample_two_step runs with default regularisation_coefficient and returns correct shape."""
    cues, coords = make_grid()
    out, template = resample_two_step(cues, coords, None, 'sh')
    assert out.shape[1] == cues.shape[1]
    assert out.shape[0] == template.csize


def test_resample_two_step_custom_regularisation_coefficient():
    """Custom regularisation_coefficient produces a different result than the default."""
    cues, coords = make_grid()
    out_default, _ = resample_two_step(cues, coords, None, 'sh')
    out_custom, _ = resample_two_step(cues, coords, None, 'sh', regularisation_coefficient=1e-1)
    assert not np.allclose(out_default, out_custom)


def test_resample_two_step_custom_condition_threshold():
    """Custom condition_threshold changes the SH order selected by find_max_order."""
    cues, coords = make_grid()
    n_tight = find_max_order(coords, condition_threshold=5.0)
    n_loose = find_max_order(coords, condition_threshold=50.0)
    assert n_loose >= n_tight


def test_resample_kwargs_forwarded():
    """resample() forwards kwargs to resample_two_step."""
    cues, coords = make_grid()
    out_default, _ = resample(cues, coords, method='sh')
    out_custom, _ = resample(cues, coords, method='sh', regularisation_coefficient=1e-1)
    assert not np.allclose(out_default, out_custom)
