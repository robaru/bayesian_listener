import numpy as np
import pyfar as pf
from bayesian_listener.resample import resample_two_step, resample, find_max_order


def make_grid():
    """Small synthetic upper-hemisphere grid for fast tests."""
    az = np.deg2rad(np.arange(0, 360, 30))
    el = np.deg2rad(np.array([0, 30, 60]))
    az_grid, el_grid = np.meshgrid(az, el)
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    coords = pf.Coordinates.from_spherical_elevation(
        az_flat, el_flat, np.ones(len(az_flat)))
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


def test_resample_barycentric():
    """Barycentric path runs and weights sum to 1 for each target."""
    cues, coords = make_grid()
    out, template = resample_two_step(cues, coords, None, 'barycentric')
    assert out.shape[1] == cues.shape[1]
    assert out.shape[0] == template.csize


def test_vbap_interpolate_weights():
    """vbap_interpolate returns rows summing to 1 with at most 3 non-zeros."""
    from bayesian_listener.utils import vbap_interpolate
    _, coords = make_grid()
    grid = coords.cartesian
    src = grid[:5]
    weights = vbap_interpolate(src, grid)
    assert weights.shape == (5, grid.shape[0])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    for row in weights:
        assert np.sum(row > 0) <= 3


def test_vbap_interpolate_norm2():
    """norm=2 rows have unit L2 norm and differ from norm=1 for off-grid src."""
    from bayesian_listener.utils import vbap_interpolate
    # full-sphere grid so all targets are enclosed
    az = np.deg2rad(np.arange(0, 360, 30))
    el = np.deg2rad(np.array([-60, -30, 0, 30, 60]))
    az_grid, el_grid = np.meshgrid(az, el)
    coords = pf.Coordinates.from_spherical_elevation(
        az_grid.ravel(), el_grid.ravel(), np.ones(az_grid.size))
    grid = coords.cartesian
    rng = np.random.default_rng(7)
    src = rng.standard_normal((5, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    w1 = vbap_interpolate(src, grid, norm=1)
    w2 = vbap_interpolate(src, grid, norm=2)
    l2_norms = np.sqrt(np.sum(w2 ** 2, axis=1))
    np.testing.assert_allclose(l2_norms, 1.0, atol=1e-10)
    assert not np.allclose(w1, w2)


def test_vbap_interpolate_matches_spaudiopy():
    """vbap_interpolate matches spaudiopy.decoder.vbap (norm=1) to 1e-6.

    Uses a full-sphere grid (both hemispheres) so all target directions are
    enclosed by the convex hull, matching the actual use case in
    resample_two_step where complement_sampling fills the bottom first.
    """
    import spaudiopy
    from bayesian_listener.utils import vbap_interpolate

    # full sphere grid
    az = np.deg2rad(np.arange(0, 360, 30))
    el = np.deg2rad(np.array([-60, -30, 0, 30, 60]))
    az_grid, el_grid = np.meshgrid(az, el)
    coords = pf.Coordinates.from_spherical_elevation(
        az_grid.ravel(), el_grid.ravel(), np.ones(az_grid.size))
    grid = coords.cartesian

    rng = np.random.default_rng(42)
    src = rng.standard_normal((10, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)

    hull = spaudiopy.decoder.LoudspeakerSetup(grid[:, 0], grid[:, 1], grid[:, 2])
    w_spa = spaudiopy.decoder.vbap(src, hull, norm=1)
    w_ours = vbap_interpolate(src, grid)

    np.testing.assert_allclose(w_ours, w_spa, atol=1e-6)


def test_resample_kwargs_forwarded():
    """resample() forwards kwargs to resample_two_step."""
    cues, coords = make_grid()
    out_default, _ = resample(cues, coords, method='sh')
    out_custom, _ = resample(cues, coords, method='sh', regularisation_coefficient=1e-1)
    assert not np.allclose(out_default, out_custom)
