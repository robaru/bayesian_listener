import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import pyfar as pf
from bayesian_listener.fitting import (
    fit_listener, negloglik, sigma_to_kappa, kappa_to_sigma, allcomb,
)

from tests.test_bayesian_listener import get_sofa_file


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_sigma_to_kappa_roundtrip():
    """sigma_to_kappa and kappa_to_sigma should be approximate inverses."""
    for sigma in [5.0, 10.0, 20.0, 45.0]:
        kappa = sigma_to_kappa(sigma)
        sigma_back = kappa_to_sigma(kappa)
        np.testing.assert_allclose(sigma_back, sigma, rtol=0.1)


def test_sigma_to_kappa_monotonic():
    """Smaller sigma should give larger kappa."""
    assert sigma_to_kappa(5) > sigma_to_kappa(10) > sigma_to_kappa(20)


def test_allcomb():
    """allcomb should return the Cartesian product."""
    result = allcomb([1, 2], [3, 4])
    expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Fixture: model + synthetic data for negloglik / fit_listener tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fitting_data():
    """Create minimal synthetic data for fit_listener."""
    sofa_path = get_sofa_file()

    # A few target directions (spherical: azi, ele, r=1) in degrees
    target_dirs = np.array([
        [0, 0, 1],
        [45, 0, 1],
        [0, 45, 1],
    ])
    targets_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(target_dirs[:, 0]),
        np.deg2rad(target_dirs[:, 1]),
        target_dirs[:, 2],
    )

    # Synthetic responses — small noise around the targets
    rng = np.random.default_rng(42)
    rows = []
    for azi, ele, _ in target_dirs:
        for _ in range(10):
            rows.append({
                'participant': 'test_subj',
                'azi_target': azi,
                'ele_target': ele,
                'azi_response': azi + rng.normal(0, 3),
                'ele_response': ele + rng.normal(0, 5),
            })
    obs_tbl = pd.DataFrame(rows)

    return sofa_path, obs_tbl, targets_coords


@pytest.fixture
def model_and_arrays(fitting_data):
    """Prepare BayesianListener model and arrays needed by negloglik."""
    from bayesian_listener import BayesianListener

    sofa_path, obs_tbl, targets_coords = fitting_data

    model = BayesianListener(sofa_path)
    model.compute_template(interpolation='SH')

    target_indices = model.coords.find_nearest(targets_coords)[0][0]
    targets = model.target[target_indices]

    subj_data = obs_tbl[obs_tbl['participant'] == 'test_subj']
    resp_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(subj_data['azi_response'].values),
        np.deg2rad(subj_data['ele_response'].values),
        np.ones(len(subj_data)),
    )

    resp_targets = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(subj_data['azi_target'].values),
        np.deg2rad(subj_data['ele_target'].values),
        np.ones(len(subj_data)),
    )
    resp_targets_idx = targets_coords.find_nearest(resp_targets)[0][0]

    return model, targets, resp_coords, resp_targets_idx


# ---------------------------------------------------------------------------
# negloglik tests
# ---------------------------------------------------------------------------

def test_negloglik_returns_finite_scalar(model_and_arrays):
    """negloglik should return a finite positive scalar for moderate kappa."""
    model, targets, resp_coords, resp_targets_idx = model_and_arrays

    sigmas_log = np.log([2.0, 8.0, 15.0, 30.0])
    nll = negloglik(model, targets, resp_coords, resp_targets_idx, sigmas_log,
                    num_repetitions=5)

    assert np.isscalar(nll)
    assert np.isfinite(nll)


@pytest.mark.parametrize("kappa", [
    1.0,            # low concentration (broad motor noise)
    50.0,           # moderate
    500.0,          # boundary of sinh overflow
    1000.0,         # well into large-kappa regime
    sigma_to_kappa(1), # extreme precision
])
def test_negloglik_finite_across_kappa_range(model_and_arrays, kappa):
    """negloglik must stay finite for kappa values spanning the full bound range."""
    model, targets, resp_coords, resp_targets_idx = model_and_arrays

    sigmas_log = np.log([2.0, 8.0, kappa, 30.0])
    nll = negloglik(model, targets, resp_coords, resp_targets_idx, sigmas_log,
                    num_repetitions=5)

    assert np.isfinite(nll), f"NaN/Inf for kappa={kappa}"


# ---------------------------------------------------------------------------
# fit_listener tests (mocked BADS)
# ---------------------------------------------------------------------------

def test_fit_listener_smoke(fitting_data):
    """Run fit_listener end-to-end with mocked BADS (no real optimisation)."""
    sofa_path, obs_tbl, targets_coords = fitting_data

    mock_bads_instance = MagicMock()
    mock_bads_instance.optimize.return_value = {
        'x': np.log([2.0, 8.0, 15.0, 30.0]),
        'fval': 1234.5,
    }

    with patch('bayesian_listener.fitting.BADS', return_value=mock_bads_instance):
        result = fit_listener(
            sofa_path=sofa_path,
            obs_tbl=obs_tbl,
            targets_coords=targets_coords,
            interpolation_method='SH',
            subject_id='test_subj',
            num_repetitions=5,
            num_grid_points=1,
            verbose=True,
        )

    assert result['success'] is True
    assert result['subject'] == 'test_subj'
    assert result['method'] == 'SH'
    for key in ('sigma_ild', 'sigma_spectral', 'kappa_motor', 'sigma_motor', 'sigma_prior'):
        assert key in result
    assert result['nll'] == 1234.5
    mock_bads_instance.optimize.assert_called_once()


def test_fit_listener_bounds_ordering(fitting_data):
    """Verify BADS is called with strictly ordered bounds (lb < plb < pub < ub)."""
    sofa_path, obs_tbl, targets_coords = fitting_data

    captured_args = {}

    def capture_bads(fun, x0, lb, ub, plb, pub, **kwargs):
        captured_args['lb'] = lb
        captured_args['plb'] = plb
        captured_args['pub'] = pub
        captured_args['ub'] = ub
        mock = MagicMock()
        mock.optimize.return_value = {
            'x': np.log([2.0, 8.0, 15.0, 30.0]),
            'fval': 1000.0,
        }
        return mock

    with patch('bayesian_listener.fitting.BADS', side_effect=capture_bads):
        result = fit_listener(
            sofa_path=sofa_path,
            obs_tbl=obs_tbl,
            targets_coords=targets_coords,
            interpolation_method='SH',
            subject_id='test_subj',
            num_repetitions=5,
            num_grid_points=1,
            verbose=False,
        )

    assert result['success'] is True

    lb = captured_args['lb']
    plb = captured_args['plb']
    pub = captured_args['pub']
    ub = captured_args['ub']

    for i in range(len(lb)):
        assert lb[i] < plb[i], f"dim {i}: lb ({lb[i]}) >= plb ({plb[i]})"
        assert plb[i] < pub[i], f"dim {i}: plb ({plb[i]}) >= pub ({pub[i]})"
        assert pub[i] < ub[i], f"dim {i}: pub ({pub[i]}) >= ub ({ub[i]})"
