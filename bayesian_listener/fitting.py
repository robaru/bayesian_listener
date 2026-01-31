import numpy as np
import sys
from pathlib import Path
import time
from itertools import product
from pybads import BADS

from bayesian_listener import BayesianListener
from bayesian_listener.coordinates import Coordinates
from bayesian_listener.metrics import METRIC_FUNCTIONS

def allcomb(*arrays):
    """Cartesian product of input arrays (equivalent to MATLAB allcomb)."""
    return np.array(list(product(*arrays)))

def sigma2kappa(sigma):
    """Convert angular standard deviation (degrees) to von Mises-Fisher concentration."""
    return 1.0 / (2.0 * np.sin(np.deg2rad(sigma) / 2.0)**2)

def kappa2sigma(kappa):
    """Convert von Mises-Fisher concentration to angular standard deviation (degrees)."""
    return 2.0 * np.rad2deg(np.arcsin(np.sqrt(1.0 / (2.0 * kappa))))

def loglik(model, targets, responses_cart, resp_targets_idx, sigmas_log,
           num_repetitions=300):
    """
    Negative log-likelihood function for parameter fitting.

    Parameters
    ----------
    model : BayesianListener
        Model instance (for running inference)
    targets : ndarray
        Target feature vectors (n_targets x n_features)
    responses_cart : ndarray
        Observed responses in Cartesian coordinates (n_obs x 3)
    resp_targets_idx : ndarray
        Index mapping each response to a target direction (n_obs,)
    sigmas_log : ndarray
        Log-transformed parameters [log(sigma_ild), log(sigma_spectral),
                                     log(kappa_motor), log(sigma_prior)]
    num_repetitions : int
        Number of Monte Carlo repetitions for likelihood evaluation.

    Returns
    -------
    neglik : float
        Negative log-likelihood
    """
    # Parameters
    num_exp = num_repetitions
    sigma_itd = 0.569  # Fixed ITD noise

    # Convert from log space
    sigmas = np.exp(sigmas_log)
    sigma_ild = sigmas[0]
    sigma_spectral = sigmas[1]
    kappa = sigmas[2]  # Motor noise as concentration parameter
    sigma_prior = sigmas[3]

    # Set model parameters (without motor noise for now)
    model.parameters = {
        'sigma_itd': sigma_itd,
        'sigma_ild': sigma_ild,
        'sigma_spectral': sigma_spectral,
        'sigma_prior': sigma_prior,
        'sigma_motor': 0
    }

    # Run model WITHOUT motor noise to get MAP predictions
    posterior = model.infer(targets, repetitions=num_exp)

    # Get MAP estimates without motor noise
    doa_estimations = model.estimate(posterior, sigma_motor=False)  # (n_targets x num_exp x 3)

    # von Mises-Fisher log-normalization constant (avoids sinh overflow)
    # log C = log(kappa / (4π sinh(kappa))) = log(kappa) - log(4π) - log(sinh(kappa))
    # For large kappa: log(sinh(kappa)) ≈ kappa - log(2)
    if kappa < 1e-6:
        log_C = -np.log(4.0 * np.pi)  # Uniform distribution
    elif kappa > 500:
        log_C = np.log(kappa) - np.log(4.0 * np.pi) - kappa + np.log(2.0)
    else:
        log_C = np.log(kappa) - np.log(4.0 * np.pi) - np.log(np.sinh(kappa))

    # Compute log-likelihood for each observation
    loglik_total = 0.0
    targets_idx = np.unique(resp_targets_idx)

    for i in range(len(targets_idx)):
        target_idx = targets_idx[i]

        # Get all observations for this target
        obs_mask = (resp_targets_idx == target_idx)
        obs = responses_cart[obs_mask, :]  # (num_obs_for_target x 3)

        # Get model predictions for this target
        est = doa_estimations[target_idx, :, :]  # (num_exp x 3)

        # Compute vMF PDF for each observation × each MC sample
        cos_angles = obs @ est.T  # (num_obs x num_exp)

        # Compute log of vMF PDF for each (observation, MC sample) pair
        log_pdfs = log_C + kappa * cos_angles  # (num_obs x num_exp)

        # Average over MC samples using log-sum-exp trick for numerical stability
        max_log_pdfs = np.max(log_pdfs, axis=1, keepdims=True)  # (num_obs x 1)
        log_mean_probs = max_log_pdfs.squeeze() + np.log(np.mean(np.exp(log_pdfs - max_log_pdfs), axis=1))

        # Accumulate log-likelihood across all observations
        loglik_total += np.sum(log_mean_probs)

    return -loglik_total


def fit_listener(sofa_path, obs_tbl, targets_coords,
                 interpolation_method, subject_id=None,
                 num_repetitions=300, num_grid_points=1,
                 verbose=True):
    """
    Fit the Bayesian listener model for a single participant.

    Parameters
    ----------
    sofa_path : str
        Path to the participant's SOFA file.
    obs_tbl : DataFrame
        Behavioral observations. Must contain columns:
        'azi_response', 'ele_response', 'azi_target', 'ele_target'.
        If subject_id is provided, must also contain 'subject'.
    targets_coords : Coordinates
        Target direction coordinates.
    interpolation_method : str
        HRTF interpolation method (e.g. 'SH', 'SHMAX', 'barycentric',
        'barumerli2023').
    subject_id : str, optional
        Subject identifier. If provided, obs_tbl is filtered to this
        subject. If None, all rows in obs_tbl are used.
    num_repetitions : int
        Number of Monte Carlo repetitions for likelihood evaluation.
    num_grid_points : int
        Number of grid points per parameter dimension for initialization.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Fitting results containing fitted parameters, NLL, and timing.
    """
    label = subject_id or Path(sofa_path).stem
    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject: {label} | Method: {interpolation_method}")
        print(f"{'='*60}")

    t_start_total = time.time()

    try:
        # Get subject data
        if subject_id is not None:
            subj_data = obs_tbl[obs_tbl['participant'] == subject_id]
        else:
            subj_data = obs_tbl
        if verbose:
            print(f"  Trials: {len(subj_data)}")

        # Load HRTF and create model
        model = BayesianListener(sofa_path)
        model.prepare_features(interpolation=interpolation_method)

        # Extract features
        target_indices = model.coords.find(targets_coords)[1]
        targets = model.represent()[target_indices, :]

        # Convert responses to Cartesian
        resp_coords = Coordinates(
            positions=np.column_stack([
                np.deg2rad(subj_data['azi_response'].values),
                np.deg2rad(subj_data['ele_response'].values),
                np.ones(len(subj_data))
            ]),
            convention='spherical'
        )
        resp_cart = resp_coords.convert('cartesian')

        resp_targets = Coordinates(
            positions=np.column_stack([
                np.deg2rad(subj_data['azi_target'].values),
                np.deg2rad(subj_data['ele_target'].values),
                np.ones(len(subj_data))
            ]),
            convention='spherical'
        )
        resp_targets_idx = targets_coords.find(resp_targets)[1]

        # Compute sdL for motor noise upper bound
        resp_hp = resp_coords.convert('horizontal-polar')
        resp_targets_hp = Coordinates(
            positions=targets_coords.convert('cartesian')[resp_targets_idx, :],
            convention='cartesian'
        ).convert('horizontal-polar')
        sdL_rad, _ = METRIC_FUNCTIONS['sdL'](resp_targets_hp, resp_hp)
        sdL_deg = np.rad2deg(sdL_rad)
        if verbose:
            print(f"  sdL: {sdL_deg:.2f}°")

        # Define likelihood function
        def fll(sigmas_log):
            return loglik(model, targets, resp_cart, resp_targets_idx, sigmas_log,
                          num_repetitions=num_repetitions)

        # [log(sigma_ild), log(sigma_spectral), log(kappa_motor), log(sigma_prior)]
        # Parameter bounds (in log space)
        # Motor noise bounds defined in sigma (degrees), then converted to kappa.
        # sigma2kappa is monotonically decreasing, so bounds flip:
        #   sigma:  lb < plb < pub < ub
        #   kappa:  ub > pub > plb > lb   (reversed)
        sigma_motor_lb = 1.0        # hard lower bound (deg)
        sigma_motor_plb = 2.0       # plausible lower bound (deg)
        sigma_motor_pub = 0.9 * sdL_deg  # plausible upper bound (deg)
        sigma_motor_ub = sdL_deg         # hard upper bound (deg)

        lb = np.array([np.log(0.1), np.log(0.1), np.log(sigma2kappa(sigma_motor_ub)), np.log(1)])
        plb = np.array([np.log(.5), np.log(1), np.log(sigma2kappa(sigma_motor_pub)), np.log(5)])
        pub = np.array([np.log(3), np.log(10), np.log(sigma2kappa(sigma_motor_plb)), np.log(50)])
        ub = np.array([np.log(50), np.log(100), np.log(sigma2kappa(sigma_motor_lb)), np.log(179.9)])

        # Grid search for initialization
        if verbose:
            print("  Grid search...")
        n = num_grid_points
        kappa_grid_lo = sigma2kappa(sigma_motor_pub)
        kappa_grid_hi = sigma2kappa(sigma_motor_plb)
        grid_points = allcomb(
            np.log(np.linspace(1, 3, n)),
            np.log(np.linspace(10, 15, n)),
            np.log(np.linspace(kappa_grid_lo, kappa_grid_hi, n)),
            np.log(np.linspace(50, 40, n))
        )

        t_grid_start = time.time()
        nll = np.array([fll(grid_points[i]) for i in range(len(grid_points))])
        t_grid = time.time() - t_grid_start

        sigmas_0 = grid_points[np.argmin(nll), :]
        if verbose:
            print(f"  Best grid NLL: {nll.min():.2f} ({t_grid:.1f}s)")

        # BADS optimization
        if verbose:
            print("  BADS optimization...")
        options = {"tol_mesh": 1e-2}

        t_bads_start = time.time()
        bads = BADS(fll, sigmas_0, lb, ub, plb, pub, options=options)
        result = bads.optimize()
        t_bads = time.time() - t_bads_start

        # Extract fitted parameters
        fitted_params = np.exp(result['x'])
        fitted_params[2] = kappa2sigma(fitted_params[2])  # Convert kappa to sigma

        t_total = time.time() - t_start_total

        if verbose:
            print(f"  Final NLL: {result['fval']:.2f} ({t_bads:.1f}s)")
            print(f"  Total time: {t_total:.1f}s")
            print(f"  sigma_ILD: {fitted_params[0]:.2f}, sigma_spectral: {fitted_params[1]:.2f}, "
                  f"sigma_motor: {fitted_params[2]:.2f}, sigma_prior: {fitted_params[3]:.2f}")

        return {
            'subject': label,
            'method': interpolation_method,
            'sigma_ild': fitted_params[0],
            'sigma_spectral': fitted_params[1],
            'sigma_motor': fitted_params[2],
            'sigma_prior': fitted_params[3],
            'sdL': sdL_deg,
            'nll': result['fval'],
            'nll_initial': nll.min(),
            'time_grid': t_grid,
            'time_bads': t_bads,
            'time_total': t_total,
            'n_trials': len(subj_data),
            'success': True
        }

    except Exception as e:
        print(f"ERROR fitting {label} with {interpolation_method}: {e}")
        return {
            'subject': label,
            'method': interpolation_method,
            'success': False,
            'error': str(e)
        }