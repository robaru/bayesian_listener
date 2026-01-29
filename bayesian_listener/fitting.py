import numpy as np
import sys
from pathlib import Path
import time
from itertools import product
from pybads import BADS

from bayesian_listener import BayesianListener
from bayesian_listener import Coordinates

def allcomb(*arrays):
    """Cartesian product of input arrays (equivalent to MATLAB allcomb)."""
    return np.array(list(product(*arrays)))

def sigma2kappa(sigma):
    """Convert angular standard deviation (degrees) to von Mises-Fisher concentration."""
    return 1.0 / (2.0 * np.sin(np.deg2rad(sigma) / 2.0)**2)

def kappa2sigma(kappa):
    """Convert von Mises-Fisher concentration to angular standard deviation (degrees)."""
    return 2.0 * np.rad2deg(np.arcsin(np.sqrt(1.0 / (2.0 * kappa))))

def loglik(model, targets, responses_cart, resp_targets_idx, sigmas_log):
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

    Returns
    -------
    neglik : float
        Negative log-likelihood
    """
    # Parameters
    num_exp = 300  # Monte Carlo samples
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

    # von Mises-Fisher normalization constant
    if kappa < 1e-6:
        C = 1.0 / (4.0 * np.pi)  # Uniform distribution
    else:
        C = kappa / (4.0 * np.pi * np.sinh(kappa))

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
        log_pdfs = np.log(C) + kappa * cos_angles  # (num_obs x num_exp)

        # Average over MC samples using log-sum-exp trick for numerical stability
        max_log_pdfs = np.max(log_pdfs, axis=1, keepdims=True)  # (num_obs x 1)
        log_mean_probs = max_log_pdfs.squeeze() + np.log(np.mean(np.exp(log_pdfs - max_log_pdfs), axis=1))

        # Accumulate log-likelihood across all observations
        loglik_total += np.sum(log_mean_probs)

    return -loglik_total


def fit_subject(subject_id, interpolation_method, data_folder, obs_tbl, targets_coords, verbose=True):
    """
    Fit the model for a single subject using specified interpolation method.

    Parameters
    ----------
    subject_id : str
        Subject identifier
    interpolation_method : str
        Either 'SH' or 'barycentric'
    data_folder : str
        Path to data folder
    obs_tbl : DataFrame
        Behavioral observations
    targets_coords : Coordinates
        Target direction coordinates
    verbose : bool
        Print progress messages

    Returns
    -------
    dict : Fitting results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject: {subject_id} | Method: {interpolation_method}")
        print(f"{'='*60}")

    t_start_total = time.time()

    try:
        # Get subject data
        subj_data = obs_tbl[obs_tbl['subject'] == subject_id]
        if verbose:
            print(f"Trials: {len(subj_data)}")

        # Load HRTF and create model
        subj_sofa_path = f'{data_folder}/hrtf_downloads/{subject_id}_FreeFieldCompMinPhase_48kHz.sofa'
        model = BayesianListener(subj_sofa_path)
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

        # Define likelihood function
        def fll(sigmas_log):
            return loglik(model, targets, resp_cart, resp_targets_idx, sigmas_log)
        # [log(sigma_ild), log(sigma_spectral), log(kappa_motor), log(sigma_prior)]
        # Parameter bounds (in log space)
        lb = np.array([np.log(0.1), np.log(0.1), np.log(1), np.log(1)])
        plb = np.array([np.log(.5), np.log(1), np.log(2), np.log(5)])
        pub = np.array([np.log(3), np.log(10), np.log(30), np.log(50)])
        ub = np.array([np.log(50), np.log(100), np.log(100), np.log(179.9)])

        # Grid search for initialization
        if verbose:
            print("Grid search...")
        grid_points = allcomb(
            np.log(np.linspace(1, 3, 1)),
            np.log(np.linspace(10, 15, 1)),
            np.log(np.linspace(10, 30, 1)),
            np.log(np.linspace(50, 40, 1))
        )

        # Evaluate grid sequentially (simpler and cleaner)
        t_grid_start = time.time()
        nll = np.array([fll(grid_points[i]) for i in range(len(grid_points))])
        t_grid = time.time() - t_grid_start

        sigmas_0 = grid_points[np.argmin(nll), :]
        if verbose:
            print(f"  Best grid NLL: {nll.min():.2f} ({t_grid:.1f}s)")

        # BADS optimization
        if verbose:
            print("BADS optimization...")
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
            print(f"  σ_ILD: {fitted_params[0]:.2f}, σ_spectral: {fitted_params[1]:.2f}, "
                  f"σ_motor: {fitted_params[2]:.2f}, σ_prior: {fitted_params[3]:.2f}")

        return {
            'subject': subject_id,
            'method': interpolation_method,
            'sigma_ild': fitted_params[0],
            'sigma_spectral': fitted_params[1],
            'sigma_motor': fitted_params[2],
            'sigma_prior': fitted_params[3],
            'nll': result['fval'],
            'nll_initial': nll.min(),
            'time_grid': t_grid,
            'time_bads': t_bads,
            'time_total': t_total,
            'n_trials': len(subj_data),
            'success': True
        }

    except Exception as e:
        print(f"ERROR fitting {subject_id} with {interpolation_method}: {e}")
        return {
            'subject': subject_id,
            'method': interpolation_method,
            'success': False,
            'error': str(e)
        }