import numpy as np
import sys
from pathlib import Path
import time
from itertools import product
from pybads import BADS
from scipy.special import i0, i1
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
from bayesian_listener import BayesianListener
import pyfar as pf
from bayesian_listener.metrics import METRIC_FUNCTIONS

def allcomb(*arrays):
    """Cartesian product of input arrays (equivalent to MATLAB allcomb)."""
    return np.array(list(product(*arrays)))

def wrap_to_pi(rad):
    """Wrap angles to [-pi, pi)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi

def von_mises_loglik_mc(kappa, resp_lat, est_lat_mc):
    """
    Negative log-likelihood for von Mises with Monte Carlo estimates.

    Parameters
    ----------
    kappa : float
        Von Mises concentration parameter.
    resp_lat : ndarray
        Observed lateral angles (n_obs,) in radians.
    est_lat_mc : ndarray
        Monte Carlo model predictions (n_obs x n_mc) in radians.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    log_C = -np.log(2 * np.pi * i0(kappa))
    lat_diff = resp_lat[:, None] - est_lat_mc
    log_pdfs = log_C + kappa * np.cos(lat_diff)
    max_log_pdfs = np.max(log_pdfs, axis=1, keepdims=True)
    log_mean_probs = max_log_pdfs[:, 0] + np.log(np.mean(np.exp(log_pdfs - max_log_pdfs), axis=1))
    return -np.sum(log_mean_probs)

def fit_kappa_ml(resp_lat, est_lat_mc):
    """
    Fit von Mises concentration parameter via maximum likelihood.

    Parameters
    ----------
    resp_lat : ndarray
        Observed lateral angles (n_obs,) in radians.
    est_lat_mc : ndarray
        Monte Carlo model predictions (n_obs x n_mc) in radians.

    Returns
    -------
    float
        Fitted kappa (concentration parameter).
    """
    result = minimize_scalar(von_mises_loglik_mc, bounds=(0.1, 1000.0),
                            args=(resp_lat, est_lat_mc), method='bounded')
    return result.x

def _bessel_ratio(kappa):
    """Compute i1(kappa)/i0(kappa) with asymptotic expansion for large kappa."""
    if kappa < 1e-10:
        return 0.0
    if kappa > 500:
        # Asymptotic expansion: i1(k)/i0(k) ≈ 1 - 1/(2k) - 1/(8k^2) - ...
        return 1.0 - 1.0 / (2.0 * kappa) - 1.0 / (8.0 * kappa**2)
    return float(i1(kappa) / i0(kappa))

def sigma_to_kappa(sigma):
    """
    Convert circular standard deviation (degrees) to von Mises concentration.
    Uses Bessel function ratio: solves i1(κ)/i0(κ) = exp(-σ²/2).

    Parameters
    ----------
    sigma : float
        Circular standard deviation in degrees.

    Returns
    -------
    float
        Von Mises concentration parameter kappa.
    """
    if sigma >= 180.0:
        return 1e-6  # Essentially uniform

    if sigma < 1e-3:
        return 1e6  # Very high concentration

    # Target mean resultant length from sigma
    # sigma = sqrt(-2 * log(R))  =>  R = exp(-sigma^2 / 2)
    sigma_rad = np.deg2rad(sigma)
    R_target = np.exp(-sigma_rad**2 / 2)

    # For high concentration (R close to 1), use asymptotic inversion:
    # i1(k)/i0(k) ≈ 1 - 1/(2k) => k ≈ 1/(2*(1-R))
    if R_target > 0.99:
        kappa_init = 1.0 / (2.0 * (1.0 - R_target))
        # Refine with one Newton step using Bessel ratio
        R_est = _bessel_ratio(kappa_init)
        if abs(R_est - R_target) < 1e-10:
            return kappa_init

    # Solve: i1(kappa) / i0(kappa) - R_target = 0
    def objective(kappa):
        return _bessel_ratio(kappa) - R_target

    try:
        kappa = brentq(objective, 1e-6, 1e4)
    except ValueError:
        # Fallback to asymptotic approximation
        kappa = 1.0 / (2.0 * (1.0 - R_target))

    return kappa

def kappa_to_sigma(kappa):
    """
    Convert von Mises concentration to circular standard deviation (degrees).
    Uses Bessel function ratio: σ = sqrt(-2 * log(i1(κ)/i0(κ))).

    Parameters
    ----------
    kappa : float
        Von Mises concentration parameter.

    Returns
    -------
    float
        Circular standard deviation in degrees.
    """
    if kappa < 1e-6:
        return 180.0
    R = _bessel_ratio(kappa)
    return np.rad2deg(np.sqrt(-2 * np.log(R)))

def estimate_motor_noise(model, obs_tbl, targets_coords, subject_id=None,
                         num_repetitions=200, seed=42):
    """
    Estimate motor noise from behavioral data using ITD+ILD cues only.

    This function estimates motor noise by:
    1. Using only ITD+ILD features to predict responses (no spectral cues)
    2. Generating Monte Carlo samples of lateral predictions
    3. Fitting a von Mises distribution to lateral angle errors
    4. Converting the fitted concentration to sigma_motor in degrees

    The approach filters responses to ±80° lateral for numerical stability.

    Parameters
    ----------
    model : BayesianListener
        Model instance with prepared features.
    obs_tbl : DataFrame
        Behavioral observations with columns:
        'azi_response', 'ele_response', 'azi_target', 'ele_target'.
        If subject_id is provided, must also contain 'participant'.
    targets_coords : pyfar.Coordinates
        Target direction coordinates.
    subject_id : str, optional
        Subject identifier for filtering obs_tbl.
    num_repetitions : int
        Number of Monte Carlo samples for predictions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'sigma_motor': Estimated motor noise in degrees
        - 'kappa_motor': Fitted concentration parameter
        - 'n_trials': Number of trials used (after filtering)
    """
    np.random.seed(seed)

    # Fixed ITD/ILD noise (from parameter recovery notebook)
    sigma_itd = 0.569
    sigma_ild = 1.0

    # Get subject data
    if subject_id is not None:
        subj_data = obs_tbl[obs_tbl['participant'] == subject_id]
    else:
        subj_data = obs_tbl

    # Template ITD+ILD features
    template_itd = model.template.itd.flatten()
    template_ild = model.template.ild.flatten()
    template_lat = model.template.coords.lateral

    # Target ITD+ILD features
    target_indices = model.coords.find_nearest(targets_coords)[0][0]
    target_itd = model.itd[target_indices].flatten()
    target_ild = model.ild[target_indices].flatten()

    # Response lateral angles
    resp_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(subj_data['azi_response'].values),
        np.deg2rad(subj_data['ele_response'].values),
        np.ones(len(subj_data)),
    )
    resp_lat = resp_coords.lateral

    # Map trials to targets
    targ_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(subj_data['azi_target'].values),
        np.deg2rad(subj_data['ele_target'].values),
        np.ones(len(subj_data)),
    )
    targ_lat = targ_coords.lateral
    trial_target_indices = targets_coords.find_nearest(targ_coords)[0][0]

    # Monte Carlo estimates for each trial
    n_trials = len(subj_data)
    est_lat_mc = np.zeros((n_trials, num_repetitions))

    for i, t_idx in enumerate(trial_target_indices):
        # Add noise to target features
        noisy_itd = target_itd[t_idx] + np.random.normal(0, sigma_itd, num_repetitions)
        noisy_ild = target_ild[t_idx] + np.random.normal(0, sigma_ild, num_repetitions)

        # Compute log-likelihood for each template direction
        itd_diff = (template_itd[None, :] - noisy_itd[:, None]) / sigma_itd
        ild_diff = (template_ild[None, :] - noisy_ild[:, None]) / sigma_ild
        loglik = -0.5 * (itd_diff**2 + ild_diff**2)

        # Find best matching template for each MC sample
        best_indices = np.argmax(loglik, axis=1)
        est_lat_mc[i, :] = template_lat[best_indices]

    # Filter to ±60° lateral for numerical stability
    mask = np.abs(targ_lat) <= np.deg2rad(30)
    resp_lat_filt = resp_lat[mask]
    est_lat_mc_filt = est_lat_mc[mask, :]

    if np.sum(mask) < 3:
        return {
            'sigma_motor': np.nan,
            'kappa_motor': np.nan,
            'n_trials': int(np.sum(mask))
        }

    # Fit von Mises concentration
    kappa_fit = fit_kappa_ml(resp_lat_filt, est_lat_mc_filt)
    sigma_motor = kappa_to_sigma(kappa_fit)

    return {
        'sigma_motor': sigma_motor,
        'kappa_motor': kappa_fit,
        'n_trials': int(np.sum(mask))
    }

def loglik(model, targets, responses, resp_targets_idx, sigmas_log,
           num_repetitions=200):
    """
    Negative log-likelihood function for parameter fitting.

    Parameters
    ----------
    model : BayesianListener
        Model instance (for running inference)
    targets : ndarray
        Target feature vectors (n_targets x n_features)
    responses : pyfar.Coordinates
        Observed responses as spatial coordinates.
    resp_targets_idx : ndarray
        Index mapping each response to a target direction (n_obs,)
    sigmas_log : ndarray
        Log-transformed parameters [log(sigma_ild), log(sigma_spectral),
                                     log(kappa_motor), log(tau_prior)]
        where tau_prior = 1 / sigma_prior^2 (precision)
    num_repetitions : int
        Number of Monte Carlo repetitions for likelihood evaluation.

    Returns
    -------
    neglik : float
        Negative log-likelihood
    """
    # Extract numpy arrays at computation boundary
    responses_cart = responses.cartesian

    # Parameters
    num_exp = num_repetitions
    sigma_itd = 0.569  # Fixed ITD noise

    # Convert from log space
    sigmas = np.exp(sigmas_log)
    sigma_ild = sigmas[0]
    sigma_spectral = sigmas[1]
    kappa = sigmas[2]  # Motor noise as concentration parameter
    tau_prior = sigmas[3]  # Precision: tau = 1/sigma^2
    sigma_prior = 1.0 / np.sqrt(tau_prior)  # Convert to sigma for model

    # Set model parameters (without motor noise for now)
    model.parameters = {
        'sigma_itd': sigma_itd,
        'sigma_ild': sigma_ild,
        'sigma_spectral': sigma_spectral,
        'sigma_prior': sigma_prior,
        'kappa_motor': 0
    }

    # Run model WITHOUT motor noise to get MAP predictions
    posterior = model.infer(targets, repetitions=num_exp)

    # Get MAP estimates without motor noise
    doa_estimations = model.estimate(posterior, kappa_motor=False).cartesian  # (n_targets x num_exp x 3)

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




# Default parameter values and bounds
# Note: kappa_motor and tau_prior are used internally during optimization
DEFAULT_PARAMS = {
    'sigma_itd': 0.569,
    'sigma_ild': 1.0,
    'sigma_spectral': 10.0,
    'kappa_motor': sigma_to_kappa(15.0),  # ~14.2 (from 15 deg via Bessel)
    'sigma_prior': 40.0  # Still stored as sigma for model interface
}

# Bounds for fitting
# kappa_motor: higher kappa = less noise (more concentrated)
#   sigma 4° -> kappa ~190, sigma 60° -> kappa ~1.6
# tau_prior = 1/sigma^2, so bounds are inverted
PARAM_BOUNDS = {
    'sigma_ild': {'lb': 0.1, 'plb': 0.5, 'pub': 3.0, 'ub': 50.0},
    'sigma_spectral': {'lb': 0.1, 'plb': 1.0, 'pub': 10.0, 'ub': 50.0},
    'kappa_motor': {'lb': sigma_to_kappa(80.0), 'plb': sigma_to_kappa(40.0),
                    'pub': sigma_to_kappa(5.0), 'ub': sigma_to_kappa(2.0)},
    'tau_prior': {'lb': 1.0/179.9**2, 'plb': 1.0/50.0**2, 'pub': 1.0/5.0**2, 'ub': 1.0/1.0**2}
}


def fit_listener(sofa_path, obs_tbl, targets_coords,
                 interpolation_method, subject_id=None,
                 num_repetitions=200, num_repetitions_motor=200,
                 num_grid_points=1, fix_sigma_ild=True,
                 motor_estimation_seed=42, verbose=True):
    """
    Fit the Bayesian listener model with motor noise estimated from ITD+ILD.

    This is the recommended fitting procedure that:
    1. Estimates motor noise from lateral errors using ITD+ILD cues only
    2. Fixes motor noise and optionally ILD noise at estimated/default values
    3. Fits only sigma_spectral and sigma_prior parameters

    This approach follows the methodology from the parameter recovery validation.

    Parameters
    ----------
    sofa_path : str
        Path to the participant's SOFA file.
    obs_tbl : DataFrame
        Behavioral observations. Must contain columns:
        'azi_response', 'ele_response', 'azi_target', 'ele_target'.
        If subject_id is provided, must also contain 'participant'.
    targets_coords : pyfar.Coordinates
        Target direction coordinates.
    interpolation_method : str
        HRTF interpolation method (e.g. 'SH', 'SHMAX', 'barycentric',
        'barumerli2023').
    subject_id : str, optional
        Subject identifier. If provided, obs_tbl is filtered to this
        subject. If None, all rows in obs_tbl are used.
    num_repetitions : int
        Number of Monte Carlo repetitions for likelihood evaluation during
        parameter fitting (default: 300).
    num_repetitions_motor : int
        Number of Monte Carlo repetitions for motor noise estimation
        (default: 200).
    num_grid_points : int
        Number of grid points per parameter dimension for initialization.
    fix_sigma_ild : bool
        If True, fixes sigma_ild to 1.0 dB (default: True).
        If False, sigma_ild is also fitted.
    motor_estimation_seed : int
        Random seed for motor noise estimation (default: 42).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Fitting results containing:
        - Fitted parameters (sigma_spectral, sigma_prior)
        - Fixed parameters (sigma_motor, sigma_ild, sigma_itd)
        - Motor estimation results (kappa_motor, n_trials_motor)
        - NLL and timing information
    """
    label = subject_id or Path(sofa_path).stem
    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject: {label} | Method: {interpolation_method}")
        print(f"{'='*60}")

    t_start_total = time.time()

    try:
        # Load HRTF and create model
        model = BayesianListener(sofa_path)
        model.prepare_features(interpolation=interpolation_method)

        # Step 1: Estimate motor noise from ITD+ILD
        if verbose:
            print("  Estimating motor noise from ITD+ILD...")
        t_motor_start = time.time()
        motor_result = estimate_motor_noise(
            model=model,
            obs_tbl=obs_tbl,
            targets_coords=targets_coords,
            subject_id=subject_id,
            num_repetitions=num_repetitions_motor,
            seed=motor_estimation_seed
        )
        t_motor = time.time() - t_motor_start

        kappa_motor_est = motor_result['kappa_motor']
        if verbose:
            print(f"  Motor noise: κ_motor = {kappa_motor_est:.2f} "
                  f"(σ = {motor_result['sigma_motor']:.2f}°, "
                  f"n = {motor_result['n_trials']}) ({t_motor:.1f}s)")

        # Step 2: Fit remaining parameters with motor noise fixed
        if verbose:
            print("  Fitting spectral and prior noise...")

        # Determine which parameters to fit
        if fix_sigma_ild:
            params_to_fit = ['sigma_spectral', 'sigma_prior']
            fixed_params = {
                'kappa_motor': kappa_motor_est,
                'sigma_ild': 1.0
            }
        else:
            params_to_fit = ['sigma_ild', 'sigma_spectral', 'sigma_prior']
            fixed_params = {
                'kappa_motor': kappa_motor_est
            }

        # Call fit_listener_partial
        result = fit_listener_partial(
            sofa_path=sofa_path,
            obs_tbl=obs_tbl,
            targets_coords=targets_coords,
            interpolation_method=interpolation_method,
            params_to_fit=params_to_fit,
            fixed_params=fixed_params,
            subject_id=subject_id,
            num_repetitions=num_repetitions,
            num_grid_points=num_grid_points,
            verbose=False  # We handle verbose output here
        )

        # Add motor estimation info to results
        result['sigma_motor_method'] = 'itd_ild_estimation'
        result['kappa_motor'] = motor_result['kappa_motor']
        result['n_trials_motor'] = motor_result['n_trials']
        result['time_motor'] = t_motor

        t_total = time.time() - t_start_total
        result['time_total'] = t_total

        if verbose:
            print(f"  Final NLL: {result['nll']:.2f}")
            print(f"  Total time: {t_total:.1f}s")
            print(f"  σ_ILD: {result['sigma_ild']:.2f}, "
                  f"σ_spectral: {result['sigma_spectral']:.2f}, "
                  f"σ_motor: {result['sigma_motor']:.2f}°, "
                  f"σ_prior: {result['sigma_prior']:.2f}")

        return result

    except Exception as e:
        print(f"ERROR fitting {label} with {interpolation_method}: {e}")
        return {
            'subject': label,
            'method': interpolation_method,
            'success': False,
            'error': str(e)
        }


def fit_listener_partial(sofa_path, obs_tbl, targets_coords,
                         interpolation_method, params_to_fit,
                         fixed_params=None, subject_id=None,
                         num_repetitions=200, num_grid_points=1,
                         verbose=True):
    """
    Fit the Bayesian listener model with a subset of parameters.

    Parameters
    ----------
    sofa_path : str
        Path to the participant's SOFA file.
    obs_tbl : DataFrame
        Behavioral observations. Must contain columns:
        'azi_response', 'ele_response', 'azi_target', 'ele_target'.
        If subject_id is provided, must also contain 'participant'.
    targets_coords : pyfar.Coordinates
        Target direction coordinates.
    interpolation_method : str
        HRTF interpolation method (e.g. 'SH', 'SHMAX', 'barycentric',
        'barumerli2023').
    params_to_fit : list of str
        List of parameter names to fit. Valid options:
        'sigma_ild', 'sigma_spectral', 'kappa_motor', 'sigma_prior'.
        Note: sigma_prior is fit internally as tau_prior (precision = 1/sigma^2)
        for better optimization landscape, but results are returned as sigma_prior.
        kappa_motor is the von Mises concentration parameter for motor noise.
    fixed_params : dict, optional
        Dictionary of fixed parameter values. Parameters not in params_to_fit
        and not specified here will use DEFAULT_PARAMS values.
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
    valid_params = ['sigma_ild', 'sigma_spectral', 'kappa_motor', 'sigma_prior']
    for p in params_to_fit:
        if p not in valid_params:
            raise ValueError(f"Invalid parameter '{p}'. Valid options: {valid_params}")

    if fixed_params is None:
        fixed_params = {}

    # Build full parameter dict with defaults, then override with fixed values
    all_params = DEFAULT_PARAMS.copy()
    all_params.update(fixed_params)

    label = subject_id or Path(sofa_path).stem
    if verbose:
        print(f"\n{'='*60}")
        print(f"Subject: {label} | Method: {interpolation_method}")
        print(f"Fitting: {params_to_fit}")
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
        target_indices = model.coords.find_nearest(targets_coords)[0][0]
        targets = model.represent()[target_indices, :]

        # Build response coordinates
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

        # Build bounds arrays for only the parameters being fit
        lb_list, plb_list, pub_list, ub_list = [], [], [], []
        for p in params_to_fit:
            if p == 'sigma_prior':
                bounds_key = 'tau_prior'
            else:
                bounds_key = p
            bounds = PARAM_BOUNDS[bounds_key]
            lb_list.append(np.log(bounds['lb']))
            plb_list.append(np.log(bounds['plb']))
            pub_list.append(np.log(bounds['pub']))
            ub_list.append(np.log(bounds['ub']))

        lb = np.array(lb_list)
        plb = np.array(plb_list)
        pub = np.array(pub_list)
        ub = np.array(ub_list)

        def fll(x_log):
            """Likelihood function that maps fitted params to full param set."""
            # Start with fixed/default values
            sigma_itd = all_params['sigma_itd']
            sigma_ild = all_params['sigma_ild']
            sigma_spectral = all_params['sigma_spectral']
            tau_prior = 1.0 / all_params['sigma_prior']**2  # Convert default sigma to tau
            kappa_motor = all_params['kappa_motor']

            # Override with fitted values
            for i, p in enumerate(params_to_fit):
                val = np.exp(x_log[i])
                if p == 'sigma_ild':
                    sigma_ild = val
                elif p == 'sigma_spectral':
                    sigma_spectral = val
                elif p == 'kappa_motor':
                    kappa_motor = val
                elif p == 'sigma_prior':
                    tau_prior = val  # Fitted as tau_prior directly

            # Build full sigmas_log array for loglik function
            # Note: loglik expects tau_prior in position 3
            sigmas_log = np.array([
                np.log(sigma_ild),
                np.log(sigma_spectral),
                np.log(kappa_motor),
                np.log(tau_prior)
            ])

            return loglik(model, targets, resp_coords, resp_targets_idx, sigmas_log,
                          num_repetitions=num_repetitions)

        # Grid search for initialization
        if verbose:
            print("  Grid search...")
        n = num_grid_points

        # Build grid for each parameter being fit
        grid_arrays = []
        for p in params_to_fit:
            if p == 'sigma_spectral':
                grid_arrays.append(np.log(np.linspace(10, 15, n)))
            elif p == 'sigma_prior':
                # Grid in tau space: sigma in [40, 50] -> tau in [1/50^2, 1/40^2]
                tau_grid_lo = 1.0 / 50.0**2
                tau_grid_hi = 1.0 / 40.0**2
                grid_arrays.append(np.log(np.linspace(tau_grid_lo, tau_grid_hi, n)))

        grid_points = allcomb(*grid_arrays) if len(grid_arrays) > 1 else grid_arrays[0].reshape(-1, 1)

        t_grid_start = time.time()
        nll = np.array([fll(grid_points[i]) for i in range(len(grid_points))])
        t_grid = time.time() - t_grid_start

        x0 = grid_points[np.argmin(nll), :]
        if verbose:
            print(f"  Best grid NLL: {nll.min():.2f} ({t_grid:.1f}s)")

        # BADS optimization
        if verbose:
            print("  BADS optimization...")
        options = {"tol_mesh": 1e-2}

        t_bads_start = time.time()
        bads = BADS(fll, x0, lb, ub, plb, pub, options=options)
        result = bads.optimize()
        t_bads = time.time() - t_bads_start

        # Build result dictionary with all parameters
        final_params = all_params.copy()
        fitted_x = np.exp(result['x'])
        for i, p in enumerate(params_to_fit):
            if p == 'sigma_prior':
                # Convert tau_prior back to sigma_prior
                final_params[p] = 1.0 / np.sqrt(fitted_x[i])
            else:
                final_params[p] = fitted_x[i]

        t_total = time.time() - t_start_total

        if verbose:
            print(f"  Final NLL: {result['fval']:.2f} ({t_bads:.1f}s)")
            print(f"  Total time: {t_total:.1f}s")
            print(f"  Fitted parameters:")
            for p in params_to_fit:
                print(f"    {p}: {final_params[p]:.2f}")

        return {
            'subject': label,
            'method': interpolation_method,
            'params_fitted': params_to_fit,
            'sigma_itd': final_params['sigma_itd'],
            'sigma_ild': final_params['sigma_ild'],
            'sigma_spectral': final_params['sigma_spectral'],
            'kappa_motor': final_params['kappa_motor'],
            'sigma_motor': kappa_to_sigma(final_params['kappa_motor']),
            'sigma_prior': final_params['sigma_prior'],
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
            'params_fitted': params_to_fit,
            'success': False,
            'error': str(e)
        }

