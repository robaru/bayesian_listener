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
                                     log(kappa_motor), log(tau_prior)]
        where tau_prior = 1 / sigma_prior^2 (precision)
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
    tau_prior = sigmas[3]  # Precision: tau = 1/sigma^2
    sigma_prior = 1.0 / np.sqrt(tau_prior)  # Convert to sigma for model

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

        # [log(sigma_ild), log(sigma_spectral), log(kappa_motor), log(tau_prior)]
        # Parameter bounds (in log space)
        # Motor noise bounds defined in sigma (degrees), then converted to kappa.
        # sigma2kappa is monotonically decreasing, so bounds flip:
        #   sigma:  lb < plb < pub < ub
        #   kappa:  ub > pub > plb > lb   (reversed)
        # tau_prior = 1/sigma_prior^2, so tau bounds also flip relative to sigma:
        #   sigma_prior: 1 < 5 < 50 < 179.9
        #   tau_prior:   1/179.9^2 < 1/50^2 < 1/5^2 < 1/1^2
        sigma_motor_lb = 1.0        # hard lower bound (deg)
        sigma_motor_plb = 2.0       # plausible lower bound (deg)
        sigma_motor_pub = 0.9 * sdL_deg  # plausible upper bound (deg)
        sigma_motor_ub = sdL_deg         # hard upper bound (deg)

        lb = np.array([np.log(0.1), np.log(0.1), np.log(sigma2kappa(sigma_motor_ub)), np.log(1.0/179.9**2)])
        plb = np.array([np.log(.5), np.log(1), np.log(sigma2kappa(sigma_motor_pub)), np.log(1.0/50.0**2)])
        pub = np.array([np.log(3), np.log(10), np.log(sigma2kappa(sigma_motor_plb)), np.log(1.0/5.0**2)])
        ub = np.array([np.log(50), np.log(100), np.log(sigma2kappa(sigma_motor_lb)), np.log(1.0/1.0**2)])

        # Grid search for initialization
        if verbose:
            print("  Grid search...")
        n = num_grid_points
        kappa_grid_lo = sigma2kappa(sigma_motor_pub)
        kappa_grid_hi = sigma2kappa(sigma_motor_plb)
        # Grid for tau_prior: tau = 1/sigma^2, so sigma in [40, 50] -> tau in [1/50^2, 1/40^2]
        tau_grid_lo = 1.0 / 50.0**2
        tau_grid_hi = 1.0 / 40.0**2
        grid_points = allcomb(
            np.log(np.linspace(1, 3, n)),
            np.log(np.linspace(10, 15, n)),
            np.log(np.linspace(kappa_grid_lo, kappa_grid_hi, n)),
            np.log(np.linspace(tau_grid_lo, tau_grid_hi, n))
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
        fitted_params[3] = 1.0 / np.sqrt(fitted_params[3])  # Convert tau to sigma_prior

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


# Default parameter values and bounds
# Note: tau_prior = 1/sigma_prior^2 (precision parameterization)
DEFAULT_PARAMS = {
    'sigma_itd': 0.569,
    'sigma_ild': 1.0,
    'sigma_spectral': 10.0,
    'sigma_motor': 15.0,
    'sigma_prior': 40.0  # Still stored as sigma for model interface
}

# Bounds for fitting (tau_prior used internally during optimization)
# tau_prior = 1/sigma^2, so bounds are inverted:
#   sigma: lb=1, plb=5, pub=50, ub=179.9
#   tau:   ub=1, pub=0.04, plb=0.0004, lb=3.1e-5
PARAM_BOUNDS = {
    'sigma_ild': {'lb': 0.1, 'plb': 0.5, 'pub': 3.0, 'ub': 50.0},
    'sigma_spectral': {'lb': 0.1, 'plb': 1.0, 'pub': 10.0, 'ub': 100.0},
    'tau_prior': {'lb': 1.0/179.9**2, 'plb': 1.0/50.0**2, 'pub': 1.0/5.0**2, 'ub': 1.0/1.0**2}
}


def fit_listener_partial(sofa_path, obs_tbl, targets_coords,
                         interpolation_method, params_to_fit,
                         fixed_params=None, subject_id=None,
                         num_repetitions=300, num_grid_points=1,
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
    targets_coords : Coordinates
        Target direction coordinates.
    interpolation_method : str
        HRTF interpolation method (e.g. 'SH', 'SHMAX', 'barycentric',
        'barumerli2023').
    params_to_fit : list of str
        List of parameter names to fit. Valid options:
        'sigma_ild', 'sigma_spectral', 'sigma_motor', 'sigma_prior'.
        Note: sigma_prior is fit internally as tau_prior (precision = 1/sigma^2)
        for better optimization landscape, but results are returned as sigma_prior.
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
    valid_params = ['sigma_ild', 'sigma_spectral', 'sigma_motor', 'sigma_prior']
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

        # Compute sdL for motor noise upper bound (if fitting motor noise)
        resp_hp = resp_coords.convert('horizontal-polar')
        resp_targets_hp = Coordinates(
            positions=targets_coords.convert('cartesian')[resp_targets_idx, :],
            convention='cartesian'
        ).convert('horizontal-polar')
        sdL_rad, _ = METRIC_FUNCTIONS['sdL'](resp_targets_hp, resp_hp)
        sdL_deg = np.rad2deg(sdL_rad)
        if verbose:
            print(f"  sdL: {sdL_deg:.2f}°")

        # Build bounds arrays for only the parameters being fit
        lb_list, plb_list, pub_list, ub_list = [], [], [], []
        for p in params_to_fit:
            if p == 'sigma_motor':
                # Motor noise bounds depend on sdL
                sigma_motor_lb = 1.0
                sigma_motor_plb = 2.0
                sigma_motor_pub = 0.9 * sdL_deg
                sigma_motor_ub = sdL_deg
                # Convert to kappa (bounds flip due to inverse relationship)
                lb_list.append(np.log(sigma2kappa(sigma_motor_ub)))
                plb_list.append(np.log(sigma2kappa(sigma_motor_pub)))
                pub_list.append(np.log(sigma2kappa(sigma_motor_plb)))
                ub_list.append(np.log(sigma2kappa(sigma_motor_lb)))
            elif p == 'sigma_prior':
                # Use tau_prior bounds (tau = 1/sigma^2)
                bounds = PARAM_BOUNDS['tau_prior']
                lb_list.append(np.log(bounds['lb']))
                plb_list.append(np.log(bounds['plb']))
                pub_list.append(np.log(bounds['pub']))
                ub_list.append(np.log(bounds['ub']))
            else:
                bounds = PARAM_BOUNDS[p]
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
            kappa_motor = sigma2kappa(all_params['sigma_motor'])

            # Override with fitted values
            for i, p in enumerate(params_to_fit):
                val = np.exp(x_log[i])
                if p == 'sigma_ild':
                    sigma_ild = val
                elif p == 'sigma_spectral':
                    sigma_spectral = val
                elif p == 'sigma_motor':
                    kappa_motor = val  # Already in kappa space
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

            return loglik(model, targets, resp_cart, resp_targets_idx, sigmas_log,
                          num_repetitions=num_repetitions)

        # Grid search for initialization
        if verbose:
            print("  Grid search...")
        n = num_grid_points

        # Build grid for each parameter being fit
        grid_arrays = []
        for p in params_to_fit:
            if p == 'sigma_motor':
                kappa_grid_lo = sigma2kappa(0.9 * sdL_deg)
                kappa_grid_hi = sigma2kappa(2.0)
                grid_arrays.append(np.log(np.linspace(kappa_grid_lo, kappa_grid_hi, n)))
            elif p == 'sigma_ild':
                grid_arrays.append(np.log(np.linspace(1, 3, n)))
            elif p == 'sigma_spectral':
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
            if p == 'sigma_motor':
                final_params[p] = kappa2sigma(fitted_x[i])
            elif p == 'sigma_prior':
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
            'sigma_motor': final_params['sigma_motor'],
            'sigma_prior': final_params['sigma_prior'],
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
            'params_fitted': params_to_fit,
            'success': False,
            'error': str(e)
        }