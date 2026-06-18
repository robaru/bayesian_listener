r"""Two-stage maximum-likelihood fitting of the Bayesian listener model.

Implements the procedure of :footcite:t:`barumerli2026`, §2.2:

1. **Stage 1 — motor noise.**  Estimate :math:`\\kappa_m` from a
   restricted lateral-only likelihood (Eq. 12 of :footcite:t:`barumerli2026`) using
   ITD+ILD cues only.  See :func:`estimate_motor_noise`, :func:`fit_kappa_ml`.
2. **Stage 2 — spectral and prior noise.**  Hold :math:`\\hat{\\kappa}_m`
   fixed and jointly fit :math:`\\sigma_{\\mathrm{mon}}` and
   :math:`\\sigma_{\\mathrm{prior}}` to the full-sphere likelihood
   (Eq. 13 of :footcite:t:`barumerli2026`) via BADS :footcite:t:`acerbi2017`.

The high-level wrapper :func:`fit_listener` runs both stages.
"""
import types
import numpy as np
from pathlib import Path
import time
from itertools import product
from pybads import BADS
from scipy.special import i0, i1
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
from bayesian_listener import BayesianListener
import pyfar as pf

def allcomb(*arrays):
    """Cartesian product of input arrays (equivalent to MATLAB ``allcomb``).

    Parameters
    ----------
    *arrays : sequence of 1-D array-like
        Input arrays to combine.  Each must be 1-D.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of shape ``(prod(len(a) for a in arrays), len(arrays))`` whose
        rows enumerate every combination of one element from each input.

    Examples
    --------
    >>> allcomb([0, 1], [10, 20])
    array([[ 0, 10],
           [ 0, 20],
           [ 1, 10],
           [ 1, 20]])
    """
    return np.array(list(product(*arrays)))

def von_mises_loglik_mc(kappa, resp_lat, est_lat_mc):
    r"""Negative log-likelihood of a von Mises with Monte Carlo predictions.

    Implements the lateral-only likelihood of Eq. 12 of :footcite:t:`barumerli2026`,
    approximated by averaging the von Mises pdf over ``n_mc`` Monte Carlo
    samples per observation.

    Parameters
    ----------
    kappa : float
        Von Mises concentration :math:`\kappa` (positive).
    resp_lat : :class:`numpy.ndarray`
        Observed lateral angles, shape ``(n_obs,)`` in radians.
    est_lat_mc : :class:`numpy.ndarray`
        Monte Carlo model predictions, shape ``(n_obs, n_mc)`` in radians.

    Returns
    -------
    float
        Negative log-likelihood in nats.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> resp = rng.normal(scale=0.1, size=20)
    >>> mc   = resp[:, None] + rng.normal(scale=0.1, size=(20, 50))
    >>> nll  = von_mises_loglik_mc(50.0, resp, mc)
    >>> bool(np.isfinite(nll))
    True
    """
    log_C = -np.log(2 * np.pi * i0(kappa))
    lat_diff = resp_lat[:, None] - est_lat_mc
    log_pdfs = log_C + kappa * np.cos(lat_diff)
    max_log_pdfs = np.max(log_pdfs, axis=1, keepdims=True)
    log_mean_probs = max_log_pdfs[:, 0] + np.log(np.mean(np.exp(log_pdfs - max_log_pdfs), axis=1))
    return -np.sum(log_mean_probs)

def fit_kappa_ml(resp_lat, est_lat_mc):
    r"""Fit the von Mises concentration :math:`\kappa` by 1-D bounded ML search.

    Wraps :func:`scipy.optimize.minimize_scalar` (``method='bounded'``,
    Brent's method) over the bracket :math:`\kappa \in [0.1, 1000]`.

    Parameters
    ----------
    resp_lat : :class:`numpy.ndarray`
        Observed lateral angles, shape ``(n_obs,)`` in radians.
    est_lat_mc : :class:`numpy.ndarray`
        Monte Carlo model predictions, shape ``(n_obs, n_mc)`` in radians.

    Returns
    -------
    float
        Maximum-likelihood concentration :math:`\hat{\kappa}`.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> resp = rng.normal(scale=0.05, size=200)
    >>> mc   = resp[:, None] + rng.normal(scale=0.05, size=(200, 100))
    >>> kappa = fit_kappa_ml(resp, mc)
    >>> bool(0.1 <= kappa <= 1000.0)
    True
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
    r"""Convert a circular standard deviation in degrees to a von Mises concentration.

    Solves :math:`I_1(\kappa)/I_0(\kappa) = \exp(-\sigma^2/2)` for
    :math:`\kappa` via :func:`scipy.optimize.brentq` on the Bessel-ratio
    identity.  Falls back to the asymptotic approximation
    :math:`\kappa \approx 1/(2(1-R))` for very small :math:`\sigma`.

    Parameters
    ----------
    sigma : float
        Circular standard deviation in degrees.  Saturates to a near-uniform
        :math:`\kappa = 10^{-6}` for ``sigma >= 180``.

    Returns
    -------
    float
        Von Mises concentration :math:`\kappa`.

    Examples
    --------
    >>> bool(abs(kappa_to_sigma(sigma_to_kappa(15.0)) - 15.0) < 1e-3)
    True
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

    # Solve: i1(kappa) / i0(kappa) - R_target = 0  # noqa: ERA001
    def objective(kappa):
        return _bessel_ratio(kappa) - R_target

    try:
        kappa = brentq(objective, 1e-6, 1e4)
    except ValueError:
        # Fallback to asymptotic approximation
        kappa = 1.0 / (2.0 * (1.0 - R_target))

    return kappa

def kappa_to_sigma(kappa):
    r"""Convert a von Mises concentration to a circular standard deviation in degrees.

    Computes :math:`\sigma = \sqrt{-2 \log(I_1(\kappa)/I_0(\kappa))}` and
    converts to degrees.

    Parameters
    ----------
    kappa : float
        Von Mises concentration :math:`\kappa` (positive).  Saturates to
        180° for :math:`\kappa < 10^{-6}`.

    Returns
    -------
    float
        Circular standard deviation in degrees.

    Examples
    --------
    >>> round(float(kappa_to_sigma(50.0)), 1)
    8.1
    """
    if kappa < 1e-6:
        return 180.0
    R = _bessel_ratio(kappa)
    return np.rad2deg(np.sqrt(-2 * np.log(R)))

def estimate_motor_noise(model, obs_tbl, targets_coords, subject_id=None,
                         num_repetitions=200, seed=42):
    r"""Estimate motor noise from behavioural data using ITD + ILD cues only (Stage 1).

    Implements Eq. 12 of :footcite:t:`barumerli2026`:

    1. Build ITD + ILD predictions from the model template (no spectral cues).
    2. Draw ``num_repetitions`` Monte Carlo lateral predictions per trial.
    3. Restrict to trials with target lateral angle :math:`|\alpha| \le 30^\circ`
       for numerical stability.
    4. Fit the von Mises concentration :math:`\hat{\kappa}_m` via
       :func:`fit_kappa_ml`.
    5. Convert to a circular SD :math:`\hat{\sigma}_m` in degrees through
       :func:`kappa_to_sigma`.

    Parameters
    ----------
    model : :class:`~bayesian_listener.BayesianListener`
        Model instance with :meth:`~bayesian_listener.BayesianListener.compute_template`
        already called.
    obs_tbl : :class:`pandas.DataFrame`
        Behavioural observations with columns
        ``'azi_response'``, ``'ele_response'``, ``'azi_target'``,
        ``'ele_target'`` (all in degrees).  If ``subject_id`` is given,
        must additionally contain a ``'participant'`` column.
    targets_coords : :class:`pyfar.Coordinates`
        Discrete target directions (one entry per unique presented direction).
    subject_id : str or None, default=None
        Participant identifier.  If ``None``, every row in ``obs_tbl`` is used.
    num_repetitions : int, default=200
        Number of Monte Carlo samples per trial.
    seed : int, default=42
        Seed for the noise generator.

    Returns
    -------
    dict
        Mapping with keys:

        - ``'sigma_motor'`` (float, degrees) — estimated motor SD
          :math:`\hat{\sigma}_m`.
        - ``'kappa_motor'`` (float) — fitted concentration
          :math:`\hat{\kappa}_m`.
        - ``'n_trials'`` (int) — number of trials retained after the
          ±30° lateral filter.

    See Also
    --------
    fit_listener : Full two-stage fitting wrapper.
    fit_kappa_ml : Underlying maximum-likelihood :math:`\kappa` fit.
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
    target_itd = model.target.itd[target_indices].flatten()
    target_ild = model.target.ild[target_indices].flatten()

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

    # Filter to ±30° lateral for numerical stability
    mask = np.abs(targ_lat) <= np.deg2rad(30)
    resp_lat_filt = resp_lat[mask]
    est_lat_mc_filt = est_lat_mc[mask, :]

    if np.sum(mask) < 3:
        return {
            'sigma_motor': np.nan,
            'kappa_motor': np.nan,
            'n_trials': int(np.sum(mask)),
        }

    # Fit von Mises concentration
    kappa_fit = fit_kappa_ml(resp_lat_filt, est_lat_mc_filt)
    sigma_motor = kappa_to_sigma(kappa_fit)

    return {
        'sigma_motor': sigma_motor,
        'kappa_motor': kappa_fit,
        'n_trials': int(np.sum(mask)),
    }

def negloglik(model, targets, responses, resp_targets_idx, sigmas_log,
              num_repetitions=200):
    r"""Negative full-sphere log-likelihood for BADS optimisation
    (Eq. 13 of :footcite:t:`barumerli2026`).

    For each observation, runs Monte Carlo inference with the supplied
    parameters, builds the von Mises–Fisher pdf around each MC prediction,
    and accumulates :math:`-\log p(\hat{\boldsymbol{\varphi}}^* \mid
    \boldsymbol{\varphi}, \boldsymbol{\theta})`.

    Parameters
    ----------
    model : :class:`~bayesian_listener.BayesianListener`
        Model instance whose ``parameters`` and ``target`` are overwritten
        in place.
    targets : :class:`~bayesian_listener.auditory_representation.Barumerli2023`
        Subset of the model template at the unique presented directions.
    responses : :class:`pyfar.Coordinates`
        Observed responses, shape ``(n_obs,)``.
    resp_targets_idx : :class:`numpy.ndarray`
        Integer mapping of each response to its target direction, shape
        ``(n_obs,)``.
    sigmas_log : :class:`numpy.ndarray`
        Log-transformed parameter vector of shape ``(4,)``:

        - ``sigmas_log[0]`` — :math:`\log \sigma_{\mathrm{ild}}` (dB).
        - ``sigmas_log[1]`` — :math:`\log \sigma_{\mathrm{mon}}` (dB).
        - ``sigmas_log[2]`` — :math:`\log \kappa_m`.
        - ``sigmas_log[3]`` — :math:`\log \tau_{\mathrm{prior}}` with
          :math:`\tau_{\mathrm{prior}} = 1 / \sigma_{\mathrm{prior}}^2`
          (precision parametrisation for a better optimisation landscape).
    num_repetitions : int, default=200
        Monte Carlo repetitions per trial in
        :meth:`~bayesian_listener.BayesianListener.infer`.

    Returns
    -------
    float
        Negative log-likelihood in nats.
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
        'kappa_motor': 0,
    }

    # Run model WITHOUT motor noise to get MAP predictions
    model.target = targets
    posterior = model.infer(repetitions=num_exp)

    # Get MAP estimates without motor noise; shape: (n_targets, num_exp, 3)
    doa_estimations = model.estimate(posterior, kappa_motor=False).cartesian

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
        log_mean_probs = (
            max_log_pdfs.squeeze()
            + np.log(np.mean(np.exp(log_pdfs - max_log_pdfs), axis=1))
        )

        # Accumulate log-likelihood across all observations
        loglik_total += np.sum(log_mean_probs)

    return -loglik_total




#: Read-only mapping of default noise-parameter values.  Used as the
#: fallback in :func:`fit_listener_partial` for any parameter that is
#: neither in ``params_to_fit`` nor ``fixed_params``.  Pass ``fixed_params``
#: to override individual values; do not mutate this object directly.
DEFAULT_PARAMS = types.MappingProxyType({
    'sigma_itd': 0.569,
    'sigma_ild': 1.0,
    'sigma_spectral': 10.0,
    'kappa_motor': sigma_to_kappa(15.0),
    'sigma_prior': 40.0,
})

#: Read-only mapping of BADS optimiser search bounds for each free parameter.
#: Keys are ``'sigma_ild'``, ``'sigma_spectral'``, ``'kappa_motor'``, and
#: ``'tau_prior'`` (precision :math:`\tau = 1/\sigma_{\mathrm{prior}}^2`,
#: used internally for a better optimisation landscape).  Each value is a
#: read-only mapping with keys ``'lb'``, ``'plb'``, ``'pub'``, ``'ub'``
#: (lower / plausible-lower / plausible-upper / upper bounds).
#: ``'kappa_motor'`` bounds are von Mises concentrations (higher = less noise).
PARAM_BOUNDS = types.MappingProxyType({
    'sigma_ild':      types.MappingProxyType({'lb': 0.1,  'plb': 0.5,  'pub': 3.0,  'ub': 50.0}),
    'sigma_spectral': types.MappingProxyType({'lb': 0.1,  'plb': 1.0,  'pub': 10.0, 'ub': 50.0}),
    'kappa_motor':    types.MappingProxyType({
        'lb': round(sigma_to_kappa(80.0), 4),
        'plb': round(sigma_to_kappa(40.0), 4),
        'pub': round(sigma_to_kappa(5.0), 4),
        'ub': round(sigma_to_kappa(2.0), 4),
    }),
    'tau_prior':      types.MappingProxyType({
        'lb': round(1.0/179.9**2, 6),
        'plb': 0.0004,
        'pub': 0.04,
        'ub': 1.0,
    }),
})


def fit_listener(sofa_path, obs_tbl, targets_coords,
                 interpolation_method, subject_id=None,
                 num_repetitions=200, num_repetitions_motor=200,
                 num_grid_points=1, fix_sigma_ild=True,
                 motor_estimation_seed=42, verbose=True):
    r"""Run the full two-stage fit recommended in :footcite:t:`barumerli2026`.

    1. Estimate :math:`\hat{\kappa}_m` (Eq. 12) via :func:`estimate_motor_noise`.
    2. Hold :math:`\hat{\kappa}_m` fixed and jointly fit
       :math:`\sigma_{\mathrm{mon}}` and :math:`\sigma_{\mathrm{prior}}`
       (Eq. 13) via :func:`fit_listener_partial`.

    Parameters
    ----------
    sofa_path : str
        Path to the participant's SOFA file.
    obs_tbl : :class:`pandas.DataFrame`
        Behavioural observations.  Must contain columns
        ``'azi_response'``, ``'ele_response'``, ``'azi_target'``,
        ``'ele_target'``.  Must additionally contain ``'participant'``
        when ``subject_id`` is given.
    targets_coords : :class:`pyfar.Coordinates`
        Target directions presented in the experiment.
    interpolation_method : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'}
        Forwarded to
        :meth:`~bayesian_listener.BayesianListener.compute_template`.
    subject_id : str or None, default=None
        Participant identifier.  ``None`` uses every row of ``obs_tbl``.
    num_repetitions : int, default=200
        Monte Carlo repetitions for the stage-2 likelihood (Eq. 13).
    num_repetitions_motor : int, default=200
        Monte Carlo repetitions for the stage-1 motor-noise estimation
        (Eq. 12).
    num_grid_points : int, default=1
        Initialisation grid size per parameter dimension before BADS.
    fix_sigma_ild : bool, default=True
        If ``True``, fix :math:`\sigma_{\mathrm{ild}} = 1.0` dB; if
        ``False``, fit it alongside the other free parameters.
    motor_estimation_seed : int, default=42
        Seed forwarded to :func:`estimate_motor_noise`.
    verbose : bool, default=True
        Print stage-by-stage progress.

    Returns
    -------
    dict
        Result mapping with keys ``sigma_itd``, ``sigma_ild``,
        ``sigma_spectral``, ``kappa_motor``, ``sigma_motor``,
        ``sigma_prior``, ``nll``, ``n_trials``, ``time_*`` (timing
        breakdown), ``success`` (bool) and on failure ``error`` (str).

    See Also
    --------
    fit_listener_partial : Fit an arbitrary subset of parameters.
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
        model.compute_template(interpolation=interpolation_method)

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
            seed=motor_estimation_seed,
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
                'sigma_ild': 1.0,
            }
        else:
            params_to_fit = ['sigma_ild', 'sigma_spectral', 'sigma_prior']
            fixed_params = {
                'kappa_motor': kappa_motor_est,
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
            verbose=False,  # We handle verbose output here
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
            'error': str(e),
        }


def fit_listener_partial(sofa_path, obs_tbl, targets_coords,
                         interpolation_method, params_to_fit,
                         fixed_params=None, subject_id=None,
                         num_repetitions=200, num_grid_points=1,
                         verbose=True):
    r"""Fit an arbitrary subset of model parameters via BADS.

    Parameters
    ----------
    sofa_path : str
        Path to the participant's SOFA file.
    obs_tbl : :class:`pandas.DataFrame`
        Behavioural observations with columns ``'azi_response'``,
        ``'ele_response'``, ``'azi_target'``, ``'ele_target'`` (and
        ``'participant'`` if ``subject_id`` is given).
    targets_coords : :class:`pyfar.Coordinates`
        Target directions presented in the experiment.
    interpolation_method : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'}
        Forwarded to
        :meth:`~bayesian_listener.BayesianListener.compute_template`.
    params_to_fit : list of str
        Subset of free parameters to fit.  Valid entries:

        - ``'sigma_ild'`` — ILD noise (dB).
        - ``'sigma_spectral'`` — monaural spectral noise (dB).
        - ``'kappa_motor'`` — vMF motor concentration.
        - ``'sigma_prior'`` — fitted internally as the precision
          :math:`\tau_{\mathrm{prior}} = 1/\sigma_{\mathrm{prior}}^2`
          (better optimisation landscape) but returned as
          :math:`\sigma_{\mathrm{prior}}` in degrees.
    fixed_params : dict or None, default=None
        Mapping ``{parameter_name: value}`` for parameters held fixed.
        Anything not listed in ``params_to_fit`` and not present here
        falls back to the module-level ``DEFAULT_PARAMS``.
    subject_id : str or None, default=None
        Participant identifier; ``None`` uses every row of ``obs_tbl``.
    num_repetitions : int, default=200
        Monte Carlo repetitions per likelihood evaluation.
    num_grid_points : int, default=1
        Initialisation grid size per parameter dimension.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    dict
        Result mapping with keys ``sigma_itd``, ``sigma_ild``,
        ``sigma_spectral``, ``kappa_motor``, ``sigma_motor``,
        ``sigma_prior``, ``nll``, ``nll_initial``, ``time_grid``,
        ``time_bads``, ``time_total``, ``n_trials``, ``params_fitted``,
        ``success``.

    Raises
    ------
    ValueError
        If any entry of ``params_to_fit`` is not one of the four valid
        names listed above.
    """
    valid_params = ['sigma_ild', 'sigma_spectral', 'kappa_motor', 'sigma_prior']
    for p in params_to_fit:
        if p not in valid_params:
            raise ValueError(f"Invalid parameter '{p}'. Valid options: {valid_params}")

    if fixed_params is None:
        fixed_params = {}

    # Build full parameter dict with defaults, then override with fixed values
    all_params = dict(DEFAULT_PARAMS)
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
        model.compute_template(interpolation=interpolation_method)

        # Extract features as AuditoryRepresentation subset
        target_indices = model.coords.find_nearest(targets_coords)[0][0]
        targets = model.target[target_indices]

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

            # Build full sigmas_log array for negloglik function
            # Note: negloglik expects tau_prior in position 3
            sigmas_log = np.array([
                np.log(sigma_ild),
                np.log(sigma_spectral),
                np.log(kappa_motor),
                np.log(tau_prior),
            ])

            return negloglik(model, targets, resp_coords, resp_targets_idx, sigmas_log,
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

        grid_points = (
            allcomb(*grid_arrays) if len(grid_arrays) > 1
            else grid_arrays[0].reshape(-1, 1)
        )

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
            print("  Fitted parameters:")
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
            'success': True,
        }

    except Exception as e:
        print(f"ERROR fitting {label} with {interpolation_method}: {e}")
        return {
            'subject': label,
            'method': interpolation_method,
            'params_fitted': params_to_fit,
            'success': False,
            'error': str(e),
        }

