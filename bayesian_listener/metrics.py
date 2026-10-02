"""Localisation-error metrics for evaluating sound-localisation responses.

Provides :func:`localization_error` as the unified entry point and a registry
of standard metrics in the interaural-polar coordinate system following
:footcite:t:`middlebrooks1999`: lateral RMS error (``sdL``, ``rmsL``), local polar RMS
error (``rmsPmedianlocal``), quadrant-error rate (``querrMiddlebrooks``),
lateral and polar bias (``accL_cutoff``, ``accP_cutoff``), and a great-circle
angular error (``angular_error``).  The polar gain (``gainP``) follows the
selective iterative regression procedure described by :footcite:t:`macpherson2003`.
New metrics can be added with the :func:`register_metric` decorator.
"""
import numpy as np
import pyfar as pf
import inspect
import warnings
import functools


def localization_error(targets, estimations, metric,
                       auxiliary_output=False, degrees=False, **kwargs):
    """
    Compute the localization error between two sets of coordinates
    using the specified metric.

    Parameters
    ----------
    targets : pyfar.Coordinates
        The target (reference) coordinates, shape ``(n_targets,)``.
    estimations : pyfar.Coordinates
        The estimated coordinates to compare against.  May have an extra
        repetitions dimension, i.e. shape ``(n_targets, repetitions)``;
        ``targets`` is broadcast accordingly and both are flattened to
        ``(n_targets * repetitions,)`` before the metric is applied.
    metric : str or :py:obj:`~typing.Callable`
        The metric to use for error computation.

        - If a string, it must be a registered metric name.  Use
          :func:`describe_metrics` to list registered names and
          :func:`describe_metrics` ``(name)`` for details on a specific one.
        - If a callable, it must accept two :class:`pyfar.Coordinates`
          arguments ``(targets, estimations)`` as the first two positional
          arguments, plus any keyword arguments forwarded via ``**kwargs``.
          The user is responsible for coordinate convention and units.
          The callable must return either a single float or a tuple
          ``(error_value, auxiliary_data)``.
    auxiliary_output : bool, default=False
        Ignored when ``metric`` is a callable (the callable handles its own
        return shape).  When ``True`` and ``metric`` is a registered string,
        returns the auxiliary output dict alongside the error value.
    degrees : bool, default=False
        When ``True``, convert the returned error value to degrees.  Has no
        effect for metrics whose ``output_unit`` is already ``'degrees'`` or
        ``'percentage'``.
    **kwargs : dict, optional
        Forwarded to the metric function.

        - For registered metrics, kwargs are validated against the function
          signature.  Unknown kwargs raise a :class:`UserWarning` and are
          dropped; valid ones are forwarded.  See :func:`describe_metrics`
          ``(name)`` for the per-metric kwarg list.
        - For callables, kwargs are forwarded as-is with no validation.

    Returns
    -------
    float or tuple :
        The computed localization error.
        If `auxiliary_output` is True, the output will be a tuple:
        (error_value, auxiliary_data_dict).
        If the metric function does not provide auxiliary data,
        auxiliary_data_dict will be an empty dictionary.

    Examples
    --------
    Registered metric with extra kwarg:

    >>> error = localization_error(targets, estimations,
    ...                            'accL_cutoff',
    ...                            cutoff=np.deg2rad(30))      # doctest: +SKIP

    Registered metric with auxiliary output:

    >>> error, aux = localization_error(targets, estimations,
    ...                                 'querrMiddlebrooks',
    ...                                 auxiliary_output=True)  # doctest: +SKIP
    >>> print(error)                                            # doctest: +SKIP
    9.375
    >>> print(aux)                                              # doctest: +SKIP
    {'confusion_count': 48, 'response_count': 512}

    Custom callable with extra kwarg:

    >>> def my_metric(targets, estimations, threshold=0.5):     # doctest: +SKIP
    ...     ...
    >>> error = localization_error(targets, estimations,
    ...                            my_metric,
    ...                            threshold=0.1)               # doctest: +SKIP
    """
    if isinstance(metric, list):
        return {m: localization_error(targets, estimations, m,
                                      auxiliary_output=auxiliary_output,
                                      degrees=degrees,
                                      **kwargs)
                for m in metric}

    # Accept only Coordinates instances
    if not isinstance(targets, pf.Coordinates) or \
       not isinstance(estimations, pf.Coordinates):
        raise TypeError(
            "Both targets and estimations must be " \
            "pyfar.Coordinates instances.")

    if estimations.cshape[:len(targets.cshape)] != targets.cshape:
        raise ValueError(
            f"Shape mismatch: targets {targets.cshape} is not a prefix of "
            f"estimations {estimations.cshape}")

    # Case 1: metric is a custom function
    if callable(metric):
        return metric(targets, estimations, **kwargs)

    # Case 2: metric is a string, but not registered in METRIC_FUNCTIONS
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric: {metric}. Available metrics are: "
            f"{list(METRIC_FUNCTIONS.keys())}")

    # Case 3: metric is a string and registered in METRIC_FUNCTIONS
    if kwargs: # Validate extra kwargs against the function's signature
        sig = inspect.signature(METRIC_FUNCTIONS[metric])
        # Skip the first two positional params (true, est)
        extra_params = set(list(sig.parameters.keys())[2:])
        invalid = set(kwargs.keys()) - extra_params
        if invalid:
            warnings.warn(
                f"localization_error: unknown kwargs {invalid} "
                f"for metric '{metric}' will be ignored. "
                f"Valid extra parameters are: {extra_params or 'none'}.",
                UserWarning,
                stacklevel=2,
            )
            kwargs = {k: v for k, v in kwargs.items() if k in extra_params}

    expected_coord_convention = \
        get_metric_metadata(metric)['coord_convention']
    expected_unit = get_metric_metadata(metric)['input_unit']

    # Expected conventions and units are internally generated
    # by the registration system, there is no need to check them here.
    # The conventions are in ['cartesian', 'spherical', 'horizontal-polar']
    # The units are in ['radians', 'degrees', 'meters']
    # For the same reason, we assume units are coherent within the conventions.

    # Convert coordinates to the expected convention
    if expected_coord_convention == 'cartesian':
        converted_tar = targets.cartesian
        converted_est = estimations.cartesian
    elif expected_coord_convention == 'spherical':
        converted_tar = targets.spherical_elevation
        converted_est = estimations.spherical_elevation
    else:  # expected_coord_convention == 'horizontal-polar'
        converted_tar = targets.spherical_side
        converted_est = estimations.spherical_side

    # Convert units if necessary
    # Coordinates class uses radians and meters internally,
    # so we only need a conversion if expected_unit is 'degrees'
    if expected_unit == 'degrees':
        # Only convert the angular components (rad, rad, m) → (deg, deg, m)
        converted_tar[..., :2] = np.rad2deg(converted_tar[..., :2])
        converted_est[..., :2] = np.rad2deg(converted_est[..., :2])

    if converted_tar.shape != converted_est.shape:
        extra = converted_est.shape[len(targets.cshape):-1]
        converted_tar = np.broadcast_to(
            converted_tar.reshape(*targets.cshape, *([1] * len(extra)), 3),
            converted_est.shape,
        )

    # Flatten (n_targets, repetitions, 3) → (n_targets * repetitions, 3)
    if converted_est.ndim > 2:
        converted_tar = converted_tar.reshape(-1, 3)
        converted_est = converted_est.reshape(-1, 3)

    value, aux_out = \
        METRIC_FUNCTIONS[metric](converted_tar, converted_est, **kwargs)

    if degrees and get_metric_metadata(metric)['output_unit'] == 'radians':
        value = np.rad2deg(value)

    return (value, aux_out) if auxiliary_output else value


# -----------------------------------------------------------------------------
# Metric Registration System

# Shared dictionary to hold metric functions and their metadata
METRIC_FUNCTIONS = {}

def register_metric(name,
                    coord_convention,
                    input_unit,
                    output_unit=None,
                    description=None,
                    kwargs_description=None,
                    **extra_metadata,
                    ):
    """
    Decorator to register a metric function with metadata.

    Parameters
    ----------
    name : str
        Name of the metric.
    coord_convention : str
        Coordinate convention used (e.g., 'horizontal-polar').
    input_unit : str
        Unit of the input data (e.g., 'radians').
    output_unit : str, optional
        Unit of the output data (e.g., 'radians', 'percentage').
    description : str, optional
        Description of the metric.
    kwargs_description : dict, optional
        Dictionary describing extra keyword arguments expected by
        the metric function. Keys are argument names, values are descriptions.
    **extra_metadata : dict
        Additional metadata to store.

    Returns
    -------
    decorator : callable
        Decorator that wraps the target function and registers it under
        ``name`` in :data:`METRIC_FUNCTIONS`.
    """
    def decorator(func):
        """
        Decorator that registers the metric function with metadata.
        """
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            """
            Wrapper to ensure uniform output format.
            """
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                value, auxiliary_output = result
            else:
                value = result
                auxiliary_output = {}
            # Every function is uniformly formatted to return a tuple
            return value, auxiliary_output
        wrapped._metadata = {
            'name': name,
            'coord_convention': coord_convention,
            'input_unit': input_unit,
            'output_unit': output_unit,
            'description': description,
            'kwargs_description': kwargs_description,
            **extra_metadata,
        }
        METRIC_FUNCTIONS[name] = wrapped
        return wrapped
    return decorator


def get_metric_metadata(name):
    """
    Retrieve metadata for a registered metric.

    Parameters
    ----------
    name : str
        Name of the metric.

    Returns
    -------
    metadata : dict
        Metadata dictionary for the metric.
    """
    func = METRIC_FUNCTIONS.get(name)
    if func is None:
        raise ValueError(f"Metric '{name}' not found.")
    # Return a copy to prevent external modification
    return func._metadata.copy()


def describe_metrics(name=None):
    """
    Print descriptions of registered metrics.

    Parameters
    ----------
    name : str, optional
        Name of the metric to describe. If None, lists all metrics.
    """
    if name:
        info = get_metric_metadata(name)
        print(f"Metric: {name}")
        for key, value in info.items():
            if key.startswith('_'): # Skip eventual private attributes
                continue
            if key == 'kwargs_description':
                if value:
                    print("  extra kwargs:")
                    for kwarg_name, kwarg_desc in value.items():
                        print(f"\t{kwarg_name}: {kwarg_desc}")
                else:
                    print("  extra kwargs: none")
            else:
                print(f"  {key}: {value}")
    else:
        print("Available metrics:")
        for name in METRIC_FUNCTIONS.keys():
            print(f"  {name}: {get_metric_metadata(name)['description']}")
        print(
            "Use describe_metrics(name) to get details for a specific metric.")


def wrap_to_pi(rad):
    r"""Wrap angles to :math:`[-\pi, \pi)`.

    Parameters
    ----------
    rad : float or :class:`numpy.ndarray`
        Angle(s) in radians.

    Returns
    -------
    float or :class:`numpy.ndarray`
        Wrapped angle(s), same shape as ``rad``, in radians.
    """
    return (rad + np.pi) % (2 * np.pi) - np.pi


def wrap_polar_angle(angle_rad):
    r"""Wrap polar (vertical) angles to :math:`[-\pi/2, 3\pi/2)`.

    The interaural-polar convention places the front pole at ``0`` and the
    rear pole at ``π``; wrapping to ``[-π/2, 3π/2)`` keeps the upper
    hemisphere contiguous and simplifies front/back error computations.

    Parameters
    ----------
    angle_rad : float or :class:`numpy.ndarray`
        Polar angle(s) in radians.

    Returns
    -------
    float or :class:`numpy.ndarray`
        Wrapped angle(s) in radians, same shape as ``angle_rad``.
    """
    return (angle_rad + np.pi / 2) % (2 * np.pi) - np.pi / 2


# -----------------------------------------------------------------------------
# Metric Functions
@register_metric(
    name="sdL",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "Lateral RMS error (in radians).\n\t"
        "RMS of the difference between response and target lateral angles\n\t"
        "within ±60° lateral.\n\t"
        "See rms lateral error in Middlebrooks (1999)"),
    ylabel="Lateral RMS error (rad)",
)
def sdL(true, est):
    r"""Lateral standard-deviation error within :math:`\pm 80^\circ` lateral.

    Returns the standard deviation (square root of variance) of the
    response–target lateral-angle difference, restricted to estimations whose
    lateral angle satisfies :math:`|\hat{\alpha}| \le 80^\circ`.  See
    :footcite:t:`middlebrooks1999` for the foundational definition.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions in horizontal-polar convention with lateral angles
        in radians, shape ``(..., 3)``.
    est : :class:`numpy.ndarray`
        Estimated directions, same shape and convention as ``true``.

    Returns
    -------
    float
        Lateral SD in radians, or ``np.nan`` if no estimations fall within
        the ±80° band.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_true = wrap_to_pi(true[..., 0])
    lat_true = np.clip(lat_true, -np.pi/2, np.pi/2) # enforce [-π/2, π/2]

    lat_est = wrap_to_pi(est[..., 0])
    lat_est = np.clip(lat_est, -np.pi/2, np.pi/2)

    mask = np.abs(lat_est) <= np.deg2rad(80)
    if not np.any(mask):
        return np.nan

    diff = wrap_to_pi(lat_est - lat_true)[mask]
    return np.sqrt(np.var(diff))


@register_metric(
    name="rmsL",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "Lateral RMS error (in radians).\n\t"
        "RMS of the difference between response and target lateral angles\n\t"
        "within ±60° lateral.\n\t"
        "See rms lateral error in Middlebrooks (1999)"),
    ylabel="Lateral RMS error (rad)",
)
def rmsL(true, est):
    r"""Lateral RMS error within :math:`\pm 60^\circ` lateral (:footcite:t:`middlebrooks1999`).

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with lateral angle in radians.
    est : :class:`numpy.ndarray`
        Estimated directions, same convention as ``true``.

    Returns
    -------
    float
        Lateral RMS in radians, or ``np.nan`` if no estimations fall within
        the ±60° band.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_true = wrap_to_pi(true[..., 0])
    lat_true = np.clip(lat_true, -np.pi/2, np.pi/2) # enforce [-π/2, π/2]

    lat_est = wrap_to_pi(est[..., 0])
    lat_est = np.clip(lat_est, -np.pi/2, np.pi/2)

    mask = np.abs(lat_est) <= np.deg2rad(60)
    if not np.any(mask):
        return np.nan

    diff = wrap_to_pi(lat_est - lat_true)[mask]
    return np.sqrt(np.mean(diff ** 2))


@register_metric(
    name="accL_cutoff",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "Lateral bias (mean signed error) within ±cutoff° lateral.\n\t"
        "Mean of the signed difference between response and\n\t"
        "target lateral angles within ±cutoff° lateral.\n\t"
        "Cutoff defaults to 180° (π radians)."
    ),
    kwargs_description={
        'cutoff': (
            "Lateral angle threshold in radians (default: π = 180°).\n\t\t"
            "Only target positions with |lateral| ≤ cutoff are included."
        ),
    },
    ylabel="Lateral bias (rad)",
)
def accL_cutoff(true, est, cutoff=np.pi):
    r"""Lateral bias (mean signed error) within :math:`\pm` ``cutoff``.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with lateral angle in radians.
    est : :class:`numpy.ndarray`
        Estimated directions.
    cutoff : float, default=π
        Lateral-angle threshold in radians; only targets with
        :math:`|\alpha| \le` ``cutoff`` are included.

    Returns
    -------
    float
        Mean signed lateral error in radians (positive: rightward bias),
        or ``np.nan`` if no targets fall within the band.
    """
    lat_true = wrap_to_pi(true[..., 0])
    lat_est = wrap_to_pi(est[..., 0])
    mask = np.abs(lat_true) <= cutoff
    if not np.any(mask):
        return np.nan
    diff = wrap_to_pi(lat_est - lat_true)[mask]
    return np.mean(diff)


@register_metric(
    name="accP_cutoff",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "Elevation bias (mean signed error) within ±cutoff° lateral.\n\t"
        "Mean of the signed difference between response and\n\t"
        "target polar angles within ±cutoff° lateral.\n\t"
        "Cutoff defaults to 30° (π/6 radians).\n\t"
        "Positive values indicate upward bias,\n\t"
        "negative values indicate downward bias."
    ),
    kwargs_description={
        'cutoff': (
            "Lateral angle threshold in radians (default: π/6 = 30°).\n\t\t"
            "Only estimations with |lateral| ≤ cutoff are included."
        ),
    },
    ylabel="Elevation bias (rad)",
)
def accP_cutoff(true, est, cutoff=np.deg2rad(30)):
    r"""Polar bias (mean signed error) within :math:`\pm` ``cutoff``
      lateral (:footcite:t:`middlebrooks1999`).

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with angles in radians.
    est : :class:`numpy.ndarray`
        Estimated directions.
    cutoff : float, default=π/6
        Lateral-angle threshold in radians; only estimations with
        :math:`|\hat{\alpha}| \le` ``cutoff`` are included.

    Returns
    -------
    float
        Mean signed polar error in radians (positive: upward bias), or
        ``np.nan`` if no estimations fall within the band.
    """
    lat_est = wrap_to_pi(est[..., 0])
    mask = np.abs(lat_est) <= cutoff
    if not np.any(mask):
        return np.nan

    pol_true = wrap_polar_angle(true[..., 1])
    pol_est = wrap_polar_angle(est[..., 1])

    diff = wrap_to_pi(pol_est - pol_true)[mask]
    return np.mean(diff)


@register_metric(
    name="rmsPmedianlocal",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="radians",
    description=(
        "RMS polar error (local, central responses only).\n\t"
        "Root mean square of polar angle error,\n\t"
        "restricted to responses with:\n\t"
        "- lateral response within ±30° (±π/6 radians)\n\t"
        "- polar error less than 90° (π/2 radians).\n\t"
        "Based on definition in Middlebrooks (1999)."
    ),
    ylabel="Local central RMS polar error (rad)",
)
def rmsPmedianlocal(true, est):
    r"""Local RMS polar error within :math:`\pm 30^\circ` lateral, excluding quadrant errors.

    Restricted to estimations with lateral angle :math:`|\hat{\alpha}| \le 30^\circ`
    and polar error :math:`|\Delta \beta| < 90^\circ`.  Definition follows
    :footcite:t:`middlebrooks1999`.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with angles in radians.
    est : :class:`numpy.ndarray`
        Estimated directions.

    Returns
    -------
    float
        Local polar RMS in radians.

    Raises
    ------
    ValueError
        If estimated lateral angles fall outside :math:`[-\pi/2, \pi/2]`,
        if no estimations land in the central band, or if every central
        estimation has a polar error :math:`\ge 90^\circ`.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    if not np.all(np.abs(lat_est) <= np.pi / 2):
        raise ValueError("Lateral angles must be in [-π/2, π/2].")

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Select central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    if not np.any(central_mask):
        raise ValueError(
            "No central responses found within ±30° lateral range.")

    # 2. Exclude responses with polar error greater than 90°
    polar_diff = wrap_to_pi(pol_est - pol_true)[central_mask]
    local_mask = np.abs(polar_diff) < np.deg2rad(90)
    if not np.any(local_mask):
        raise ValueError("No responses with polar error < 90° found.")

    local_polar_diff = polar_diff[local_mask]
    return np.sqrt(np.mean(local_polar_diff ** 2))


@register_metric(
    name="querrMiddlebrooks",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="percentage",
    description=(
        "Quadrant error rate as defined in Middlebrooks (1999).\n\t"
        "Fraction of responses with polar error ≥ 90° (π/2 rad),\n\t"
        "restricted to responses with lateral angle in ±30° (±π/6 rad)."
    ),
    ylabel="Quadrant errors (%)",
    auxiliary_output={
        'confusion_count': 'Number of confusions (polar error ≥ 90°)',
        'response_count': \
            'Number of responses within the lateral range (|lat| ≤ 30°)',
    },
)
def querrMiddlebrooks(true, est):
    r"""Quadrant-error rate within :math:`\pm 30^\circ` lateral (:footcite:t:`middlebrooks1999`).

    Counts the fraction of central-band estimations whose polar error
    satisfies :math:`|\Delta \beta| \ge 90^\circ`.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with angles in radians.
    est : :class:`numpy.ndarray`
        Estimated directions.

    Returns
    -------
    qerr : float
        Quadrant-error rate as a percentage.
    aux : dict
        Mapping with keys:

        - ``'confusion_count'`` (int) — number of estimations with
          :math:`|\Delta \beta| \ge 90^\circ`.
        - ``'response_count'`` (int) — total estimations within the
          central ±30° lateral band.

    Raises
    ------
    ValueError
        If estimated lateral angles fall outside :math:`[-\pi/2, \pi/2]`,
        or if no estimations land in the central ±30° band.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    if not np.all(np.abs(lat_est) <= np.pi / 2):
        raise ValueError("Lateral angles must be in [-π/2, π/2].")

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Filter central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    if not np.any(central_mask):
        raise ValueError(
            "No central responses found within ±30° lateral range.")

    # 2. Compute polar error and count confusions (polar error ≥ 90°)
    polar_error = np.abs(wrap_to_pi(pol_est - pol_true))[central_mask]
    n_confusions = np.sum(polar_error >= np.deg2rad(90))
    n_total = np.int64(len(polar_error))

    qerr = 100 * n_confusions / n_total
    return qerr, {'confusion_count': n_confusions, 'response_count': n_total}


def _sirp(x, y, init, delta, n_min, maxiter):
    r"""Selective iterative regression procedure (SIRP) for one hemifield.

    Implements the procedure as described by :footcite:t:`macpherson2003`.
    A line is fitted to the initial inliers
    (the responses in the correct hemifield); every point of the pool lying
    closer to that line than ``delta`` becomes the next inlier set, and the
    line is refitted.  Points dropped in one iteration can be reselected in a
    later one.  The procedure stops when the inlier set no longer changes,
    and the result is the fit on that converged set.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Target polar angles of the hemifield pool in radians, shape ``(n,)``.
    y : :class:`numpy.ndarray`
        Response polar angles in radians, same shape as ``x``.
    init : :class:`numpy.ndarray`
        Boolean mask of the initial inliers, same shape as ``x``.
    delta : float
        Criterion distance in radians; a point is an inlier when its
        residual, wrapped to :math:`[-\pi, \pi)`, satisfies
        :math:`|y - \hat{y}| <` ``delta``.
    n_min : int
        Minimum number of inliers required at every iteration.
    maxiter : int
        Maximum number of regressions before the procedure is declared
        non-convergent.

    Returns
    -------
    dict
        Mapping with keys:

        - ``'gain'`` (float) — slope of the regression on the converged
          inlier set, ``np.nan`` if the procedure failed.
        - ``'bias'`` (float) — intercept in radians, ``np.nan`` on failure.
        - ``'n'`` (int) — size of the converged inlier set, or of the last
          set considered when the procedure stopped early.
        - ``'converged'`` (bool) — whether the inlier set converged.

        The procedure fails when an inlier set has fewer than ``n_min``
        points or no spread in target polar angle, when the inlier sets
        cycle, or when ``maxiter`` is reached; the latter two emit a
        :class:`UserWarning`.
    """
    def result(inliers, coef=(np.nan, np.nan), converged=False):
        return {'gain': coef[0], 'bias': coef[1],
                'n': int(np.sum(inliers)), 'converged': converged}

    def degenerate(inliers):
        # A line needs at least two distinct target angles
        return np.sum(inliers) < max(n_min, 2) or np.ptp(x[inliers]) == 0

    inliers = np.asarray(init, dtype=bool)
    if degenerate(inliers):
        return result(inliers)

    seen = {inliers.tobytes()}
    for _ in range(maxiter):
        coef = np.polyfit(x[inliers], y[inliers], 1)
        residual = wrap_to_pi(y - np.polyval(coef, x))
        new_inliers = np.abs(residual) < delta
        if np.array_equal(new_inliers, inliers):
            return result(inliers, coef, converged=True)
        if degenerate(new_inliers):
            return result(new_inliers)
        if new_inliers.tobytes() in seen:
            warnings.warn(
                "gainP: the selective iterative regression entered a cycle "
                "of inlier sets and did not converge; returning NaN for this "
                "hemifield.", UserWarning, stacklevel=5)
            return result(new_inliers)
        seen.add(new_inliers.tobytes())
        inliers = new_inliers

    warnings.warn(
        f"gainP: the selective iterative regression did not converge within "
        f"maxiter={maxiter} iterations; returning NaN for this hemifield.",
        UserWarning, stacklevel=5)
    return result(inliers)


@register_metric(
    name="gainP",
    coord_convention="horizontal-polar",
    input_unit="radians",
    output_unit="unitless",
    description=(
        "Polar gain (unitless), Macpherson and Middlebrooks (2003).\n\t"
        "Mean of the front- and rear-hemifield slopes of response vs.\n\t"
        "target polar angle, fitted with the selective iterative\n\t"
        "regression procedure (SIRP) to targets within ±cutoff lateral\n\t"
        "(default 30°).  NaN if either hemifield fails."
    ),
    kwargs_description={
        'cutoff': (
            "Lateral angle threshold in radians (default: π/6 = 30°).\n\t\t"
            "Only targets with |lateral| ≤ cutoff are included."
        ),
        'delta': (
            "SIRP criterion distance in radians (default: 2π/9 = 40°).\n\t\t"
            "Points closer than delta to the regression line are inliers."
        ),
        'n_min': (
            "Minimum number of inliers per hemifield (default: 5).\n\t\t"
            "Fewer inliers yield NaN for that hemifield."
        ),
        'maxiter': (
            "Maximum number of SIRP iterations per hemifield (default: 100).\n\t\t"
            "Non-convergence yields NaN and a UserWarning."
        ),
    },
    ylabel="Polar gain",
    auxiliary_output={
        'gain_front': 'Front-hemifield polar gain (slope)',
        'gain_rear': 'Rear-hemifield polar gain (slope)',
        'bias_front': 'Front-hemifield intercept (rad)',
        'bias_rear': 'Rear-hemifield intercept (rad)',
        'n_front': 'Number of front-hemifield inliers',
        'n_rear': 'Number of rear-hemifield inliers',
        'converged_front': 'Whether the front-hemifield SIRP converged',
        'converged_rear': 'Whether the rear-hemifield SIRP converged',
    },
)
def gainP(true, est, cutoff=np.deg2rad(30), delta=np.deg2rad(40), n_min=5,
          maxiter=100):
    r"""Polar gain from the selective iterative regression procedure.

    The polar gain is the slope of the linear regression of response on
    target polar angle, computed separately for the front
    (:math:`-90^\circ` to :math:`+90^\circ`) and rear (:math:`+90^\circ` to
    :math:`+270^\circ`) hemifields with the selective iterative regression
    procedure (SIRP) described by :footcite:t:`macpherson2003`, and averaged
    across the two hemifields.
    A gain of 1 indicates veridical polar localisation, 0 indicates
    responses unrelated to the target polar angle.

    For each hemifield:

    1. The pool contains all responses to the targets in that hemifield
       (target polar angle :math:`\le 90^\circ` for front,
       :math:`\ge 90^\circ` for rear), polar angles wrapped to
       :math:`[-90^\circ, 270^\circ)`.
    2. The inliers are initialised with the responses in the correct
       hemifield.
    3. A line is fitted by least squares to the inliers, and the new inliers
       are all pool points whose residual, wrapped to
       :math:`[-180^\circ, 180^\circ)`, is smaller than ``delta``.
    4. Step 3 is repeated until the inlier set no longer changes; the gain
       is the slope of the fit on that converged set.

    Only targets with :math:`|\alpha| \le` ``cutoff`` enter the analysis.
    Unlike :func:`querrMiddlebrooks` and :func:`rmsPmedianlocal`, which select
    on the *response* lateral angle, this selection is on the *target*
    lateral angle.  It reflects the stimulus design of
    :footcite:t:`macpherson2003` (targets within :math:`30^\circ` of the
    median plane) rather than the regression procedure itself.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions, horizontal-polar with angles in radians.
    est : :class:`numpy.ndarray`
        Estimated directions, same convention as ``true``.
    cutoff : float, default=π/6
        Lateral-angle threshold in radians; only targets with
        :math:`|\alpha| \le` ``cutoff`` are included.
    delta : float, default=2π/9
        Criterion distance :math:`\Delta` in radians (:math:`40^\circ`).
    n_min : int, default=5
        Minimum number of inliers per hemifield, checked at initialisation
        and at every iteration.  Fewer inliers yield ``np.nan`` for that
        hemifield.
    maxiter : int, default=100
        Maximum number of regressions per hemifield.

    Returns
    -------
    gain : float
        Mean of the front and rear polar gains (unitless), or ``np.nan`` if
        either hemifield fails: fewer than ``n_min`` inliers, inliers that
        all share one target polar angle (slope undefined), or no
        convergence.  Unaffected by ``degrees=True`` in
        :func:`localization_error`.
    aux : dict
        Mapping with keys:

        - ``'gain_front'``, ``'gain_rear'`` (float) — per-hemifield gains.
        - ``'bias_front'``, ``'bias_rear'`` (float) — per-hemifield
          intercepts in radians (not converted by ``degrees=True``).
        - ``'n_front'``, ``'n_rear'`` (int) — size of the converged inlier
          set, or of the last set considered when the procedure stopped
          early.
        - ``'converged_front'``, ``'converged_rear'`` (bool) — whether the
          inlier set converged.

    Warns
    -----
    UserWarning
        If the inlier sets of a hemifield cycle or ``maxiter`` is reached;
        that hemifield is then ``np.nan``.

    Notes
    -----
    Results can differ from AMT's ``localizationerror(m, 'gainP')`` in a
    minority of cases: AMT fits the final line to the iteration with the
    most inliers, whereas this implementation, following
    :footcite:t:`macpherson2003`, uses the converged inlier set.
    """
    lat_true = wrap_to_pi(true[..., 0])
    central = np.abs(lat_true) <= cutoff
    pol_true = wrap_polar_angle(true[..., 1])[central]
    pol_est = wrap_polar_angle(est[..., 1])[central]

    aux = {}
    for side in ('front', 'rear'):
        if side == 'front':
            pool = pol_true <= np.pi / 2
            correct = pol_est[pool] <= np.pi / 2
        else:
            pool = pol_true >= np.pi / 2
            correct = pol_est[pool] >= np.pi / 2
        res = _sirp(pol_true[pool], pol_est[pool], correct,
                    delta=delta, n_min=n_min, maxiter=maxiter)
        aux[f'gain_{side}'] = res['gain']
        aux[f'bias_{side}'] = res['bias']
        aux[f'n_{side}'] = res['n']
        aux[f'converged_{side}'] = res['converged']

    gain = np.mean([aux['gain_front'], aux['gain_rear']])
    return gain, aux


@register_metric(
    name='angular_error',
    coord_convention='cartesian',
    input_unit='meters',
    output_unit='radians',
    description=(
        "Great-circle angular error (in radians).\n\t"
        "Computed as arccos of the dot product between target\n\t"
        "and estimation directions, each normalised to unit length,\n\t"
        "so the result is independent of the source distance.\n\t"
        "Returns the mean angular error across all observations."),
    ylabel="Angular error (rad)",
)
def angular_error(true, est):
    r"""Mean great-circle angular error between target and estimation directions.

    Computes :math:`\bar{\theta} = \frac{1}{N} \sum \arccos(
    \hat{\mathbf{t}}_i \cdot \hat{\mathbf{e}}_i)` where
    :math:`\hat{\mathbf{x}} = \mathbf{x} / \lVert \mathbf{x} \rVert`, with the
    dot product clipped to :math:`[-1, 1]` for numerical safety.

    Both inputs are normalised to unit length, so the result depends only on
    direction and is invariant to the source distance.  This matters because
    HRTF datasets are measured at different radii (e.g. 1.5 m vs 3.0 m); using
    the raw dot product would scale it by :math:`r_{\text{true}} r_{\text{est}}`
    and saturate the ``arccos`` clip for all but the largest angular errors.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions in Cartesian coordinates, shape ``(..., 3)``;
        any non-zero norm is accepted.
    est : :class:`numpy.ndarray`
        Estimated directions, same shape and convention.

    Returns
    -------
    float
        Mean angular error in radians.

    Raises
    ------
    ValueError
        If any target or estimation vector has zero length, in which case its
        direction is undefined.
    """
    true_norm = np.linalg.norm(true, axis=-1, keepdims=True)
    est_norm = np.linalg.norm(est, axis=-1, keepdims=True)
    if np.any(true_norm == 0) or np.any(est_norm == 0):
        raise ValueError(
            "angular_error: zero-length direction vector; the direction of "
            "the origin is undefined.")

    # Dot product row-wise on unit vectors, clipped for numerical safety
    dots = np.sum((true / true_norm) * (est / est_norm), axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return np.mean(angles)
