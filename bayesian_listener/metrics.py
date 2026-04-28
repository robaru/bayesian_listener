"""Localisation-error metrics for evaluating sound-localisation responses.

Provides :func:`localization_error` as the unified entry point and a registry
of standard metrics in the interaural-polar coordinate system following
[middlebrooks1999]_: lateral RMS error (``sdL``, ``rmsL``), local polar RMS
error (``rmsPmedianlocal``), quadrant-error rate (``querrMiddlebrooks``),
lateral and polar bias (``accL_cutoff``, ``accP_cutoff``), and a great-circle
angular error (``angular_error``).  New metrics can be added with the
:func:`register_metric` decorator.
"""
import numpy as np
import pyfar as pf
import inspect
import warnings
import functools


def localization_error(targets, estimations, metric,
                       auxiliary_output=False, **kwargs):
    """
    Compute the localization error between two sets of coordinates
    using the specified metric.

    Parameters
    ----------
    targets : pyfar.Coordinates
        The target (reference) coordinates.
    estimations : pyfar.Coordinates
        The estimated coordinates to compare against.
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
    # Accept only Coordinates instances
    if not isinstance(targets, pf.Coordinates) or \
       not isinstance(estimations, pf.Coordinates):
        raise TypeError(
            "Both targets and estimations must be " \
            "pyfar.Coordinates instances.")

    if targets.cshape != estimations.cshape:
        raise ValueError(
            f"Shape mismatch: {targets.cshape} vs {estimations.cshape}")

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
        converted_tar[:, :2] = np.rad2deg(converted_tar[:, :2])
        converted_est[:, :2] = np.rad2deg(converted_est[:, :2])

    value, aux_out = \
        METRIC_FUNCTIONS[metric](converted_tar, converted_est, **kwargs)

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
    [middlebrooks1999]_ for the foundational definition.

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
    r"""Lateral RMS error within :math:`\pm 60^\circ` lateral ([middlebrooks1999]_).

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
    r"""Polar bias (mean signed error) within :math:`\pm` ``cutoff`` lateral ([middlebrooks1999]_).

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
    [middlebrooks1999]_.

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
    r"""Quadrant-error rate within :math:`\pm 30^\circ` lateral ([middlebrooks1999]_).

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


@register_metric(
    name='angular_error',
    coord_convention='cartesian',
    input_unit='meters',
    output_unit='radians',
    description=(
        "Great-circle angular error (in radians).\n\t"
        "Computed as arccos of the dot product between target\n\t"
        "and estimation unit vectors.\n\t"
        "Returns the mean angular error across all observations."),
    ylabel="Angular error (rad)",
)
def angular_error(true, est):
    r"""Mean great-circle angular error between target and estimation unit vectors.

    Computes :math:`\bar{\theta} = \frac{1}{N} \sum \arccos(
    \mathbf{t}_i \cdot \hat{\mathbf{e}}_i)` with the dot product clipped
    to :math:`[-1, 1]` for numerical safety.

    Parameters
    ----------
    true : :class:`numpy.ndarray`
        Target directions in Cartesian coordinates, shape ``(..., 3)``;
        each row should be unit-norm.
    est : :class:`numpy.ndarray`
        Estimated directions, same shape and convention.

    Returns
    -------
    float
        Mean angular error in radians.
    """
    # Dot product row-wise, clipped to [-1, 1] for numerical safety
    dots = np.sum(true * est, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return np.mean(angles)
