"""
This module contains functions to compute localization errors based on a set
of target and response directions.
"""
import numpy as np

# Shared dictionary to hold metric functions and their metadata
METRIC_FUNCTIONS = {}

def register_metric(name,
                    coord_convention,
                    input_unit,
                    output_unit=None,
                    description=None,
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
    **extra_metadata : dict
        Additional metadata to store.

    Returns
    -------
    decorator : function
        Decorator that registers the metric function.
    """
    def decorator(func):
        """
        Decorator that registers the metric function with metadata.
        """
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
            print(f"  {key}: {value}")
    else:
        print("Available metrics:")
        for name in METRIC_FUNCTIONS.keys():
            print(f"  {name}: {get_metric_metadata(name)['description']}")
        print(
            "Use describe_metrics(name) to get details for a specific metric.")


def wrap_to_pi(rad):
    """Wrap angles to [-π, π)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


def wrap_polar_angle(angle_rad):
    """
    Wrap polar angles to the range [-π/2, 3π/2) ≡ [-90°, 270°).
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
    """
    Compute lateral RMS error within ±60° lateral.
    More details in the decorator above.
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
    """
    Compute lateral RMS error within ±60° lateral.
    More details in the decorator above.
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
    """
    Compute local RMS polar error within ±30° lateral and polar error < 90°.
    More details in the decorator above.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    assert np.all(np.abs(lat_est) <= np.pi/2), \
        "Lateral angles must be in [-π/2, π/2]"

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Select central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    assert np.any(central_mask), \
        "No central responses found within ±30° lateral range."

    # 2. Exclude responses with polar error greater than 90°
    polar_diff = wrap_to_pi(pol_est - pol_true)[central_mask]
    local_mask = np.abs(polar_diff) < np.deg2rad(90)
    assert np.any(local_mask), "No responses with polar error < 90° found."

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
    """
    Compute quadrant error rate as defined in Middlebrooks (1999).
    More details in the decorator above.
    """
    # lateral in [-π, π), then restrict to [-π/2, π/2]
    lat_est = wrap_to_pi(est[..., 0])
    assert np.all(np.abs(lat_est) <= np.pi/2), \
        "Lateral angles must be in [-π/2, π/2]"

    pol_true = wrap_polar_angle(true[..., 1])  # polar in [-π/2, 3π/2)
    pol_est = wrap_polar_angle(est[..., 1])

    # 1. Filter central responses: lateral response within ±30°
    central_mask = np.abs(lat_est) <= np.deg2rad(30)
    assert np.any(central_mask), \
        "No central responses found within ±30° lateral range."

    # 2. Compute polar error and count confusions (polar error ≥ 90°)
    polar_error = np.abs(wrap_to_pi(pol_est - pol_true))[central_mask]
    n_confusions = np.sum(polar_error >= np.deg2rad(90))
    n_total = np.int64(len(polar_error))

    qerr = 100 * n_confusions / n_total
    return qerr, {'confusion_count': n_confusions, 'response_count': n_total}
