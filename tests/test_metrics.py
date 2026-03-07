"""
Test suite for localization error metrics.

This module contains comprehensive tests for the localization_error
module, including tests for individual metrics, the main
localization_error function, and utility functions.
"""
import pytest
import numpy as np
import pyfar as pf
from bayesian_listener.metrics import (
    localization_error,
    describe_metrics,
    get_metric_metadata,
    wrap_to_pi,
    wrap_polar_angle,
    METRIC_FUNCTIONS,
)


# =============================================================================
# Test utility functions
# =============================================================================

def test_wrap_to_pi():
    """Test wrap_to_pi wraps angles to [-π, π) correctly."""
    # Test standard wrapping
    assert np.isclose(wrap_to_pi(0), 0)
    assert np.isclose(wrap_to_pi(np.pi), -np.pi)  # π wraps to -π
    assert np.isclose(wrap_to_pi(-np.pi), -np.pi)
    assert np.isclose(wrap_to_pi(2 * np.pi), 0, atol=1e-10)
    assert np.isclose(wrap_to_pi(3 * np.pi), -np.pi)
    assert np.isclose(wrap_to_pi(-3 * np.pi), -np.pi)

    # Test values strictly inside the range
    assert np.isclose(wrap_to_pi(np.pi/2), np.pi/2)
    assert np.isclose(wrap_to_pi(-np.pi/2), -np.pi/2)

    # Test with array
    angles = np.array([0, np.pi/2, 2*np.pi, 3*np.pi, -np.pi])
    wrapped = wrap_to_pi(angles)
    expected = np.array([0, np.pi/2, 0, -np.pi, -np.pi])
    np.testing.assert_allclose(wrapped, expected, atol=1e-10)


def test_wrap_polar_angle():
    """Test wrap_polar_angle wraps to [-π/2, 3π/2) correctly."""
    # Test standard wrapping
    assert np.isclose(wrap_polar_angle(0), 0)
    assert np.isclose(wrap_polar_angle(np.pi), np.pi)
    assert np.isclose(wrap_polar_angle(-np.pi/2), -np.pi/2)
    # 3π/2 is at the boundary (excluded), wraps to -π/2
    assert np.isclose(wrap_polar_angle(3*np.pi/2), -np.pi/2, atol=1e-10)
    assert np.isclose(wrap_polar_angle(2*np.pi), 0, atol=1e-10)

    # Test wrapping from other ranges
    # -π wraps to π (since -π + π/2 = -π/2, then (-π/2 + π/2) % 2π - π/2)
    # Actually: (-π + π/2) % 2π - π/2 = -π/2 % 2π - π/2 = 3π/2 - π/2 = π
    assert np.isclose(wrap_polar_angle(-np.pi), np.pi, atol=1e-10)

    # 5π/2 = 3π/2 + π, wraps to π/2
    assert np.isclose(wrap_polar_angle(5*np.pi/2), np.pi/2, atol=1e-10)

    # Test values strictly inside the range
    assert np.isclose(wrap_polar_angle(np.pi/4), np.pi/4)
    assert np.isclose(wrap_polar_angle(np.pi/2), np.pi/2)


# =============================================================================
# Test metric registration system
# =============================================================================

def test_get_metric_metadata():
    """Test retrieval of metric metadata."""
    # Test valid metric
    metadata = get_metric_metadata('rmsL')
    assert metadata['name'] == 'rmsL'
    assert metadata['coord_convention'] == 'horizontal-polar'
    assert metadata['input_unit'] == 'radians'
    assert metadata['output_unit'] == 'radians'
    assert 'description' in metadata

    # Test invalid metric
    with pytest.raises(ValueError, match="Metric .* not found"):
        get_metric_metadata('nonexistent_metric')


def test_describe_metrics(capsys):
    """Test describe_metrics prints information correctly."""
    # Test listing all metrics
    describe_metrics()
    captured = capsys.readouterr()
    assert 'Available metrics:' in captured.out
    assert 'rmsL' in captured.out
    assert 'rmsPmedianlocal' in captured.out
    assert 'querrMiddlebrooks' in captured.out

    # Test describing specific metric
    describe_metrics('rmsL')
    captured = capsys.readouterr()
    assert 'Metric: rmsL' in captured.out
    assert 'coord_convention: horizontal-polar' in captured.out


def test_all_metrics_registered():
    """Test that all expected metrics are registered."""
    expected_metrics = ['rmsL',
                        'rmsPmedianlocal',
                        'querrMiddlebrooks',
                        ]
    for metric_name in expected_metrics:
        assert metric_name in METRIC_FUNCTIONS, \
            f"Metric {metric_name} not registered"


# =============================================================================
# Test main localization_error function
# =============================================================================

def test_localization_error_invalid_inputs():
    """Test localization_error raises TypeError for invalid inputs."""
    # Create valid Coordinates object
    valid_coords = pf.Coordinates.from_cartesian(
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([1]),
    )

    # Test with non-Coordinates inputs
    with pytest.raises(TypeError, match="must be pyfar.Coordinates instances"):
        localization_error(
            np.array([[0, 0, 1]]),
            valid_coords,
            'rmsL',
        )

    with pytest.raises(TypeError, match="must be pyfar.Coordinates instances"):
        localization_error(
            valid_coords,
            np.array([[0, 0, 1]]),
            'rmsL',
        )


def test_localization_error_shape_mismatch():
    """Test localization_error raises ValueError for shape mismatch."""
    coords1 = pf.Coordinates.from_cartesian(
        x=np.array([0, 1]),
        y=np.array([0, 0]),
        z=np.array([1, 0]),
    )
    coords2 = pf.Coordinates.from_cartesian(
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([1]),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        localization_error(coords1, coords2, 'rmsL')


def test_localization_error_unknown_metric():
    """Test localization_error raises ValueError for unknown metric."""
    coords = pf.Coordinates.from_cartesian(
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([1]),
    )

    with pytest.raises(ValueError, match="Unknown metric"):
        localization_error(coords, coords, 'nonexistent_metric')


def test_localization_error_with_callable():
    """Test localization_error with custom callable metric function."""
    # Create simple test data
    targets = pf.Coordinates.from_cartesian(
        x=np.array([0, 1]),
        y=np.array([0, 0]),
        z=np.array([1, 0]),
    )
    estimations = pf.Coordinates.from_cartesian(
        x=np.array([0, 0.9]),
        y=np.array([0, 0.1]),
        z=np.array([1, 0]),
    )

    # Define simple callable: Euclidean distance
    def euclidean_distance(t, e):
        """Compute mean Euclidean distance."""
        distances = np.linalg.norm(t - e, axis=1)
        return np.mean(distances)

    result = localization_error(targets, estimations, euclidean_distance)
    assert isinstance(result, (float, np.floating))
    assert result >= 0

    # Test callable returning tuple (value, auxiliary)
    def euclidean_with_aux(t, e):
        """Compute Euclidean distance with auxiliary output."""
        distances = np.linalg.norm(t - e, axis=1)
        mean_dist = np.mean(distances)
        aux = {'max_distance': np.max(distances)}
        return mean_dist, aux

    result = localization_error(
        targets,
        estimations,
        euclidean_with_aux,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], dict)


def test_localization_error_auxiliary_output():
    """Test auxiliary_output parameter returns extra information."""
    # Create test data in horizontal-polar
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, 0]),
        polar=np.array([0, np.pi]),
        radius=np.array([1, 1]),
    )
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, 0]),
        polar=np.array([0, 0]),  # This will be a quadrant error
        radius=np.array([1, 1]),
    )

    # Test without auxiliary output
    error = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
        auxiliary_output=False,
    )
    assert isinstance(error, (float, np.floating))

    # Test with auxiliary output
    error, aux = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
        auxiliary_output=True,
    )
    assert isinstance(error, (float, np.floating))
    assert isinstance(aux, dict)
    assert 'confusion_count' in aux
    assert 'response_count' in aux
    assert aux['confusion_count'] == 1
    assert aux['response_count'] == 2


# =============================================================================
# Test individual metrics with known outputs
# =============================================================================

def test_rmsL_perfect_estimation():
    """Test rmsL returns zero for perfect lateral estimation."""
    # Create identical targets and estimations
    lateral = np.array([0, np.pi/4, -np.pi/4])  # 0°, 45° right, 45° left
    polar = np.array([0, 0, 0])  # All at polar angle 0
    radius = np.array([1, 1, 1])  # All at radius 1

    targets = pf.Coordinates.from_spherical_side(
        lateral=lateral,
        polar=polar,
        radius=radius,
    )
    estimations = pf.Coordinates.from_spherical_side(
        lateral=lateral,
        polar=polar,
        radius=radius,
    )

    error = localization_error(targets, estimations, 'rmsL')
    assert np.isclose(error, 0, atol=1e-10)


def test_rmsL_known_output():
    """Test rmsL with synthetic data producing known output."""
    # Create targets at center (lateral = 0)
    n_samples = 4
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.zeros(n_samples),
        polar=np.zeros(n_samples),
        radius=np.ones(n_samples),
    )

    # Create estimations with known lateral errors: [10°, -10°, 20°, -20°]
    # All within ±60° so all will be included
    lateral_errors_deg = np.array([10, -10, 20, -20])
    lateral_errors_rad = np.deg2rad(lateral_errors_deg)
    estimations = pf.Coordinates.from_spherical_side(
        lateral=lateral_errors_rad,
        polar=np.zeros(n_samples),
        radius=np.ones(n_samples),
    )

    # Expected RMS: sqrt(mean([10², 10², 20², 20²])) = sqrt(250) degrees
    # In radians: sqrt(mean([0.1745², 0.1745², 0.3491², 0.3491²]))
    expected_rms = np.sqrt(np.mean(lateral_errors_rad ** 2))

    error = localization_error(targets, estimations, 'rmsL')
    assert np.isclose(error, expected_rms, rtol=1e-3)


def test_rmsL_outside_60deg_excluded():
    """Test rmsL excludes responses outside ±60° lateral."""
    # Create targets at center
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, 0]),
        polar=np.array([0, 0]),
        radius=np.array([1, 1]),
    )

    # One estimation within ±60°, one outside
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([np.deg2rad(30), np.deg2rad(70)]),
        polar=np.array([0, 0]),
        radius=np.array([1, 1]),
    )

    # Only the first error should be counted
    expected_rms = np.deg2rad(30)  # RMS of [30°]

    error = localization_error(targets, estimations, 'rmsL')
    assert np.isclose(error, expected_rms, rtol=1e-3)


def test_rmsPmedianlocal_perfect_estimation():
    """Test rmsPmedianlocal returns zero for perfect estimation."""
    # Create central responses (lateral within ±30°)
    lateral = np.array([0, np.deg2rad(20)])  # 0°, 20° lateral
    polar = np.array([0, np.pi/2])  # 0°, 90° polar
    radius = np.array([1, 1])  # All at radius 1

    targets = pf.Coordinates.from_spherical_side(
        lateral=lateral,
        polar=polar,
        radius=radius,
    )

    estimations = pf.Coordinates.from_spherical_side(
        lateral=lateral,
        polar=polar,
        radius=radius,
    )

    error = localization_error(targets, estimations, 'rmsPmedianlocal')
    assert np.isclose(error, 0, atol=1e-10)


def test_rmsPmedianlocal_known_output():
    """Test rmsPmedianlocal with synthetic data."""
    # Create targets at center with polar = 0
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(15), np.deg2rad(-20)]),
        polar=np.array([0, 0, 0]),
        radius=np.array([1, 1, 1]),
    )

    # Create estimations with polar errors: [30°, 45°, 60°]
    # All lateral responses within ±30°, all polar errors < 90°
    polar_errors_deg = np.array([30, 45, 60])
    polar_errors_rad = np.deg2rad(polar_errors_deg)
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(15), np.deg2rad(-20)]),
        polar=polar_errors_rad,
        radius=np.array([1, 1, 1]),
    )

    # Expected RMS of polar errors
    expected_rms = np.sqrt(np.mean(polar_errors_rad ** 2))

    error = localization_error(targets, estimations, 'rmsPmedianlocal')
    assert np.isclose(error, expected_rms, rtol=1e-3)


def test_rmsPmedianlocal_excludes_large_polar_errors():
    """Test rmsPmedianlocal excludes polar errors >= 90°."""
    # Create targets with lateral within ±30°
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(20)]),
        polar=np.array([0, 0]),
        radius=np.array([1, 1]),
    )

    # First has small polar error, second has large (>=90°)
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(20)]),
        polar=np.array([np.deg2rad(30), np.deg2rad(100)]),
        radius=np.array([1, 1]),
    )

    # Only first error should count
    expected_rms = np.deg2rad(30)

    error = localization_error(targets, estimations, 'rmsPmedianlocal')
    assert np.isclose(error, expected_rms, rtol=1e-3)


def test_querrMiddlebrooks_zero_errors():
    """Test querrMiddlebrooks returns 0% for no quadrant errors."""
    # Create central responses with small polar errors
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(15), np.deg2rad(-20)]),
        polar=np.array([0, 0, 0]),
        radius=np.array([1, 1, 1]),
    )

    # Small polar errors (all < 90°)
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(15), np.deg2rad(-20)]),
        polar=np.array([np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)]),
        radius=np.array([1, 1, 1]),
    )

    error, aux = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
        auxiliary_output=True,
    )

    assert np.isclose(error, 0, atol=1e-10)
    assert aux['confusion_count'] == 0
    assert aux['response_count'] == 3


def test_querrMiddlebrooks_known_output():
    """Test querrMiddlebrooks with known quadrant error rate."""
    # Create 4 targets, all with lateral within ±30°
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(10), np.deg2rad(-15), np.deg2rad(25)]),
        polar=np.array([0, 0, 0, 0]),
        radius=np.array([1, 1, 1, 1]),
    )

    # 2 have small polar errors, 2 have large (>=90°)
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([
            0, np.deg2rad(10), np.deg2rad(-15), np.deg2rad(25),
        ]),
        polar=np.array([
            np.deg2rad(30), np.deg2rad(100), np.deg2rad(60), np.deg2rad(120),
        ]),
        radius=np.array([1, 1, 1, 1]),
    )

    # Expected: 2/4 = 50% quadrant errors
    expected_error = 50.0

    error, aux = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
        auxiliary_output=True,
    )

    assert np.isclose(error, expected_error, rtol=1e-3)
    assert aux['confusion_count'] == 2
    assert aux['response_count'] == 4


def test_querrMiddlebrooks_excludes_peripheral():
    """Test querrMiddlebrooks excludes lateral responses > ±30°."""
    # Create targets with varying lateral positions
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(50)]),
        polar=np.array([0, 0]),
        radius=np.array([1, 1]),
    )

    # Both have large polar errors
    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.deg2rad(50)]),
        polar=np.array([np.deg2rad(100), np.deg2rad(100)]),
        radius=np.array([1, 1]),
    )

    # Only first response should count
    # 1 confusion out of 1 response = 100%
    expected_error = 100.0

    error, aux = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
        auxiliary_output=True,
    )

    assert np.isclose(error, expected_error, rtol=1e-3)
    assert aux['confusion_count'] == 1
    assert aux['response_count'] == 1


# =============================================================================
# Test coordinate conversion in localization_error
# =============================================================================

def test_localization_error_coordinate_conversion():
    """Test that localization_error converts coordinates correctly."""
    # Create targets in Cartesian
    targets_cart = pf.Coordinates.from_cartesian(
        x=np.array([1, 0]),
        y=np.array([0, 1]),
        z=np.array([0, 0]),
    )

    # Create estimations in spherical (same positions)
    # [1, 0, 0] in Cartesian = [0°, 0°, 1] in spherical
    # [0, 1, 0] in Cartesian = [90°, 0°, 1] in spherical
    estimations_sph = pf.Coordinates.from_spherical_side(
        lateral=np.array([0, np.pi/2]),
        polar=np.array([0, 0]),
        radius=np.array([1, 1]),
    )

    # Should convert both to horizontal-polar and compute
    # Perfect match, so error should be near zero
    error = localization_error(targets_cart, estimations_sph, 'rmsL')

    # Allow small numerical errors from conversions
    assert error < 0.01


# =============================================================================
# Edge cases and special conditions
# =============================================================================

def test_rmsL_all_outside_range():
    """Test rmsL returns NaN when all responses outside ±60°."""
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0]),
        polar=np.array([0]),
        radius=np.array([1]),
    )

    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([np.deg2rad(80)]),
        polar=np.array([0]),
        radius=np.array([1]),
    )

    error = localization_error(targets, estimations, 'rmsL')
    assert np.isnan(error)


def test_single_position():
    """Test metrics work with single position."""
    targets = pf.Coordinates.from_spherical_side(
        lateral=np.array([0]),
        polar=np.array([0]),
        radius=np.array([1]),
    )

    estimations = pf.Coordinates.from_spherical_side(
        lateral=np.array([np.deg2rad(10)]),
        polar=np.array([np.deg2rad(20)]),
        radius=np.array([1]),
    )

    # Should not raise any errors
    error_rmsL = localization_error(targets, estimations, 'rmsL')
    assert isinstance(error_rmsL, (float, np.floating))

    error_rmsP = localization_error(
        targets,
        estimations,
        'rmsPmedianlocal',
    )
    assert isinstance(error_rmsP, (float, np.floating))

    error_qerr = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
    )
    assert isinstance(error_qerr, (float, np.floating))


def test_large_dataset():
    """Test metrics handle larger datasets efficiently."""
    n_samples = 1000

    # Create random positions within valid ranges
    rng = np.random.default_rng(42)
    lateral = rng.uniform(-np.pi/6, np.pi/6, n_samples)  # ±30°
    polar = rng.uniform(0, np.pi, n_samples)

    targets = pf.Coordinates.from_spherical_side(
        lateral=lateral,
        polar=polar,
        radius=np.ones(n_samples),
    )

    # Add small random errors
    lateral_est = lateral + rng.normal(0, 0.1, n_samples)
    polar_est = polar + rng.normal(0, 0.2, n_samples)

    estimations = pf.Coordinates.from_spherical_side(
        lateral=lateral_est,
        polar=polar_est,
        radius=np.ones(n_samples),
    )

    # Should complete without errors
    error_rmsL = localization_error(targets, estimations, 'rmsL')
    assert isinstance(error_rmsL, (float, np.floating))
    assert error_rmsL > 0  # Should have some error

    error_qerr = localization_error(
        targets,
        estimations,
        'querrMiddlebrooks',
    )
    assert isinstance(error_qerr, (float, np.floating))
    assert 0 <= error_qerr <= 100  # Should be a valid percentage
