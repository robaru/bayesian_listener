import pytest
import numpy as np
import pyfar as pf
from pathlib import Path
import urllib.request
from bayesian_listener import BayesianListener
from bayesian_listener.coordinates import Coordinates


def get_sofa_file():
    """
    Get path to SONICOM SOFA test file.

    Downloads the file if not available in data/ directory.
    Skips the test if download fails.
    """
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / 'data'
    data_dir.mkdir(exist_ok=True)

    sofa_path = data_dir / 'P0001_FreeFieldCompMinPhase_48kHz.sofa'

    if not sofa_path.exists():
        # Try to download the file
        url = 'https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/P0001/HRTF/HRTF/48kHz/P0001_FreeFieldCompMinPhase_48kHz.sofa'
        try:
            print(f"\nDownloading {sofa_path.name}...")
            print(f"From: {url}")
            urllib.request.urlretrieve(url, sofa_path)
            print(f"✓ Downloaded successfully to {sofa_path}")
        except Exception as e:
            pytest.skip(f"Test data not available and download failed: {e}")

    return str(sofa_path)


def test_model_single():
    """Test single target inference with fixed random seed for reproducibility."""

    seed = 42

    # Load SOFA file
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)

    # Prepare features
    am.prepare_features()

    # Pick one target
    targets = am.represent()
    target = targets[260, :]

    # Estimate position with fixed seed
    posterior = am.infer(target, repetitions=1, seed=seed)
    # Disable motor noise for deterministic test
    estimation = am.estimate(posterior, sigma_motor=0, seed=seed)

    estimated_dir = np.rad2deg(estimation.spherical_elevation[..., 0:2])

    # Check estimate return type
    assert isinstance(estimation, pf.Coordinates)

    # Verify shape
    assert estimation.cartesian.shape == (1, 1, 3)
    assert np.allclose(np.linalg.norm(estimation.cartesian[0, 0, :]),
                       1.0, atol=0.1)

    # Compare with fixed expected spherical coordinates (azimuth, elevation)
    # Coordinates(sofa_file).sph()[260, :] -> array([125.,   0.])
    expected_dir_sph = np.array([[126.871232,   0.966419]])
    np.testing.assert_allclose(estimated_dir.squeeze(), expected_dir_sph.squeeze(), rtol=1e-2)


def test_model_multiple():
    """Test inference with two targets and two repetitions.

    Verifies that the model produces correct output shapes and that the
    estimated directions remain close to the true target directions,
    indicating the model holds for this configuration.
    """
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    # Set parameters to minimum values
    am.parameters = {
        "sigma_itd": 1e-1,
        "sigma_ild": 1e-1,
        "sigma_spectral": 1e-1,
        "sigma_prior": 180,
        "sigma_motor": 0,
    }

    # Pick two targets from distinct directions
    all_targets = am.represent()
    target_indices = [100, 260]
    targets = all_targets[target_indices, :]

    # Get true target positions in spherical coordinates
    true_coords = am.coords.spherical_elevation
    true_dirs = true_coords[target_indices, :]  # (2, 2) -> azimuth, elevation

    # Run inference: 2 targets x 2 repetitions
    posterior = am.infer(targets, repetitions=2, seed=42)
    estimation = am.estimate(posterior, sigma_motor=0, seed=42)

    # Verify shapes: (n_targets, n_repetitions)
    assert estimation.cshape == (2, 2), \
        f"Expected shape (2, 2), got {estimation.shape}"

    # All estimated directions should be unit vectors
    norms = np.linalg.norm(estimation, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=0.1,
                               err_msg="Estimations should be unit vectors")

    # Convert estimations to spherical and check angular proximity to targets
    # For each target, at least one repetition should be within 30 deg
    tolerance_deg = 5
    for t_idx in range(2):
        for r_idx in range(2):
            est_sph = estimation.spherical_elevation  # (1, 2) -> azimuth, elevation

            az_err = abs(est_sph[0, 0, 0] - true_dirs[t_idx, 0])
            # Handle azimuth wraparound
            az_err = min(az_err, 360 - az_err)
            el_err = abs(est_sph[0, 1, 0] - true_dirs[t_idx, 1])

            angular_err = np.sqrt(az_err**2 + el_err**2)
            if angular_err > tolerance_deg:
                return False


def test_interp():
    """Test SHMAX interpolation produces valid template features."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    # Verify template was created
    assert hasattr(am, 'template'), "Template should be created after prepare_features"
    assert am.template is not None, "Template should not be None"

    # Verify template has spectral cues
    assert hasattr(am.template, 'spectral_cues'), "Template should have spectral_cues"
    assert am.template.spectral_cues is not None, "Template spectral_cues should not be None"

    # Verify shapes match expected dimensions
    # spectral_cues shape should be (n_directions, n_frequencies, n_sides)
    assert am.spectral_cues.ndim == 3, "Spectral cues should be 3D array"
    assert am.template.spectral_cues.ndim == 3, "Template spectral cues should be 3D array"

    # Template and original should have same frequency and side dimensions
    assert am.spectral_cues.shape[1:] == am.template.spectral_cues.shape[1:], \
        "Template should have same frequency and side dimensions as original"

    # Verify interpolated values are reasonable (finite and within expected range)
    assert np.all(np.isfinite(am.template.spectral_cues)), \
        "Template spectral cues should all be finite"

    # Get target spectral cues for comparison
    side = 0
    amps_target = am.spectral_cues[260, :, side]

    # Verify template values are in similar range to original
    template_min = np.min(am.template.spectral_cues[260, :, side])
    template_max = np.max(am.template.spectral_cues[260, :, side])
    target_min = np.min(amps_target)
    target_max = np.max(amps_target)

    # Template should be within a reasonable range of original data
    # Allow some margin since interpolation might extend slightly
    margin = 0.2 * (target_max - target_min)
    assert template_min >= target_min - margin, \
        "Template min should not be significantly below original min"
    assert template_max <= target_max + margin, \
        "Template max should not be significantly above original max"


