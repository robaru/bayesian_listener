import pytest


def test_import_bayesian_listener():
    try:
        import bayesian_listener           # noqa
    except ImportError:
        pytest.fail('import bayesian_listener failed')

import pytest
import numpy as np
from pathlib import Path
import urllib.request
from bayesian_listener.bayesian_listener import BayesianListener
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
    # Set random seed for reproducibility (for both infer and estimate)
    np.random.seed(42)

    # Load SOFA file
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)

    # Prepare features
    am.prepare_features()

    # Pick one target
    targets = am.represent()
    target = targets[260, :]

    # Estimate position with fixed seed
    posterior = am.infer(target, repetitions=1, seed=42)
    # Disable motor noise for deterministic test
    estimation = am.estimate(posterior, sigma_motor=0)

    # Get original and estimated directions in spherical coordinates
    estimated_coords = Coordinates(
        sofa_file=None,
        positions=estimation[:, 0, :],
        convention='cartesian'
    )
    estimated_dir = estimated_coords.sph()

    # Verify shape
    assert estimation.shape == (1, 1, 3)
    assert np.allclose(np.linalg.norm(estimation[0, 0, :]), 1.0, atol=0.1)

    # Compare with fixed expected spherical coordinates (azimuth, elevation)
    expected_dir_sph = np.array([[126.58887 ,  -9.108036]])
    np.testing.assert_allclose(estimated_dir, expected_dir_sph, rtol=1e-2)