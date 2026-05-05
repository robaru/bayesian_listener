import pytest
import numpy as np
import pyfar as pf
import sofar
from pathlib import Path
import urllib.request
from bayesian_listener import BayesianListener, Barumerli2023
from bayesian_listener.auditory_representation import Barumerli2023pge
from bayesian_listener.utils import save_to_cache, load_from_cache


def get_sofa_file():
    """
    Get path to SONICOM SOFA test file.

    Downloads the file if not available in data/ directory.
    Skips the test if download fails.
    """
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / 'data'
    data_dir.mkdir(exist_ok=True)

    sofa_file = 'P0001_FreeFieldCompMinPhase_48kHz.sofa'
    sofa_path = data_dir / sofa_file

    if not sofa_path.exists():
        url = ('https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/'
               'P0001/HRTF/HRTF/48kHz/' + sofa_file)
        try:
            print(f"\nDownloading {sofa_path.name}...")
            urllib.request.urlretrieve(url, sofa_path)
            print(f"✓ Downloaded successfully to {sofa_path}")
        except Exception as e:
            pytest.skip(f"Test data not available and download failed: {e}")

    return str(sofa_path)


def test_model_single():
    """Test single target inference with fixed random seed for reproducibility."""
    seed = 42
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    am.target = am.target[260]

    posterior = am.infer(repetitions=1, seed=seed)
    estimation = am.estimate(posterior, kappa_motor=0, seed=seed)

    estimated_dir = np.rad2deg(estimation.spherical_elevation[..., 0:2])

    assert isinstance(estimation, pf.Coordinates)
    assert estimation.cartesian.shape == (1, 1, 3)
    assert np.allclose(np.linalg.norm(estimation.cartesian[0, 0, :]),
                       1.0, atol=0.1)

    expected_dir_sph = np.array([[106.874247,  48.32293401]])
    np.testing.assert_allclose(
        estimated_dir.squeeze(), expected_dir_sph.squeeze(), rtol=1e-2)


def test_model_multiple():
    """Test inference with two targets and two repetitions."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    am.parameters = {
        'sigma_itd':      1e-1,
        'sigma_ild':      1e-1,
        'sigma_spectral': 1e-1,
        'sigma_prior':    180,
        'kappa_motor':    0,
    }

    target_indices = [100, 260]
    true_cart = am.coords.cartesian[target_indices, :]

    am.target = am.target[target_indices]

    posterior = am.infer(repetitions=2, seed=42)
    estimation = am.estimate(posterior, kappa_motor=0)
    estimation_cart = estimation.cartesian

    assert estimation_cart.shape == (2, 2, 3), \
        f"Expected shape (2, 2, 3), got {estimation_cart.shape}"

    norms = np.linalg.norm(estimation_cart, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=0.1,
                               err_msg='Estimations should be unit vectors')

    tolerance_deg = 5
    for t_idx in range(2):
        for r_idx in range(2):
            dot = np.clip(
                estimation_cart[t_idx, r_idx, :] @ true_cart[t_idx], -1, 1)
            angular_err_deg = np.rad2deg(np.arccos(dot))
            if angular_err_deg > tolerance_deg:
                return False


def test_interp():
    """Test SHMAX interpolation produces valid template features."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    assert am.template is not None, 'Template should not be None'
    assert isinstance(am.template, Barumerli2023)

    assert am.target.spectral_cues.ndim == 3
    assert am.template.spectral_cues.ndim == 3
    assert am.target.spectral_cues.shape[1:] == am.template.spectral_cues.shape[1:]

    assert np.all(np.isfinite(am.template.spectral_cues)), \
        'Template spectral cues should all be finite'

    side = 0
    amps_target = am.target.spectral_cues[260, :, side]
    template_min = np.min(am.template.spectral_cues[260, :, side])
    template_max = np.max(am.template.spectral_cues[260, :, side])
    target_min = np.min(amps_target)
    target_max = np.max(amps_target)

    margin = 0.2 * (target_max - target_min)
    assert template_min >= target_min - margin
    assert template_max <= target_max + margin


def test_sofa_object_input():
    """Test that BayesianListener accepts a sofar.Sofa object directly (#32)."""
    sofa_file = get_sofa_file()
    sofa_data = sofar.read_sofa(sofa_file, verbose=False)

    am = BayesianListener(sofa_data)

    assert am.sofa_file is None
    assert isinstance(am.coords, pf.Coordinates)
    assert am.hrir is not None
    assert am.fs > 0


def test_sofa_object_no_sofa_data_attr():
    """Test that BayesianListener does not store sofa_data on self (#31)."""
    sofa_file = get_sofa_file()

    am_path = BayesianListener(sofa_file)
    assert not hasattr(am_path, 'sofa_data')

    sofa_data = sofar.read_sofa(sofa_file, verbose=False)
    am_obj = BayesianListener(sofa_data)
    assert not hasattr(am_obj, 'sofa_data')


def test_sofa_object_prepare_features():
    """Test that prepare_features works with sofar.Sofa input (#32)."""
    sofa_file = get_sofa_file()
    sofa_data = sofar.read_sofa(sofa_file, verbose=False)

    am = BayesianListener(sofa_data)
    am.prepare_features(use_cache=False)

    assert am.target is not None
    assert am.template is not None


def test_load_cached_data():
    """Test that data can be saved to and loaded from the cache."""
    sofa_file = get_sofa_file()

    repo_root = Path(__file__).parent.parent
    cache_dir = repo_root / 'data' / 'preprocessed'

    if not cache_dir.exists() or not any(cache_dir.iterdir()):
        pytest.skip('No pre-built cache found — skipping cache load test.')

    loaded = load_from_cache(
        cache_dir, sofa_file,
        attributes_to_restore=['target', 'template'],
        interpolation='SHMAX',
    )

    assert loaded is not None


# -----------------------------------------------------------------------
# New tests for the refactored API
# -----------------------------------------------------------------------

def test_compute_target_sets_barumerli2023():
    """compute_target() sets self.target as Barumerli2023 with correct shapes."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.compute_target()

    assert isinstance(am.target, Barumerli2023)
    n_dirs = am.hrir.shape[0]
    assert am.target.itd.shape[0] == n_dirs
    assert am.target.ild.shape[0] == n_dirs
    assert am.target.spectral_cues.shape[0] == n_dirs
    assert am.target.spectral_cues.ndim == 3
    assert am.target.features.shape[0] == n_dirs


def test_compute_template_sets_Barumerli2023():
    """compute_template() sets self.template on a uniform grid."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.compute_target()
    am.compute_template()

    assert isinstance(am.template, Barumerli2023)
    assert am.template.coords is not None
    assert am.template.spectral_cues.ndim == 3
    assert am.template.features.ndim == 2


def test_compute_template_requires_target():
    """compute_template() raises ValueError if called before compute_target()."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)

    with pytest.raises(ValueError, match='compute_target'):
        am.compute_template()


def test_non_individual_workflow():
    """Non-individual: template from one listener, target swapped from another."""
    sofa_file = get_sofa_file()

    individual = BayesianListener(sofa_file)
    individual.prepare_features()

    # Same file stands in for a different HRTF
    foreign = BayesianListener(sofa_file)
    foreign.prepare_features(compute_template=False)

    assert foreign.target is not None
    assert foreign.template is None

    individual.target = foreign.target
    posterior = individual.infer(repetitions=2, seed=0)
    assert posterior.shape[0] == foreign.target.features.shape[0]


def test_convention_mismatch_raises():
    """infer() raises ValueError when target/template have different types."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.prepare_features()

    class FakeConvention(Barumerli2023):
        convention = 'barumerli2023pge'

    am.template = FakeConvention(
        coords=am.template.coords,
        itd=am.template.itd,
        ild=am.template.ild,
        spectral_cues=am.template.spectral_cues,
        freqs=am.template.freqs,
    )

    with pytest.raises(ValueError, match='Convention mismatch'):
        am.infer()


def test_sigma_motor_roundtrip():
    """sigma_motor getter/setter round-trips through kappa_motor."""
    am = BayesianListener.__new__(BayesianListener)
    am._parameters = {
        'sigma_itd': 0.569, 'sigma_ild': 1.0, 'sigma_spectral': 10.4,
        'sigma_prior': 69.0, 'kappa_motor': 23.31,
    }
    original_kappa = am.parameters['kappa_motor']
    sigma_val = am.sigma_motor
    am.sigma_motor = sigma_val
    np.testing.assert_allclose(
        am.parameters['kappa_motor'], original_kappa, rtol=1e-4)


def test_barumerli2023pge_not_implemented():
    """Instantiating Barumerli2023pge raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Barumerli2023pge(coords=None, itd=None, ild=None,
                         spectral_gradient=None, freqs=None)
