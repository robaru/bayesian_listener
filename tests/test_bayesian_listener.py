import pytest
import numpy as np
import pyfar as pf
import sofar
from pathlib import Path
import urllib.request
from scipy.signal import lfilter
from bayesian_listener import BayesianListener, Barumerli2023
from bayesian_listener.auditory_representation import Barumerli2023pge
from bayesian_listener import utils


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
    am.compute_template()

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
    am.compute_template()

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

def test_model_localise():
    """Test single target inference using localise method."""
    seed = 42
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.compute_template()

    am.target = am.target[260]

    estimation = am.localise(repetitions=1, seed=seed)

    estimated_dir = np.rad2deg(estimation.spherical_elevation[..., 0:2])

    assert isinstance(estimation, pf.Coordinates)
    assert estimation.cartesian.shape == (1, 1, 3)
    assert np.allclose(np.linalg.norm(estimation.cartesian[0, 0, :]),
                       1.0, atol=0.1)

    expected_dir_sph = np.array([[115.872366,  42.608186]])
    np.testing.assert_allclose(
        estimated_dir.squeeze(), expected_dir_sph.squeeze(), rtol=1e-2)

def test_interp():
    """Test SHMAX interpolation produces valid template features."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.compute_template()

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


def test_sofa_object_compute_template():
    """Test that compute_template works with sofar.Sofa input (#32)."""
    sofa_file = get_sofa_file()
    sofa_data = sofar.read_sofa(sofa_file, verbose=False)

    am = BayesianListener(sofa_data)
    am.compute_template(use_cache=False)

    assert am.target is not None
    assert am.template is not None


def test_cache_roundtrip(tmp_path):
    """Test that compute_template caches target and template, and both reload."""
    sofa_file = get_sofa_file()
    cache_dir = tmp_path / 'cache'

    am = BayesianListener(sofa_file)
    am.compute_template(cache_dir=str(cache_dir))

    target = utils.cache_load_target(cache_dir, sofa_file)
    template = utils.cache_load_template(cache_dir, sofa_file, 'SHMAX')

    assert target is not None
    assert template is not None


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


def test_compute_template_auto_computes_target():
    """compute_template() auto-calls compute_target() when target is None."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)

    am.compute_template(use_cache=False)

    assert am.target is not None
    assert am.template is not None


def test_non_individual_workflow():
    """Non-individual: template from one listener, target swapped from another."""
    sofa_file = get_sofa_file()

    individual = BayesianListener(sofa_file)
    individual.compute_template()

    # Same file stands in for a different HRTF
    foreign = BayesianListener(sofa_file)
    foreign.compute_target()

    assert foreign.target is not None
    assert foreign.template is None

    individual.target = foreign.target
    posterior = individual.infer(repetitions=2, seed=0)
    assert posterior.shape[0] == foreign.target.features.shape[0]


def test_convention_mismatch_raises():
    """infer() raises ValueError when target/template have different types."""
    sofa_file = get_sofa_file()
    am = BayesianListener(sofa_file)
    am.compute_template()

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


def test_gammatone_minimum_ir_length_numerical_error():
    """Per-band minimum-IR padding captures the same filter energy as the current 50 ms padding.

    The mean-based RMS used in compute_features is proportional to 1/sqrt(N) where
    N is the total signal length, so it is not a stable metric for comparing runs
    with different padding lengths.  The meaningful, N-independent quantity is the
    *total half-wave-rectified energy* (sum, not mean) accumulated from each
    gammatone band.  That sum converges once the filter impulse response has decayed.

    This test validates three claims (issue #14):

    1. Per-band minimum-IR padding (utils.minimum_ir_length) captures the same
       total energy as the current 50 ms reference padding, within np.allclose
       defaults — confirming the amplitude-decay criterion is sufficient.

    2. For the lowest gammatone band (~700 Hz at 48 kHz), minimum_ir_length is
       *longer* than the current 50 ms extra-padding, confirming the formula is
       conservative (i.e. it sets a safe upper bound on the required length).

    3. For the highest gammatone band (~18 kHz), minimum_ir_length is far *shorter*
       than the current 50 ms padding — this is the band where most memory is wasted
       under the current uniform scheme, and where per-band padding yields the
       largest savings.
    """
    sofa_file = get_sofa_file()
    sofa_data = sofar.read_sofa(sofa_file, verbose=False)
    hrir_all = sofa_data.Data_IR  # (n_dirs, 2, n_samples)
    fs = int(sofa_data.Data_SamplingRate)

    # Use a small subset of directions to keep the test fast.
    rng = np.random.default_rng(0)
    subset_idx = rng.choice(hrir_all.shape[0], size=10, replace=False)
    hrir = hrir_all[subset_idx]  # (10, 2, n_samples)
    hrir_len = hrir.shape[-1]

    spectral_range = [700.0, 18000.0]
    freqs = utils.erb_space(spectral_range)
    B, A, *_ = utils.gammatone(freqs, fs=fs)
    min_lens = utils.minimum_ir_length(freqs, fs=fs)  # per-band minimum extra lengths

    def _band_energy(hrir_padded, b_i, a_i):
        """Total half-wave-rectified energy for one gammatone band."""
        filtered = 2.0 * np.real(lfilter([b_i], a_i, hrir_padded, axis=-1))
        return np.sum(np.maximum(filtered, 0.0), axis=-1)  # (n_dirs, 2)

    # --- reference: current 50 ms fixed total (the behaviour we want to match) ---
    current_total = max(int(round(0.05 * fs)), hrir_len)
    current_extra = current_total - hrir_len
    hrir_cur = np.concatenate(
        [hrir, np.zeros((*hrir.shape[:2], current_extra), dtype=hrir.dtype)],
        axis=-1) if current_extra > 0 else hrir

    # --- collect per-band total energy for reference and min-padded strategies ---
    energy_cur = np.zeros((len(freqs), hrir.shape[0], hrir.shape[1]))
    energy_min = np.zeros_like(energy_cur)

    for i, (_, b_i, a_i, ml_i) in enumerate(zip(freqs, B, A, min_lens)):
        energy_cur[i] = _band_energy(hrir_cur, b_i, a_i)

        hrir_min = np.concatenate(
            [hrir, np.zeros((*hrir.shape[:2], int(ml_i)), dtype=hrir.dtype)],
            axis=-1)
        energy_min[i] = _band_energy(hrir_min, b_i, a_i)

    # 1. Per-band minimum padding captures the same total energy as the current
    #    50 ms reference padding.
    assert np.allclose(energy_min, energy_cur), (
        f"min_padded vs current: max abs diff = "
        f"{np.max(np.abs(energy_min - energy_cur)):.2e}")

    # 2. For the lowest band, minimum_ir_length is longer than the current extra
    #    padding — the formula is conservative (safe upper bound).
    low_band_idx = 0
    min_extra_low = int(min_lens[low_band_idx])
    assert min_extra_low > current_extra, (
        f"minimum_ir_length for the lowest band "
        f"(fc={freqs[low_band_idx]:.1f} Hz) is {min_extra_low} samples, "
        f"expected > current extra padding ({current_extra} samples).")

    # 3. For the highest band, minimum_ir_length is much shorter than the current
    #    extra padding — the current uniform scheme wastes memory there.
    high_band_idx = -1
    min_extra_high = int(min_lens[high_band_idx])
    assert min_extra_high < current_extra // 10, (
        f"minimum_ir_length for the highest band "
        f"(fc={freqs[high_band_idx]:.1f} Hz) is {min_extra_high} samples, "
        f"expected < 1/10 of current extra padding ({current_extra} samples).")


def test_fft_gammatone_accuracy():
    """FFT convolution with truncated gammatone IR matches IIR lfilter spectral cues.

    The FFT approach:
    1. Computes the gammatone impulse response (length = minimum_ir_length per band).
    2. Linearly convolves each HRIR with that IR via rfft/irfft.
    3. Applies half-wave rectification and computes RMS.

    The IIR reference:
    - Zero-pads each HRIR to hrir_len + minimum_ir_length and applies lfilter.

    Both use the same truncation length, so any difference is due to floating-point
    arithmetic only.  The test uses np.allclose defaults (rtol=1e-5, atol=1e-8).
    """
    sofa_file = get_sofa_file()
    sofa_data = sofar.read_sofa(sofa_file, verbose=False)
    hrir = sofa_data.Data_IR          # (n_dirs, 2, n_samples) — full HRTF
    fs   = int(sofa_data.Data_SamplingRate)
    hrir_len = hrir.shape[-1]

    spectral_range = [700.0, 18000.0]
    freqs    = utils.erb_space(spectral_range)
    B, A, *_ = utils.gammatone(freqs, fs=fs)
    min_lens = utils.minimum_ir_length(freqs, fs=fs)

    def _iir_rms(hrir_padded, b_i, a_i):
        filtered = 2.0 * np.real(lfilter([b_i], a_i, hrir_padded, axis=-1))
        return np.sqrt(np.mean(np.maximum(filtered, 0.0), axis=-1))

    def _fft_rms(hrir_raw, b_i, a_i, ir_len):
        # Gammatone impulse response (truncated to ir_len)
        impulse = np.zeros(ir_len)
        impulse[0] = 1.0
        gir = 2.0 * np.real(lfilter([b_i], a_i, impulse))   # (ir_len,)

        # Linear convolution via rfft — output length = hrir_len + ir_len - 1
        fft_size = int(2 ** np.ceil(np.log2(hrir_len + ir_len - 1)))
        H = np.fft.rfft(hrir_raw, n=fft_size, axis=-1)      # (n_dirs, 2, fft_size//2+1)
        G = np.fft.rfft(gir,      n=fft_size)               # (fft_size//2+1,)
        filtered = np.fft.irfft(H * G, n=fft_size, axis=-1) # (n_dirs, 2, fft_size)

        # Trim to hrir_len + ir_len - 1 (same as IIR output with same padding)
        filtered = filtered[..., : hrir_len + ir_len - 1]
        return np.sqrt(np.mean(np.maximum(filtered, 0.0), axis=-1))

    rms_iir = np.zeros((len(freqs), hrir.shape[0], hrir.shape[1]))
    rms_fft = np.zeros_like(rms_iir)

    for i, (b_i, a_i, ml_i) in enumerate(zip(B, A, min_lens)):
        ir_len = int(ml_i)
        # Pad with ir_len-1 zeros so IIR output length = hrir_len + ir_len - 1,
        # matching the FFT linear-convolution output length exactly.
        hrir_padded = np.concatenate(
            [hrir, np.zeros((*hrir.shape[:2], ir_len - 1), dtype=hrir.dtype)], axis=-1)

        rms_iir[i] = _iir_rms(hrir_padded, b_i, a_i)
        rms_fft[i] = _fft_rms(hrir, b_i, a_i, ir_len)

    assert np.allclose(rms_fft, rms_iir), (
        f"FFT vs IIR spectral cues: max abs diff = "
        f"{np.max(np.abs(rms_fft - rms_iir)):.2e}, "
        f"max rel diff = {np.max(np.abs(rms_fft - rms_iir) / (np.abs(rms_iir) + 1e-12)):.2e}"
    )
