"""This model contains helpful utility functions."""
import numpy as np
from scipy.signal import butter, hilbert, correlate, lfilter
from scipy.io import loadmat
from math import factorial
from numba import jit, prange
from joblib import Parallel, delayed
import pyfar as pf
import hashlib
import pandas as pd
from pathlib import Path
import pickle
import datetime
import psutil
import os

# feature functions
def mag2db(mag):
    return 20 * np.log10(mag)

def erb_space(freq_range=[7e2, 18e3], erb_spacing=1):
    #translated from amt/audspacebw.m focusing on ERB-rate scale

    # Convert frequency limits to auditory scale (ERB-rate scale)
    audlimits = \
        9.2645 * np.sign(freq_range) * np.log(1 + np.abs(freq_range) * 0.00437)
    audrange = audlimits[1] - audlimits[0]

    # Calculate number of points (excluding final point)
    n = int(np.floor(audrange / erb_spacing))

    # Compute remainder to center points between low and high freq
    remainder = audrange - n * erb_spacing

    # Auditory points
    audpoints = audlimits[0] + np.arange(n + 1) * erb_spacing + remainder / 2

    # Add final point
    n += 1

    # Convert auditory scale points back to Hz
    fc = (1 / 0.00437) * np.sign(audpoints) * \
        (np.exp(np.abs(audpoints) / 9.2645) - 1)

    return fc

def gammatone(
    fc,
    fs,
    n=4,
    betamul=None,
    scale="0dBforall",     # {"0dBforall", "6dBperoctave"}
    phase="causalphase",   # {"causalphase", "peakphase", "exppeakphase"}
):
    """
    Complex-valued, all-pole gammatone filter coefficients as in Lyon, 1997.
    The function has been taken from the AMT/gammatone.m.

    Parameters
    ----------
    fc : array-like or scalar
        Center frequency(ies) in Hz. Must be within (0, fs/2].
    fs : float
        Sampling rate (Hz), positive.
    n : int, default=4
        Filter order (positive integer).
    betamul : float or None
        Multiplier for ERB bandwidth. If None, uses the MATLAB formula:
            betamul = (factorial(n-1)^2) / (pi*factorial(2n-2) * 2^(-(2n-2)))
    scale : {"0dBforall", "6dBperoctave"}
        Amplitude scaling mode.
    phase : {"causalphase", "peakphase", "exppeakphase"}
        Phase option. "exppeakphase" additionally aligns maxima using an
        impulse response simulation (requires SciPy).

    Returns
    -------
    b : np.ndarray, shape (n_channels,)
        Numerator scalar for each channel (complex).
    a : np.ndarray, shape (n_channels, n+1)
        Denominator polynomial coefficients (complex), leading 1 per row.
    delay : np.ndarray, shape (n_channels,)
        Peak time (seconds) of the envelope: 3 / (2*pi*beta).
    z : list
        Zeros per channel (always empty lists here, all-pole).
    p : list
        Poles per channel (length n per channel; all identical).
    k : list
        Gain per channel (always 1 here, to mirror the zpk form in MATLAB).
    """
    # ---- validation ----
    if fs <= 0 or not np.isscalar(fs):
        raise ValueError("fs must be a positive scalar.")

    fc = np.atleast_1d(np.asarray(fc, dtype=float))
    if np.any(fc <= 0) or np.any(fc > fs / 2):
        raise ValueError("fc must be > 0 and <= fs/2.")

    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise ValueError("n must be a positive integer.")

    if scale not in {"0dBforall", "6dBperoctave"}:
        raise ValueError("scale must be '0dBforall' or '6dBperoctave'.")

    if phase not in {"causalphase", "peakphase", "exppeakphase"}:
        raise ValueError(
            "phase must be 'causalphase', 'peakphase', or 'exppeakphase'.")

    # ---- bandwidth multiplier (match the MATLAB formula literally) ----
    if betamul is None:
        betamul = (factorial(n - 1) ** 2) / (
            np.pi * factorial(2 * n - 2) * (2 ** (-(2 * n - 2)))
        )
    elif not (np.isscalar(betamul) and betamul > 0):
        raise ValueError("betamul must be a positive scalar.")

    # ---- bandwidths and per-channel constants ----
    # critical bandwidth of the auditory filter at center frequency fc
    #  defined in equivalent rectangular bandwidthGlasberg and Moore (1990)
    audfiltbws = 24.7 + fc/9.265
    beta = betamul * audfiltbws # ourbeta in MATLAB
    nch = fc.size

    # Peak time of the envelope (seconds)
    delay = 3.0 / (2.0 * np.pi * beta)

    # Allocate outputs
    b = np.zeros((nch,), dtype=np.complex128)
    a = np.zeros((nch, n + 1), dtype=np.complex128)
    z = []
    p = []
    k = []

    # ---- design each channel ----
    for i in range(nch):
        # Complex pole location (all-pole, repeated n times)
        atilde = np.exp(-2*np.pi * beta[i] / fs - 1j * 2*np.pi * fc[i] / fs)

        # Denominator from repeated root (length n+1, leading 1)
        # np.poly takes roots and returns monic polynomial coefficients.
        a_i = np.poly(np.full(n, atilde))

        # Base numerator (MATLAB: btmp = 1 - exp(-2*pi*beta/fs))
        btmp = 1.0 - np.exp(-2 * np.pi * beta[i] / fs)

        # Amplitude scaling
        if scale == "6dBperoctave":
            b_i = (btmp ** n) * (fs / fc[i] / n)
        else:
            b_i = (btmp ** n)

        # Phase options
        if phase == "peakphase":
            # Multiply by exp(j*2*pi*fc*delay)
            b_i = b_i * np.exp(1j * 2 * np.pi * fc[i] * delay[i])

        elif phase == "exppeakphase":
            # Simulate impulse to find envelope & signal peak offset
            insig = np.zeros(8192, dtype=np.complex128)
            insig[0] = 1.0
            outsig = lfilter([b_i], a_i, insig)  # complex IIR
            # Following the MATLAB: use 2*real(...) when ultimately used;
            # here we mirror its peak alignment logic.
            tmp = 2.0 * np.real(outsig)
            envmax = np.argmax(np.abs(tmp))
            sigmax = np.argmax(tmp)
            # Equation analogous to the MATLAB code:
            phi_delay = \
                fc[i] * (-2*np.pi - np.pi / 4.0) * (envmax - sigmax) / fs
            b_i = b_i * np.exp(1j * phi_delay)

        # Store results
        b[i] = b_i
        a[i, :] = a_i
        z.append(np.array([], dtype=np.complex128))      # no zeros (all-pole)
        p.append(np.full(n, atilde, dtype=np.complex128))  # n identical poles
        k.append(1.0)

    return b, a, delay, z, p, k

def itdestimator(signals, fs=None):
    """
    Estimate ITD from the given stimulus.

    Parameters
    ----------
        Obj : 3D numpy array or object with IR data
        fs : Sampling rate (required if Obj is a 3D array)

    Returns
    -------
        toa_diff : Time of arrival difference
    """

    pos = signals.shape[0]
    ear = signals.shape[1]
    Ns = signals.shape[2]
    IR = signals

    if fs is None:
        raise ValueError('No sampling rate (fs) provided.')

    # Initialize variables
    toa_diff = np.zeros((pos, 1))
    IACC = np.zeros(pos)

    # Example: Applying low-pass filter (if needed)
    # Assuming AMT values for Butterworth filter parameters
    butterpoly = 10
    upper_cutfreq = 3000
    cut_off_freq_norm = upper_cutfreq / (fs / 2)
    lp_b, lp_a = butter(butterpoly, cut_off_freq_norm)

    f_ir = np.zeros((pos, ear, Ns))
    for ii in range(pos):
        for jj in range(ear):
            f_ir[ii, jj, :] = lfilter(lp_b, lp_a, IR[ii, jj, :])

    # Compute ITD using MaxIACCe mode
    for ii in range(pos):
        e_sir1 = np.abs(hilbert(f_ir[ii, 0, :]))
        e_sir2 = np.abs(hilbert(f_ir[ii, 1, :]))
        cc = correlate(e_sir1, e_sir2, mode='full')
        IACC[ii] = np.max(np.abs(cc))
        idx_lag = np.argmax(np.abs(cc))
        toa_diff[ii] = idx_lag - (Ns - 1)  # Adjust for 0-based indexing

    return toa_diff / fs

# -----------------------------------
# SPHERICAL UTILITIES
# -----------------------------------

def scatter_von_mises(dirs, kappa, seed = None):
    """Apply von Mises-Fisher distributed noise to direction vectors.

    Parameters
    ----------
    dirs : ndarray
        Direction vectors in Cartesian coordinates (n x 3) or (3,).
    kappa : float
        Von Mises-Fisher concentration parameter (higher = less noise).
    seed : int, optional
        Random seed for reproducibility.
    """
    assert dirs.shape[1] == 3 or dirs.size == 3
    assert kappa > 0, 'kappa must be positive'

    dirs = np.squeeze(dirs)

    dirs_new = np.zeros_like(dirs)

    if dirs.ndim > 1:
        for i in range(dirs.shape[0]):
            dirs_new[i, :] = randvmf(kappa, dirs[i, :], seed=seed)
    else:
        dirs_new = randvmf(kappa, dirs, seed=seed)

    return dirs_new

def randvmf(kappa, mu, seed = None):
    rng = np.random.default_rng(seed)


    assert mu is not None
    assert kappa > 0

    # remove useless dimensions
    mu = mu.squeeze()

    Np = np.array([0., 0., 1.])

    ## density
    # Rubinstein 81, p.39, Fisher 87, p.59
    kappaS = np.sign(kappa)
    kappa = abs(kappa)
    U = rng.random()
    x = np.log(2. * U * np.sinh(kappa) + np.exp(-kappa)) / kappa
    x = kappaS * x

    psi = 2. * np.pi * rng.random()
    s_x = np.sqrt(1. - x**2.)
    y = np.array([np.cos(psi) * s_x, np.sin(psi) * s_x, x])

    mu = mu / np.linalg.norm(mu)

    if np.linalg.norm(mu - Np) > np.finfo(float).eps:
        if mu[2] != 1:
            Ux = np.cross(Np, mu.T)
            Ux = Ux / np.linalg.norm(Ux)
            thetaX = np.arccos(mu[2])
            Rg = rodriguesrotation(Ux * thetaX)
            y = y @ Rg

    return y

def rodriguesrotation(axis_angle):
    theta = np.linalg.norm(axis_angle)
    if theta < np.finfo(float).eps:
        return np.eye(3)
    axis = axis_angle.T / theta
    K = np.zeros((3,3))
    K[0,:]=[0., -axis[2], axis[1]]
    K[1,:]=[axis[2], 0., -axis[0]]
    K[2,:]=[-axis[1], axis[0], 0.]

    M = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    # not sure why the transpose but it is correct
    return M.T

# -----------------------------------
# INFERENCE
# -----------------------------------

# inference functions
def multiple_logpdfs_vec_input(xs, means, covs):
    """
    `multiple_logpdfs` assuming `xs` has shape (N samples, P features).
    means is NxP and covs is NxPxP
    https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs = 1./vals

    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us   = vecs * np.sqrt(valsinvs)[:, None]
    devs = xs[:, None, :] - means[None, :, :]

    # Use `einsum` for matrix-vector multiplications
    # across the first dimension.
    devUs = np.einsum('jnk,nki->jni', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=2)

    # Compute and broadcast scalar normalizers.
    dim    = xs.shape[1]
    log2pi = np.log(2 * np.pi)

    out = -0.5 * (dim * log2pi + mahas + logdets[None, :])
    return out

def _multiple_logpdfs_vec_input_single_cov_numpy(xs, means, logdet, Us):
    """Pure-numpy reference implementation of multivariate normal log-pdf.

    Equivalent to the numba-accelerated `multiple_logpdfs_vec_input_single_cov`
    but without JIT compilation. Useful for debugging and validating the numba
    version, since it is easier to inspect intermediate arrays.

    Parameters
    ----------
    xs : ndarray, shape (N, P)
        Sample points.
    means : ndarray, shape (M, P)
        Distribution means.
    logdet : float
        Log-determinant of the covariance matrix.
    Us : ndarray, shape (P, P)
        Whitening matrix (inverse square-root of covariance).

    Returns
    -------
    out : ndarray, shape (N, M)
        Log-pdf of each sample under each mean.

    See Also
    --------
    multiple_logpdfs_vec_input_single_cov : Numba-accelerated version used in
        production.
    """
    devs = xs[:, None, :] - means[None, :, :]

    # Use `einsum` for matrix-vector multiplications
    # across the first dimension.
    # devUs = np.einsum('jnk,ki->jni', devs, Us) ->
    # using this notation is very slow (twice as much)
    devUs = devs @ Us

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=2)

    # Compute and broadcast scalar normalizers.
    dim    = xs.shape[1]
    log2pi = np.log(2 * np.pi)

    out = -0.5 * (dim * log2pi + mahas + logdet)
    return out

@jit(nopython=True, parallel=True)
def multiple_logpdfs_vec_input_single_cov(xs, means, logdet, Us):
    """
    `multiple_logpdfs` assuming `xs` has shape (N samples, P features).
    means is NxP and cov is PxP
    https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/

    The big idea is to do one intensive operation, eigenvalue decomposition,
    and then use that decomposition to compute the matrix inverse
    and determinant cheaply.
    """

    n_samples = xs.shape[0]
    n_means = means.shape[0]
    n_features = xs.shape[1]

    # Pre-allocate output
    out = np.empty((n_samples, n_means))

    # Compute scalar normalizers once
    dim = n_features
    log2pi = np.log(2 * np.pi)
    normalizer = -0.5 * (dim * log2pi + logdet)

    # Parallelize over samples
    for i in prange(n_samples):
        for j in range(n_means):
            # Compute deviation
            dev = xs[i] - means[j]

            # Matrix-vector multiplication
            devU = dev @ Us

            # Mahalanobis distance
            maha = np.sum(devU * devU)

            # Final log pdf
            out[i, j] = normalizer - 0.5 * maha

    return out

@jit(nopython=True, parallel=True)
def multiple_logpdfs_vec_input_single_cov_diagonal(xs,
                                                   means,
                                                   logdet,
                                                   sigma_inv_diag):
    """
    Optimized version for diagonal covariance.
    sigma_inv_diag: (P,) array of 1/sigma_i²
    """
    devs = xs[:, None, :] - means[None, :, :]  # (n_samples, n_means, P)
    # Mahalanobis distance for diagonal cov:
    # dev.T @ inv(Σ) @ dev = sum(dev² / sigma²)
    mahas = np.sum(devs**2 * sigma_inv_diag, axis=2)  # (n_samples, n_means)

    dim = xs.shape[1]
    normalizer = -0.5 * (dim * np.log(2*np.pi) + logdet)
    return normalizer - 0.5 * mahas


# -----------------------------------
# CACHING SYSTEM
# -----------------------------------

# helper functions for sofa caching
def _compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clear_cache(sofa_file=None):
    """
    Remove cached preprocessed data.

    Parameters
    ----------
    sofa_file : str or None
        If provided, only removes cache for this specific file.
        If None, removes all cached data.
    """
    cache_dir = Path('data/preprocessed')
    cache_index_file = cache_dir / 'cache_index.csv'

    if not cache_dir.exists():
        print("✓ No cache to clear")
        return

    if sofa_file is None:
        # Clear everything
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ All cache cleared: {cache_dir}")
    else:
        # Clear specific file
        sofa_name = Path(sofa_file).name
        if cache_index_file.exists():
            cache_df = pd.read_csv(cache_index_file)
            matches = cache_df[cache_df['sofa_name'] == sofa_name]

            if not matches.empty:
                # Remove pickle files
                for pkl_file in matches['pkl_file']:
                    pkl_path = cache_dir / pkl_file
                    if pkl_path.exists():
                        pkl_path.unlink()
                        print(f"✓ Removed: {pkl_file}")

                # Update index
                cache_df = cache_df[cache_df['sofa_name'] != sofa_name]
                cache_df.to_csv(cache_index_file, index=False)
                print(f"✓ Cache cleared for: {sofa_name}")
            else:
                print(f"✓ No cache found for: {sofa_name}")
        else:
            print("✓ No cache index found")

def load_from_cache(cache_dir,
                    sofa_file,
                    attributes_to_restore,
                    interpolation='SH'):
    """
    Try to load cached data from pickle file.

    Parameters
    ----------
    cache_dir : Path or str
        Directory containing cache files
    sofa_file : str
        Path to SOFA file (used for hash matching)
    attributes_to_restore : list of str
        List of attribute names to restore from cache
    interpolation : str
        Interpolation method used (e.g., 'SH', 'barumerli2023')

    Returns
    -------
    dict or None
        Dictionary of cached attributes, or None if cache not found/invalid
    """
    cache_dir = Path(cache_dir)
    cache_index_file = cache_dir / 'cache_index.csv'

    if not cache_index_file.exists():
        return None

    file_hash = _compute_file_hash(sofa_file)
    sofa_name = Path(sofa_file).name

    # Load cache index
    cache_df = pd.read_csv(cache_index_file)

    # Handle backward compatibility: add 'interpolation' column if missing
    if 'interpolation' not in cache_df.columns:
        # Assume old caches used default SH method
        cache_df['interpolation'] = 'SH'

    # Check for matching entry (including interpolation method)
    match = cache_df[(cache_df['sofa_name'] == sofa_name) &
                     (cache_df['file_hash'] == file_hash) &
                     (cache_df['interpolation'] == interpolation)]

    if match.empty:
        return None

    pkl_file = cache_dir / match.iloc[0]['pkl_file']
    if not pkl_file.exists():
        return None

    print(f"✓ Loading from cache: {pkl_file.name}")
    try:
        with open(pkl_file, 'rb') as f:
            cached_data = pickle.load(f)

        # Validate that all required attributes are present
        missing = [
            attr for attr in attributes_to_restore if attr not in cached_data]
        if missing:
            print(f"⚠ Cache missing attributes: {missing}")
            return None

        print("✓ Cache loaded successfully")
        return cached_data

    except Exception as e:
        print(f"⚠ Cache loading failed: {e}")
        return None

def save_to_cache(cache_dir, sofa_file, data_to_cache, interpolation='SH'):
    """
    Save data to cache with automatic indexing.

    Parameters
    ----------
    cache_dir : Path or str
        Directory to store cache files
    sofa_file : str
        Path to SOFA file (used for naming and indexing)
    data_to_cache : dict
        Dictionary of data to pickle
    interpolation : str
        Interpolation method used (e.g., 'SH', 'barumerli2023')

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index_file = cache_dir / 'cache_index.csv'

    file_hash = _compute_file_hash(sofa_file)
    sofa_name = Path(sofa_file).name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load existing cache index
    if cache_index_file.exists():
        cache_df = pd.read_csv(cache_index_file)
        # Handle backward compatibility: add 'interpolation' column if missing
        if 'interpolation' not in cache_df.columns:
            # Assume old caches used default SH method
            cache_df['interpolation'] = 'SH'
    else:
        cache_df = pd.DataFrame(columns=[
            'sofa_name',
            'file_hash',
            'interpolation',
            'pkl_file',
            'timestamp',
            ])

    # Check if entry already exists (same name, hash, and interpolation)
    match = cache_df[(cache_df['sofa_name'] == sofa_name) &
                     (cache_df['file_hash'] == file_hash) &
                     (cache_df['interpolation'] == interpolation)]

    # Create new pickle filename with interpolation method and timestamp
    pkl_filename = (
        f"{Path(sofa_name).stem}_{file_hash[:8]}"
        f"_{interpolation}_{timestamp}.pkl"
    )

    pkl_path = cache_dir / pkl_filename

    if not match.empty:
        # Remove old pickle file
        old_pkl_filename = match.iloc[0]['pkl_file']
        old_pkl_path = cache_dir / old_pkl_filename
        if old_pkl_path.exists():
            old_pkl_path.unlink()
            print(f"→ Removed old cache: {old_pkl_filename}")
        print(f"→ Updating cache: {pkl_filename}")
    else:
        print(f"→ Saving to cache: {pkl_filename}")

    try:
        # Save new pickle file
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        if not match.empty:
            # Update existing entry with new filename and timestamp
            cache_df.loc[match.index[0], 'pkl_file'] = pkl_filename
            cache_df.loc[match.index[0], 'timestamp'] = timestamp
        else:
            # Remove old entries for same file with different hash
            # (but keep different interpolations)
            old_entries = cache_df[
                (cache_df['sofa_name'] == sofa_name) &
                (cache_df['file_hash'] != file_hash) &
                (cache_df['interpolation'] == interpolation)]

            # Delete old pickle files with different hashes
            for _, row in old_entries.iterrows():
                old_file = cache_dir / row['pkl_file']
                if old_file.exists():
                    old_file.unlink()
                    print(f"→ Removed outdated cache: {row['pkl_file']}")

            # Remove old entries from dataframe
            cache_df = cache_df[~(
                (cache_df['sofa_name'] == sofa_name) &
                (cache_df['file_hash'] != file_hash) &
                (cache_df['interpolation'] == interpolation))]

            # Add new entry
            new_row = pd.DataFrame([{
                'sofa_name': sofa_name,
                'file_hash': file_hash,
                'interpolation': interpolation,
                'pkl_file': pkl_filename,
                'timestamp': timestamp,
            }])
            cache_df = pd.concat([cache_df, new_row], ignore_index=True)

        cache_df.to_csv(cache_index_file, index=False)
        print("✓ Cache saved and index updated")
        return True

    except Exception as e:
        print(f"⚠ Cache save failed: {e}")
        return False


# -----------------------------------
# VARIOUS
# -----------------------------------

def print_memory_usage(label=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    print(f"[{label}] Memory usage: {mem_gb:.2f} GB")


def compute_features(hrir, coords, fs, spectral_range=[7e2, 18e3]):
    """Compute ITD, ILD, and spectral cues from HRIRs.

    Parameters
    ----------
    hrir : ndarray
        Head-related impulse responses, shape (n_dirs, 2, n_samples).
    coords : pyfar.Coordinates
        Source positions corresponding to the HRIR directions.
    fs : int
        Sampling rate in Hz.
    spectral_range : list, default=[7e2, 18e3]
        Frequency range [low, high] in Hz for spectral analysis.

    Returns
    -------
    itd : ndarray
        Interaural time differences, shape (n_dirs, 1).
    ild : ndarray
        Interaural level differences, shape (n_dirs, 1).
    spectral_cues : ndarray
        Spectral amplitudes, shape (n_dirs, n_freqs, 2).
    freqs : ndarray
        Centre frequencies of the gammatone filterbank.
    """
    # normalize hrirs to frontal position
    coords2find = pf.Coordinates.from_cartesian(1, 0, 0)
    idx, _ = coords.find_nearest(coords2find)
    hrirs_temp = hrir / np.max(np.abs(hrir[idx]))

    a = 32.5e-6
    b = 0.095

    # ITD
    itd_raw = itdestimator(hrirs_temp, fs=fs)
    itd = np.sign(itd_raw) * ((np.log(a + b*np.abs(itd_raw)) - np.log(a)) / b)

    # ILD
    ild = np.ones_like(itd)
    ild[:, 0] = (
        mag2db(np.sqrt(np.mean(hrirs_temp[:, 0, :]**2, axis=1))) -
        mag2db(np.sqrt(np.mean(hrirs_temp[:, 1, :]**2, axis=1)))
    )

    # padding to account for longer filter responses
    pad_len_sec = 0.05  # 50 ms
    time_len = hrirs_temp.shape[2]
    dir_len = hrirs_temp.shape[0]
    ear_len = hrirs_temp.shape[1]
    target_samples = int(round(pad_len_sec * fs))

    if time_len < target_samples:
        pad_samples = target_samples - time_len
        pad_mat = np.zeros((dir_len, ear_len, pad_samples),
                           dtype=hrirs_temp.dtype)
        hrirs_temp = np.concatenate([hrirs_temp, pad_mat], axis=2)

    # generate gammatone filterbank
    freqs = erb_space(spectral_range)
    B, A, *_ = gammatone(freqs, fs=fs)

    # Preallocate output array
    hrirs_filt = np.zeros((len(freqs), *hrirs_temp.shape), dtype=float)

    # Parallel gammatone filtering
    def apply_filter(i):
        return 2 * np.real(lfilter([B[i]], A[i], hrirs_temp, axis=-1))

    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(apply_filter)(i) for i in range(len(freqs))
    )

    for i, result in enumerate(results):
        hrirs_filt[i] = result

    # Rectification and compression
    hrirs_filt = np.sqrt(np.maximum(hrirs_filt, 0))

    # average over time -> spectral amplitude
    rms = np.sqrt(np.mean(hrirs_filt**2, axis=-1))
    spectral_cues = mag2db(rms).transpose(1, 0, 2)

    return itd, ild, spectral_cues, freqs


def load_n_design(degree):
    """Load a spherical t-design (n-design) grid of the given degree.

    Points are loaded from a bundled .mat file covering degrees 1–124.
    These are Chebyshev-type quadrature rules on the unit sphere, equivalent
    to modern t-designs.

    Parameters
    ----------
    degree : int
        Degree of exactness, between 1 and 124.

    Returns
    -------
    vecs : ndarray, shape (M, 3)
        Cartesian coordinates of the grid points on the unit sphere.

    References
    ----------
    The grid data (``n_designs_1_124.mat``) was originally published by
    Manuel Gräf at https://homepage.univie.ac.at/manuel.graef/quadrature.php
    and redistributed by spaudiopy (MIT License, Copyright 2019 Chris Hold).
    The upstream data source does not carry an explicit license.
    """
    if degree < 1 or degree > 124:
        raise ValueError('degree must be between 1 and 124.')

    mat_path = Path(__file__).parent / 'data' / 'n_designs_1_124.mat'
    mat = loadmat(mat_path)

    key = 'N' + f'{degree:03}'
    if key not in mat:
        return load_n_design(degree + 1)

    return mat[key]


def vbap_interpolate(src, grid, norm=1):
    """Compute VBAP interpolation weights on the unit sphere.

    For each source direction, finds the enclosing triangle on the convex hull
    of `grid` and returns the panning gains normalised according to `norm`.

    Parameters
    ----------
    src : ndarray, shape (n_src, 3)
        Cartesian coordinates of target directions.
    grid : ndarray, shape (n_grid, 3)
        Cartesian coordinates of the source grid (unit sphere).
    norm : {1, 2}, default=1
        Gain normalisation:
        ``1`` — gains sum to 1 (anechoic, equivalent to barycentric
        interpolation);
        ``2`` — sum of squared gains equals 1 (energy-preserving /
        reverberant).

    Returns
    -------
    weights : ndarray, shape (n_src, n_grid)
        Sparse-like weight matrix. Each row has at most 3 non-zero entries
        normalised according to `norm`.

    References
    ----------
    Algorithm adapted from spaudiopy (MIT License, Copyright 2019 Chris Hold,
    https://github.com/chris-hold/spaudiopy), based on:
    Pulkki, V. (1997). Virtual Sound Source Positioning Using Vector Base
    Amplitude Panning. JAES, 45(6), 456–466.
    """
    from scipy.spatial import ConvexHull

    if norm not in (1, 2):
        raise ValueError('norm must be 1 or 2.')

    hull = ConvexHull(grid)
    n_src = src.shape[0]
    n_grid = grid.shape[0]
    weights = np.zeros((n_src, n_grid))

    for i, s in enumerate(src):
        best_tri = None
        best_g = None
        best_neg = np.inf

        for simplex in hull.simplices:
            V = grid[simplex].T          # (3, 3)
            try:
                g = np.linalg.solve(V, s)
            except np.linalg.LinAlgError:
                continue
            neg = -np.min(g)             # 0 if all gains >= 0
            if neg < best_neg:
                best_neg = neg
                best_tri = simplex
                best_g = g

        if best_tri is not None:
            g = np.maximum(best_g, 0)
            if norm == 1:
                g /= g.sum()
            else:
                g /= np.sqrt(np.sum(g ** 2))
            weights[i, best_tri] = g

    return weights
# %%
