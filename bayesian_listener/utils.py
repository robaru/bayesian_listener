"""Utility functions for feature extraction, spherical sampling, and caching.

Public sections:

- **Feature extraction.**  :func:`compute_features`, :func:`gammatone`,
  :func:`erb_space`, :func:`itdestimator`, :func:`mag2db`.
- **Spherical utilities.**  :func:`scatter_von_mises`, :func:`load_n_design`,
  :func:`vbap_interpolate`.
- **Inference helpers.**  Vectorised Gaussian log-pdf evaluators
  :func:`multiple_logpdfs_vec_input`,
  :func:`multiple_logpdfs_vec_input_single_cov`, and
  :func:`multiple_logpdfs_vec_input_single_cov_diagonal`.
- **Caching.**  :func:`save_to_cache`, :func:`load_from_cache`,
  :func:`clear_cache`.
"""
import numpy as np
from scipy.signal import butter, hilbert, correlate, lfilter
from scipy.io import loadmat
from math import factorial
from numba import jit, prange
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
    r"""Convert a linear magnitude to decibels.

    Computes :math:`20 \log_{10}(\mathrm{mag})`.

    Parameters
    ----------
    mag : float or :class:`numpy.ndarray`
        Linear magnitude (positive).

    Returns
    -------
    float or :class:`numpy.ndarray`
        Magnitude in dB, same shape as ``mag``.

    Examples
    --------
    >>> float(mag2db(10.0))
    20.0
    """
    return 20 * np.log10(mag)

def erb_space(freq_range=[7e2, 18e3], erb_spacing=1):
    r"""Generate centre frequencies on the equivalent-rectangular-bandwidth (ERB) scale.

    Translated from AMT 1.x ``audspacebw.m`` (:footcite:t:`glasberg1990`); converts the
    bracketing frequencies to the ERB-rate scale, places points spaced by
    ``erb_spacing`` ERBs, then maps back to Hz.

    Parameters
    ----------
    freq_range : list of float, default=[700.0, 18000.0]
        ``[low_Hz, high_Hz]`` bracket.
    erb_spacing : float, default=1
        Spacing between centre frequencies on the ERB-rate scale.

    Returns
    -------
    :class:`numpy.ndarray`
        Centre frequencies in Hz, shape ``(n_freqs,)``, monotonically
        increasing.

    Examples
    --------
    >>> fc = erb_space([1000.0, 4000.0])
    >>> bool(fc[0] >= 1000.0 and fc[-1] <= 4000.0)
    True
    """
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


def minimum_ir_length(fc, fs, n=4, tolerance=5e-5):
    """Minimum impulse response length for Lyon 1997 gammatone band(s).

    Adapts the pole-decay algorithm from pyfar / MATLAB ``impzlength`` to the
    Lyon 1997 all-pole gammatone filter.  The dominant pole of each band has
    magnitude ``|z| = exp(-2π·β/fs)``; the impulse response of an ``n``-th
    order filter with that repeated pole decays as ``|z|^k / k^(n-1)``, so
    the number of samples required to fall below ``tolerance`` is

        ``n · log10(tolerance) / log10(|z|)``

    which is the pyfar formula for the "no-oscillation" IIR case with pole
    multiplicity ``n``.

    Parameters
    ----------
    fc : float or array-like
        Centre frequency (Hz) of each gammatone band.  Must satisfy
        ``0 < fc <= fs/2``.
    fs : float
        Sampling rate (Hz).
    n : int, default=4
        Filter order (must match the ``n`` passed to :func:`gammatone`).
    tolerance : float, default=5e-5
        Amplitude threshold below which the impulse response is considered
        negligible.  The default ``5e-5`` matches pyfar's default.

    Returns
    -------
    lengths : ndarray of int, shape ``(n_bands,)``
        Minimum number of samples for the impulse response of each band to
        decay to ``tolerance``.  Always at least ``n + 1``.
    """
    fc = np.atleast_1d(np.asarray(fc, dtype=float))
    audfiltbws = 24.7 + fc / 9.265
    betamul = (factorial(n - 1) ** 2) / (
        np.pi * factorial(2 * n - 2) * 2 ** (-(2 * n - 2))
    )
    beta = betamul * audfiltbws
    pole_mag = np.exp(-2.0 * np.pi * beta / fs)
    lengths = n * np.log10(tolerance) / np.log10(pole_mag)
    return np.maximum(np.ceil(lengths).astype(int), n + 1)


@jit(nopython=True, parallel=True)
def _gammatone_rms_numba(hrir, B, A, min_lens, halfwave_rectifier):
    """Fused gammatone IIR filter → optional half-wave rectification → RMS.

    Processes all gammatone bands concurrently across CPU cores (OpenMP via
    ``prange``) without storing any intermediate filtered signal.  For each
    band/direction/ear triplet the IIR recursion, rectification, and
    accumulation happen in a single sequential pass over the time axis, keeping
    memory use proportional only to the output array.

    The filter order is fixed at 4 (Lyon 1997 all-pole gammatone), so ``A``
    must have shape ``(n_bands, 5)``.

    Parameters
    ----------
    hrir : ndarray, shape (n_dirs, n_ears, n_samples), float64
        Normalised HRIRs.  Must be C-contiguous float64.
    B : ndarray, shape (n_bands,), complex128
        Gammatone numerator scalars from :func:`gammatone`.
    A : ndarray, shape (n_bands, 5), complex128
        Gammatone denominator coefficients from :func:`gammatone`.
    min_lens : ndarray, shape (n_bands,), int64
        Per-band minimum impulse response lengths from
        :func:`minimum_ir_length`.  Band ``i`` is extended by
        ``min_lens[i] - 1`` implicit zeros beyond the HRIR.
    halfwave_rectifier : bool
        ``True``  → accumulate ``max(val, 0)``   → ``sqrt(mean(hwr(x)))``.
        ``False`` → accumulate ``val²``           → standard RMS.

    Returns
    -------
    ndarray, shape (n_bands, n_dirs, n_ears), float64
        Per-band amplitude (linear, before dB conversion), normalised by the
        original HRIR length so the value is invariant to padding length.
    """
    n_bands = B.shape[0]
    n_dirs  = hrir.shape[0]
    n_ears  = hrir.shape[1]
    n_samp  = hrir.shape[2]
    rms_out = np.zeros((n_bands, n_dirs, n_ears))
    for band in prange(n_bands):
        ir_len    = min_lens[band]
        total_len = n_samp + ir_len - 1
        b  = B[band]
        a1 = A[band, 1]
        a2 = A[band, 2]
        a3 = A[band, 3]
        a4 = A[band, 4]
        for d in range(n_dirs):
            for e in range(n_ears):
                y1 = y2 = y3 = y4 = 0j
                acc = 0.0
                for n in range(total_len):
                    xn = hrir[d, e, n] if n < n_samp else 0.0
                    yn = b * xn - a1 * y1 - a2 * y2 - a3 * y3 - a4 * y4
                    val = 2.0 * yn.real
                    if halfwave_rectifier:
                        # accumulates max(x,0); sqrt(mean(max(x,0))) is
                        # equivalent to the original sqrt(mean((sqrt(max(x,0)))²))
                        # because (sqrt(a))² = a for a >= 0
                        if val > 0.0:
                            acc += val
                    else:
                        acc += val * val
                    y4 = y3
                    y3 = y2
                    y2 = y1
                    y1 = yn
                # normalise by original HRIR length (energy per input sample),
                # not the padded total_len,
                # so the cue is invariant to padding and compatible with
                # sigma_spectral values fitted on the old fixed-50ms implementation
                rms_out[band, d, e] = np.sqrt(acc / n_samp)
    return rms_out


def itdestimator(signals, fs=None):
    r"""Estimate the interaural time difference (ITD) by Hilbert-envelope MaxIACCe.

    For each direction, low-pass filters the binaural HRIRs at 3 kHz with a
    10th-order Butterworth filter, computes the analytic envelope of each
    ear via the Hilbert transform, and locates the peak of their cross-
    correlation (``MaxIACCe`` mode of AMT 1.x ``itdestimator.m``).

    Parameters
    ----------
    signals : :class:`numpy.ndarray`
        Binaural HRIRs of shape ``(n_dirs, 2, n_samples)``.
    fs : int
        Sampling rate in Hz.  Required.

    Returns
    -------
    :class:`numpy.ndarray`
        ITD in seconds, shape ``(n_dirs, 1)``.

    Raises
    ------
    ValueError
        If ``fs`` is ``None``.
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
    r"""Perturb unit-direction vectors with von Mises–Fisher noise.

    Implements Eq. 7 of :footcite:t:`barumerli2023`: each input direction is replaced
    by a sample from :math:`\mathrm{vMF}(\boldsymbol{\mu}_i, \kappa)`.
    The output preserves the input shape.

    Parameters
    ----------
    dirs : :class:`numpy.ndarray`
        Direction vectors in Cartesian coordinates, shape ``(n, 3)`` or
        ``(3,)`` (each row should be unit-norm).
    kappa : float
        Von Mises–Fisher concentration; higher values yield tighter samples.
        Must be positive.
    seed : int or None, default=None
        Seed forwarded to :func:`numpy.random.default_rng`.

    Returns
    -------
    :class:`numpy.ndarray`
        Perturbed direction vectors, same shape as ``dirs``.

    Raises
    ------
    ValueError
        If ``dirs`` does not have 3 components in its last dimension, or
        if ``kappa`` is not positive.
    """
    if not (dirs.shape[1] == 3 or dirs.size == 3):
        raise ValueError(
            "dirs must be of shape (n, 3) or (3,); "
            f"got shape {dirs.shape}.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    dirs = np.squeeze(dirs)

    dirs_new = np.zeros_like(dirs)

    if dirs.ndim > 1:
        for i in range(dirs.shape[0]):
            dirs_new[i, :] = randvmf(kappa, dirs[i, :], seed=seed)
    else:
        dirs_new = randvmf(kappa, dirs, seed=seed)

    return dirs_new

def randvmf(kappa, mu, seed = None):
    r"""Draw a single sample from a 3-D von Mises–Fisher distribution.

    Uses the ``z``-axis tangent-rotation algorithm of Rubinstein (1981) and
    Fisher et al. (1987): sample on a vMF aligned with the north pole, then
    rotate to align with ``mu`` via a Rodrigues rotation.

    Parameters
    ----------
    kappa : float
        Concentration parameter (positive).
    mu : :class:`numpy.ndarray`
        Mean direction (unit vector), shape ``(3,)``.
    seed : int or None, default=None
        Seed forwarded to :func:`numpy.random.default_rng`.

    Returns
    -------
    :class:`numpy.ndarray`
        Single direction vector, shape ``(3,)``.
    """
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
    r"""Build a 3×3 rotation matrix from a Rodrigues axis–angle vector.

    Given :math:`\boldsymbol{\omega} = \theta \hat{\mathbf{n}}`, returns
    :math:`\mathbf{R} = \mathbf{I} + \sin\theta\, \mathbf{K} +
    (1 - \cos\theta)\, \mathbf{K}^2`, where :math:`\mathbf{K}` is the
    skew-symmetric cross-product matrix of :math:`\hat{\mathbf{n}}`.

    Parameters
    ----------
    axis_angle : :class:`numpy.ndarray`
        Axis–angle vector of shape ``(3,)``; its magnitude is the rotation
        angle in radians.

    Returns
    -------
    :class:`numpy.ndarray`
        Rotation matrix of shape ``(3, 3)``.  Returns the identity when
        ``axis_angle`` has near-zero magnitude.
    """
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
    r"""Vectorised log-pdf of multivariate Gaussians with per-mean covariances.

    Adapted from Gregory Gundersen's blog post on group multivariate normal
    pdfs (https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/).

    Parameters
    ----------
    xs : :class:`numpy.ndarray`
        Sample matrix of shape ``(n_samples, n_features)``.
    means : :class:`numpy.ndarray`
        Distribution means, shape ``(n_means, n_features)``.
    covs : :class:`numpy.ndarray`
        Per-mean covariance matrices, shape
        ``(n_means, n_features, n_features)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Log-pdf evaluations of shape ``(n_samples, n_means)``.
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
    r"""Numba-accelerated multivariate-Gaussian log-pdf with a shared covariance.

    The covariance is supplied through its eigen-decomposition ``(logdet, Us)``
    (computed once by the caller) so that this hot-path function avoids any
    per-call linear algebra.  See the numpy reference
    :func:`_multiple_logpdfs_vec_input_single_cov_numpy` for the equivalent
    body.  Algorithm based on the Gundersen blog post linked from
    :func:`multiple_logpdfs_vec_input`.

    Parameters
    ----------
    xs : :class:`numpy.ndarray`
        Sample matrix of shape ``(n_samples, n_features)``.
    means : :class:`numpy.ndarray`
        Distribution means, shape ``(n_means, n_features)``.
    logdet : float
        Pre-computed log-determinant of the shared covariance.
    Us : :class:`numpy.ndarray`
        Whitening matrix, shape ``(n_features, n_features)``, equal to
        the eigenvectors scaled by inverse square-root eigenvalues.

    Returns
    -------
    :class:`numpy.ndarray`
        Log-pdf evaluations of shape ``(n_samples, n_means)``.
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
    r"""Numba log-pdf specialisation for a diagonal covariance.

    For a diagonal :math:`\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_i^2)`,
    the Mahalanobis distance reduces to
    :math:`\sum_i (x_i - \mu_i)^2 / \sigma_i^2`.

    Parameters
    ----------
    xs : :class:`numpy.ndarray`
        Sample matrix of shape ``(n_samples, n_features)``.
    means : :class:`numpy.ndarray`
        Distribution means, shape ``(n_means, n_features)``.
    logdet : float
        Pre-computed log-determinant of :math:`\boldsymbol{\Sigma}`,
        i.e. :math:`\sum_i \log \sigma_i^2`.
    sigma_inv_diag : :class:`numpy.ndarray`
        Diagonal of :math:`\boldsymbol{\Sigma}^{-1}`, shape ``(n_features,)``,
        i.e. :math:`1/\sigma_i^2`.

    Returns
    -------
    :class:`numpy.ndarray`
        Log-pdf evaluations of shape ``(n_samples, n_means)``.
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

# --- private cache helpers ---

def _cache_index_file(cache_dir):
    return Path(cache_dir) / 'cache_index.csv'


def _load_index(cache_dir):
    idx = _cache_index_file(cache_dir)
    if not idx.exists():
        return pd.DataFrame(
            columns=['sofa_name', 'file_hash', 'pkl_file', 'timestamp'])
    return pd.read_csv(idx)


def _save_index(cache_dir, df):
    df.to_csv(_cache_index_file(cache_dir), index=False)


def _find_index_row(df, sofa_name, file_hash):
    match = df[(df['sofa_name'] == sofa_name) & (df['file_hash'] == file_hash)]
    return match.iloc[0] if not match.empty else None


def _write_pkl(pkl_path, data):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# --- public cache helpers ---
# Each HRTF gets one pickle: {'target': ..., 'templates': {'SHMAX': ..., ...}}
# The index has one row per HRTF (sofa_name + file_hash), no interpolation column.

def cache_load_target(cache_dir, sofa_file):
    """Load the cached target for a SOFA file, or ``None`` if not found."""
    cache_dir = Path(cache_dir)
    df = _load_index(cache_dir)
    if df.empty:
        return None
    file_hash = _compute_file_hash(sofa_file)
    row = _find_index_row(df, Path(sofa_file).name, file_hash)
    if row is None:
        return None
    pkl_path = cache_dir / row['pkl_file']
    if not pkl_path.exists():
        return None
    try:
        return _read_pkl(pkl_path).get('target')
    except Exception:
        return None


def cache_load_template(cache_dir, sofa_file, interpolation):
    """Load a cached template for a SOFA file and interpolation method, or ``None``."""
    cache_dir = Path(cache_dir)
    df = _load_index(cache_dir)
    if df.empty:
        return None
    file_hash = _compute_file_hash(sofa_file)
    row = _find_index_row(df, Path(sofa_file).name, file_hash)
    if row is None:
        return None
    pkl_path = cache_dir / row['pkl_file']
    if not pkl_path.exists():
        return None
    try:
        return _read_pkl(pkl_path).get('templates', {}).get(interpolation)
    except Exception:
        return None


def cache_save_target(cache_dir, sofa_file, target):
    """Save *target* to the HRTF pickle, creating the cache entry if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = _load_index(cache_dir)
    file_hash = _compute_file_hash(sofa_file)
    sofa_name = Path(sofa_file).name
    row = _find_index_row(df, sofa_name, file_hash)

    if row is not None:
        pkl_path = cache_dir / row['pkl_file']
        try:
            data = _read_pkl(pkl_path)
        except Exception:
            data = {'templates': {}}
        data['target'] = target
        _write_pkl(pkl_path, data)
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_filename = f"{Path(sofa_name).stem}_{file_hash[:8]}_{timestamp}.pkl"
        pkl_path = cache_dir / pkl_filename
        _write_pkl(pkl_path, {'target': target, 'templates': {}})
        # Invalidate stale entries for the same SOFA name with a different hash.
        stale = df[(df['sofa_name'] == sofa_name) & (df['file_hash'] != file_hash)]
        for _, old_row in stale.iterrows():
            old_file = cache_dir / old_row['pkl_file']
            if old_file.exists():
                old_file.unlink()
        df = df[~((df['sofa_name'] == sofa_name) & (df['file_hash'] != file_hash))]
        new_row = pd.DataFrame([{
            'sofa_name': sofa_name, 'file_hash': file_hash,
            'pkl_file': pkl_filename, 'timestamp': timestamp,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        _save_index(cache_dir, df)
    print(f"✓ Target cached: {pkl_path.name}")


def cache_save_template(cache_dir, sofa_file, interpolation, template):
    """Add *template* to the HRTF pickle under the given interpolation key."""
    cache_dir = Path(cache_dir)
    df = _load_index(cache_dir)
    file_hash = _compute_file_hash(sofa_file)
    sofa_name = Path(sofa_file).name
    row = _find_index_row(df, sofa_name, file_hash)

    if row is not None:
        pkl_path = cache_dir / row['pkl_file']
        try:
            data = _read_pkl(pkl_path)
        except Exception:
            data = {'target': None, 'templates': {}}
        data.setdefault('templates', {})[interpolation] = template
        _write_pkl(pkl_path, data)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_filename = f"{Path(sofa_name).stem}_{file_hash[:8]}_{timestamp}.pkl"
        pkl_path = cache_dir / pkl_filename
        _write_pkl(pkl_path, {'target': None, 'templates': {interpolation: template}})
        new_row = pd.DataFrame([{
            'sofa_name': sofa_name, 'file_hash': file_hash,
            'pkl_file': pkl_filename, 'timestamp': timestamp,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        _save_index(cache_dir, df)
    print(f"✓ Template '{interpolation}' cached: {pkl_path.name}")


# -----------------------------------
# VARIOUS
# -----------------------------------

def print_memory_usage(label=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    print(f"[{label}] Memory usage: {mem_gb:.2f} GB")


def compute_features(hrir, coords, fs, spectral_range=[7e2, 18e3],
                     halfwave_rectifier=True):
    r"""Compute ITD, ILD, and monaural spectral cues from binaural HRIRs.

    Implements the feature extraction of Eq. 1 of :footcite:t:`barumerli2023`:

    1. Normalise HRIRs to the frontal direction.
    2. Estimate ITD via :func:`itdestimator` and apply the signed-log
       perceptual warp :math:`\mathrm{sgn}(t)\,(\log(a + b\,|t|) - \log a)/b`
       with :math:`a = 32.5\,\mu\mathrm{s}` and :math:`b = 0.095`.
    3. Compute ILD as the dB ratio of broadband RMS energies.
    4. Filter both ears with an ERB-spaced gammatone bank
       (:func:`gammatone`) and convert per-band amplitude to dB.

    Gammatone filtering uses a Numba-compiled fused kernel
    (``_gammatone_rms_numba``) that processes all bands concurrently
    across CPU cores without storing any intermediate filtered signal.
    Each band is zero-padded by its minimum impulse response length
    (:func:`minimum_ir_length`) rather than a fixed 50 ms, reducing peak
    memory from ~3.4 GB to ~0.4 MB for a 793-direction HRTF at 48 kHz.
    The first call incurs a one-time JIT compilation cost of a few seconds.

    Parameters
    ----------
    hrir : :class:`numpy.ndarray`
        Head-related impulse responses, shape ``(n_dirs, 2, n_samples)``.
    coords : :class:`pyfar.Coordinates`
        Source positions, one per HRIR row.
    fs : int
        Sampling rate in Hz.
    spectral_range : list of float, default=[700.0, 18000.0]
        ``[low_Hz, high_Hz]`` bracket for the gammatone filterbank.
    halfwave_rectifier : bool, default=True
        If ``True``, apply half-wave rectification before computing the
        per-band mean: :math:`\sqrt{\mathrm{mean}(\max(x,0))}`.
        If ``False``, compute the full-wave RMS instead:
        :math:`\sqrt{\mathrm{mean}(x^2)}`.

    Returns
    -------
    itd : :class:`numpy.ndarray`
        Warped ITD, shape ``(n_dirs, 1)`` (dimensionless after warping).
    ild : :class:`numpy.ndarray`
        ILD in dB, shape ``(n_dirs, 1)``.
    spectral_cues : :class:`numpy.ndarray`
        Monaural spectral amplitudes in dB, shape ``(n_dirs, n_freqs, 2)``.
    freqs : :class:`numpy.ndarray`
        Filterbank centre frequencies in Hz, shape ``(n_freqs,)``.
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

    # generate gammatone filterbank
    freqs = erb_space(spectral_range)
    B, A, *_ = gammatone(freqs, fs=fs)

    # per-band minimum IR lengths (replaces fixed 50 ms padding)
    min_lens = minimum_ir_length(freqs, fs)

    # fused filter + rectify + RMS via Numba kernel (no intermediate storage)
    hrirs_c = np.ascontiguousarray(hrirs_temp, dtype=np.float64)
    rms = _gammatone_rms_numba(hrirs_c, B, A, min_lens, halfwave_rectifier)

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
