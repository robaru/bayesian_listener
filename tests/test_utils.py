"""Tests for :mod:`bayesian_listener.utils` spherical helpers.

The von Mises-Fisher tests below guard a regression in which the RNG was
constructed *inside* ``randvmf``, which ``scatter_von_mises`` then called once
per direction with the same seed.  Every direction therefore drew the identical
(U, psi) and the "scatter" collapsed to a single fixed rotation with zero
trial-to-trial variability (angular deviation std == 0.0 exactly).
"""
import numpy as np
import pytest

from bayesian_listener import utils
from bayesian_listener.utils import (
    randvmf, rodriguesrotation, scatter_von_mises)


def sigma_to_kappa_deg(sigma_deg):
    """Concentration matching a circular SD of ``sigma_deg`` degrees."""
    return 1.0 / np.deg2rad(sigma_deg) ** 2


def unit_dirs(n, seed=0):
    """``n`` random unit vectors."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def deviation_deg(a, b):
    """Great-circle angle between rows of ``a`` and ``b``, in degrees."""
    cos = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


# ---------------------------------------------------------------------------
# regression: the scatter must actually vary across directions
# ---------------------------------------------------------------------------

def test_scatter_varies_across_directions():
    """A seeded call must not apply one constant rotation to every direction.

    Under the bug every row received the identical (U, psi), so the deviation
    angle was constant and its std was exactly 0.0.
    """
    sigma = 8.5
    v = unit_dirs(2000)
    out = scatter_von_mises(v, sigma_to_kappa_deg(sigma), seed=42)

    ang = deviation_deg(v, out)
    assert ang.std() > 1.0, "seeded scatter collapsed to a constant rotation"
    assert ang.min() < ang.max()


@pytest.mark.parametrize("sigma", [8.5, 14.0])
def test_scatter_matches_rayleigh_theory(sigma):
    """vMF(kappa = 1/sigma^2) deviation is Rayleigh with scale ``sigma``.

    Hence mean = sigma*sqrt(pi/2), rms = sigma*sqrt(2), mean/rms ~ 0.886.
    """
    v = unit_dirs(20000, seed=1)
    out = scatter_von_mises(v, sigma_to_kappa_deg(sigma), seed=7)
    ang = deviation_deg(v, out)

    rms = np.sqrt(np.mean(ang ** 2))
    assert np.isclose(ang.mean(), sigma * np.sqrt(np.pi / 2), atol=0.5)
    assert np.isclose(rms, sigma * np.sqrt(2), atol=0.5)
    assert np.isclose(ang.mean() / rms, 0.886, atol=0.02)


def test_scatter_agrees_with_randvmf_distribution():
    """Vectorised and scalar samplers agree distributionally.

    Element-wise equality is not a meaningful invariant (the draw ordering
    differs), so compare summary statistics instead.
    """
    sigma = 10.0
    kappa = sigma_to_kappa_deg(sigma)
    v = unit_dirs(5000, seed=2)

    vec = deviation_deg(v, scatter_von_mises(v, kappa, seed=3))

    rng = np.random.default_rng(4)
    scalar = np.array([randvmf(kappa, d, seed=int(s))
                       for d, s in zip(v, rng.integers(0, 2**31, len(v)))])
    scalar = deviation_deg(v, scalar)

    assert np.isclose(vec.mean(), scalar.mean(), atol=0.5)
    assert np.isclose(vec.std(), scalar.std(), atol=0.5)


# ---------------------------------------------------------------------------
# seeding semantics
# ---------------------------------------------------------------------------

def test_seeded_calls_are_reproducible():
    """Same seed -> identical output; different seed -> different output."""
    kappa = sigma_to_kappa_deg(8.5)
    v = unit_dirs(500, seed=5)

    assert np.array_equal(scatter_von_mises(v, kappa, seed=42),
                          scatter_von_mises(v, kappa, seed=42))
    assert not np.array_equal(scatter_von_mises(v, kappa, seed=42),
                              scatter_von_mises(v, kappa, seed=43))


def test_result_is_independent_of_chunk_size(monkeypatch):
    """``_VMF_CHUNK`` is a pure memory knob and must not change the output.

    U and psi are drawn for all rows up front, so chunking only splits the
    deterministic rotation.
    """
    kappa = sigma_to_kappa_deg(8.5)
    v = unit_dirs(3000, seed=6)

    monkeypatch.setattr(utils, "_VMF_CHUNK", 10 ** 9)   # single block
    whole = scatter_von_mises(v, kappa, seed=42)

    monkeypatch.setattr(utils, "_VMF_CHUNK", 128)       # many blocks
    chunked = scatter_von_mises(v, kappa, seed=42)

    assert np.array_equal(whole, chunked)


# ---------------------------------------------------------------------------
# shape contract, poles, norms
# ---------------------------------------------------------------------------

def test_shape_contract():
    """(3,) in -> (3,) out; (n, 3) in -> (n, 3) out."""
    kappa = sigma_to_kappa_deg(8.5)

    single = scatter_von_mises(np.array([1.0, 0.0, 0.0]), kappa, seed=0)
    assert single.shape == (3,)

    many = scatter_von_mises(unit_dirs(17, seed=8), kappa, seed=0)
    assert many.shape == (17, 3)


@pytest.mark.parametrize("pole", [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
def test_poles_are_finite(pole):
    """The rotation axis degenerates at +-z; output must stay finite.

    The south pole previously produced NaN (division by a zero-norm axis).
    """
    kappa = sigma_to_kappa_deg(8.5)
    mu = np.tile(np.array(pole), (50, 1))

    out = scatter_von_mises(mu, kappa, seed=0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)

    single = randvmf(kappa, np.array(pole), seed=0)
    assert np.all(np.isfinite(single))
    np.testing.assert_allclose(np.linalg.norm(single), 1.0, atol=1e-12)


def test_outputs_are_unit_norm():
    kappa = sigma_to_kappa_deg(8.5)
    out = scatter_von_mises(unit_dirs(1000, seed=9), kappa, seed=1)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)


def test_scatter_rejects_bad_input():
    v = unit_dirs(10)
    with pytest.raises(ValueError):
        scatter_von_mises(v, -1.0, seed=0)
    with pytest.raises(ValueError):
        scatter_von_mises(np.zeros((10, 2)), 10.0, seed=0)


# ---------------------------------------------------------------------------
# rotation equivalence
# ---------------------------------------------------------------------------

def test_vector_rotation_matches_matrix_form():
    """``_vmf_rotate_to`` must equal the explicit Rodrigues-matrix rotation.

    Guards against a transpose/handedness slip: ``rodriguesrotation`` returns
    ``M.T``, so the scalar ``y @ Rg`` corresponds to ``M @ y``.
    """
    mu = unit_dirs(200, seed=10)
    y = unit_dirs(200, seed=11)

    vec = utils._vmf_rotate_to(mu, y)

    ref = np.empty_like(y)
    Np = np.array([0.0, 0.0, 1.0])
    for i in range(mu.shape[0]):
        axis = np.cross(Np, mu[i])
        nrm = np.linalg.norm(axis)
        if nrm > np.finfo(float).eps:
            theta = np.arccos(np.clip(mu[i, 2], -1.0, 1.0))
            ref[i] = y[i] @ rodriguesrotation(axis / nrm * theta)
        else:
            ref[i] = y[i] if mu[i, 2] > 0 else y[i] * np.array([1., -1., -1.])

    np.testing.assert_allclose(vec, ref, atol=1e-12)
