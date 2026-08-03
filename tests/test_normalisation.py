"""Level-reference tests: what the model does and does not see when the level changes.

The model divides the whole HRIR set by a single reference scalar ``k`` before any
cue is extracted (``utils.compute_features``, ``reference`` argument).  These tests
pin down the consequences:

* a **global** broadband gain on the target (same factor for every direction and
  both ears) must leave every feature and every prediction untouched, because it
  scales ``k`` by the same factor;
* a **single-ear** gain must *not* be absorbed -- that is a real cue change, and it
  is the negative control proving the tests above have any power;
* changing ``reference`` on a single HRTF set shifts the monaural spectra by one
  constant and nothing else;
* the cache tells all of these apart instead of silently serving one for another.
"""
import numpy as np
import pyfar as pf
import pytest
import sofar

from bayesian_listener import BayesianListener, utils

# +2.4 dB: the largest level difference between the six simplehrtf conditions.
GAIN_DB = 2.4
GAIN = 10 ** (GAIN_DB / 20)

# Tolerances. ITD/ILD come out bitwise identical; the spectral cues go through the
# numba fastmath kernel, which reorders floating-point operations, so they agree to
# ~1e-8 dB rather than exactly.
ATOL_INTERAURAL = 1e-12
ATOL_SPECTRAL_DB = 1e-6

# Three well-separated measured directions, enough to catch a shifted argmax
# without paying for all 793.
TARGET_IDX = [100, 260, 400]


def _listener(sofa_path, gain=None, ear=None):
    """Build a listener from the test SOFA file, optionally scaling its HRIRs.

    Parameters
    ----------
    sofa_path : str
        Path to the SOFA file.
    gain : float or None
        Linear factor applied to the impulse responses.  ``None`` leaves them alone.
    ear : int or None
        Ear index to scale (0 = left, 1 = right).  ``None`` scales both, i.e. a
        global broadband gain.
    """
    sofa_data = sofar.read_sofa(sofa_path, verbose=False)
    listener = BayesianListener(sofa_data)
    if gain is not None:
        listener.hrir = listener.hrir.copy()
        if ear is None:
            listener.hrir *= gain
        else:
            listener.hrir[:, ear, :] *= gain
    return listener


@pytest.fixture(scope='module')
def unscaled(sofa_path):
    """Listener with target and SHMAX template from the unmodified HRTF set."""
    listener = _listener(sofa_path)
    listener.compute_target(use_cache=False)
    listener.compute_template(use_cache=False)
    return listener


@pytest.fixture(scope='module')
def scaled(sofa_path):
    """Listener with target from HRIRs scaled by a global +2.4 dB."""
    listener = _listener(sofa_path, gain=GAIN)
    listener.compute_target(use_cache=False)
    return listener


# -----------------------------------------------------------------------
# Global gain: the model must be blind to it
# -----------------------------------------------------------------------

def test_global_gain_leaves_features_unchanged(unscaled, scaled):
    """A global broadband gain is absorbed by the level reference."""
    np.testing.assert_allclose(
        scaled.target.itd, unscaled.target.itd, atol=ATOL_INTERAURAL)
    np.testing.assert_allclose(
        scaled.target.ild, unscaled.target.ild, atol=ATOL_INTERAURAL)
    np.testing.assert_allclose(
        scaled.target.spectral_cues, unscaled.target.spectral_cues,
        atol=ATOL_SPECTRAL_DB)
    np.testing.assert_allclose(
        scaled.target.features, unscaled.target.features, atol=ATOL_SPECTRAL_DB)


def test_global_gain_leaves_predictions_unchanged(sofa_path, unscaled, scaled):
    """A scaled target against an unscaled template gives the same MAP directions."""
    baseline = _listener(sofa_path)
    baseline.template = unscaled.template
    baseline.target = unscaled.target[TARGET_IDX]
    posterior_ref = baseline.infer(repetitions=5, seed=42)

    mismatched = _listener(sofa_path)
    mismatched.template = unscaled.template          # from the unscaled set
    mismatched.target = scaled.target[TARGET_IDX]    # from the scaled set
    posterior_scaled = mismatched.infer(repetitions=5, seed=42)

    assert np.array_equal(posterior_ref, posterior_scaled)


def test_single_ear_gain_changes_features(sofa_path, unscaled):
    """Negative control: a one-eared gain is a real cue change and must survive.

    Without this, the invariance tests above would still pass on a feature
    extractor that had been made blind to level differences altogether.
    """
    one_eared = _listener(sofa_path, gain=GAIN, ear=0)
    one_eared.compute_target(use_cache=False)

    assert not np.allclose(one_eared.target.ild, unscaled.target.ild,
                           atol=ATOL_SPECTRAL_DB)
    assert not np.allclose(one_eared.target.spectral_cues,
                           unscaled.target.spectral_cues, atol=ATOL_SPECTRAL_DB)


# -----------------------------------------------------------------------
# reference=: inert within one HRTF set
# -----------------------------------------------------------------------

@pytest.fixture(scope='module')
def reference_targets(sofa_path):
    """Targets from one HRTF set computed with each ``reference`` option."""
    targets = {}
    for reference in ('frontal', 'global', 'none'):
        listener = _listener(sofa_path)
        listener.compute_target(use_cache=False, reference=reference)
        targets[reference] = listener.target
    return targets


def test_reference_leaves_interaural_cues_unchanged(reference_targets):
    """ITD is a timing readout and ILD a level ratio, so neither sees ``k``."""
    frontal = reference_targets['frontal']
    for reference in ('global', 'none'):
        other = reference_targets[reference]
        np.testing.assert_allclose(other.itd, frontal.itd, atol=ATOL_INTERAURAL)
        np.testing.assert_allclose(other.ild, frontal.ild, atol=ATOL_INTERAURAL)


def test_reference_shifts_spectra_by_one_constant(reference_targets):
    """The spectra move by a single constant, identical over directions and bands.

    The spread is bounded by ``ATOL_SPECTRAL_DB`` rather than the 1e-9 suggested in
    ``notes/NORMALISATION_BRIEF.md``: the fastmath kernel reorders operations, which
    leaves ~3e-8 dB of scatter on an otherwise exactly constant offset.
    """
    frontal = reference_targets['frontal']
    for reference in ('global', 'none'):
        offset = reference_targets[reference].spectral_cues - frontal.spectral_cues
        assert np.ptp(offset) < ATOL_SPECTRAL_DB, (
            f"reference={reference!r} shifted the spectra by a non-constant "
            f"amount (spread {np.ptp(offset):.3e} dB)")


def test_reference_offset_matches_rectifier_exponent(sofa_path):
    """The size of that constant depends on the rectifier, and is *not* 20*log10(k).

    With ``halfwave_rectifier=True`` (the default) the per-band value is
    ``sqrt(mean(max(x, 0)))``, which scales as ``sqrt(k)``, so dividing the HRIRs by
    ``k`` shifts the dB cues by ``10*log10(k)``.  Only the full-wave RMS
    (``halfwave_rectifier=False``) gives the ``20*log10(k)`` stated in
    ``notes/NORMALISATION_BRIEF.md``.
    """
    listener = _listener(sofa_path)
    idx, _ = listener.coords.find_nearest(pf.Coordinates.from_cartesian(1, 0, 0))
    k = np.max(np.abs(listener.hrir[idx]))

    for halfwave_rectifier, factor in ((True, 10.0), (False, 20.0)):
        _, _, cues_frontal, _ = utils.compute_features(
            listener.hrir, listener.coords, listener.fs,
            halfwave_rectifier=halfwave_rectifier, reference='frontal')
        _, _, cues_none, _ = utils.compute_features(
            listener.hrir, listener.coords, listener.fs,
            halfwave_rectifier=halfwave_rectifier, reference='none')

        offset = np.mean(cues_none - cues_frontal)
        np.testing.assert_allclose(offset, factor * np.log10(k), atol=1e-6)


def test_reference_invalid_raises(sofa_path):
    """An unknown reference is rejected rather than silently defaulting."""
    listener = _listener(sofa_path)
    with pytest.raises(ValueError, match="reference must be"):
        listener.compute_target(use_cache=False, reference='rms')


# -----------------------------------------------------------------------
# Cache: parameters that change the features must not collide
# -----------------------------------------------------------------------

@pytest.mark.parametrize(('first', 'second'), [
    ({'reference': 'frontal'}, {'reference': 'global'}),
    ({'halfwave_rectifier': True}, {'halfwave_rectifier': False}),
    ({'spectral_range': [7e2, 18e3]}, {'spectral_range': [5e2, 16e3]}),
])
def test_cache_distinguishes_feature_parameters(sofa_path, tmp_path, first, second):
    """Computing with A, then B, then A again must return A's features, not B's."""
    cache_dir = tmp_path / 'cache'

    listener_a = BayesianListener(sofa_path)
    listener_a.compute_target(cache_dir=cache_dir, **first)
    features_a = listener_a.target.features.copy()

    listener_b = BayesianListener(sofa_path)
    listener_b.compute_target(cache_dir=cache_dir, **second)
    features_b = listener_b.target.features

    # The two parameter sets must genuinely produce different features, otherwise
    # this test cannot detect a collision.
    assert (features_a.shape != features_b.shape
            or not np.allclose(features_a, features_b))

    listener_a_again = BayesianListener(sofa_path)
    listener_a_again.compute_target(cache_dir=cache_dir, **first)
    np.testing.assert_array_equal(listener_a_again.target.features, features_a)


def test_cache_roundtrip_uses_feature_key(sofa_path, tmp_path):
    """The cached target and template are both retrievable under the feature key."""
    cache_dir = tmp_path / 'cache'

    listener = BayesianListener(sofa_path)
    listener.compute_target(cache_dir=cache_dir, reference='global')
    listener.compute_template(cache_dir=cache_dir)

    key = utils.feature_cache_key('Barumerli2023', [7e2, 18e3], True, 'global')
    assert utils.cache_load_target(cache_dir, sofa_path, key) is not None
    assert utils.cache_load_template(cache_dir, sofa_path, key, 'SHMAX') is not None

    frontal_key = utils.feature_cache_key(
        'Barumerli2023', [7e2, 18e3], True, 'frontal')
    assert utils.cache_load_target(cache_dir, sofa_path, frontal_key) is None
