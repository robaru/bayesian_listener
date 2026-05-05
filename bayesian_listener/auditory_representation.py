"""Auditory representation classes for the Bayesian listener model."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pyfar as pf


class _AuditoryRepresentation(ABC):
    r"""Abstract base for all auditory representations.

    Each subclass encapsulates a specific set of spatial cues
    (e.g. ITD, ILD, monaural spectra) and the diagonal covariance structure
    :math:`\\boldsymbol{\\Sigma}` used in the Gaussian sensory likelihood
    (Eq. 2 of [barumerli2023]_).  The feature matrix is concatenated once
    at construction and stored as :attr:`features`, so that
    :meth:`~bayesian_listener.BayesianListener.infer` can compare target and
    template features by simple matrix algebra.

    Attributes
    ----------
    convention : str
        Short label identifying the representation, e.g. ``'Barumerli2023'``.
    coords : :class:`pyfar.Coordinates`
        Source positions corresponding to the rows of :attr:`features`.
    freqs : :class:`numpy.ndarray`
        Centre frequencies of the gammatone filterbank, shape ``(n_freqs,)``.
    features : :class:`numpy.ndarray`
        Concatenated feature matrix, shape ``(n_dirs, n_features)``.
        Computed in ``__post_init__`` of each concrete subclass.

    See Also
    --------
    Barumerli2023 : Concrete implementation with ITD + ILD + spectral envelope.
    bayesian_listener.BayesianListener : Consumer of this representation.
    """

    convention: str
    coords: pf.Coordinates
    freqs: np.ndarray
    features: np.ndarray

    @abstractmethod
    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        r"""Return the diagonal covariance :math:`\boldsymbol{\Sigma}` for the sensory likelihood.

        Parameters
        ----------
        parameters : dict
            Model noise parameters with keys ``sigma_itd`` (dimensionless),
            ``sigma_ild`` (dB), ``sigma_spectral`` (dB), and any subclass-
            specific entries.

        Returns
        -------
        :class:`numpy.ndarray`
            Square covariance of shape ``(n_features, n_features)``.
        """
        ...


@dataclass
class Barumerli2023(_AuditoryRepresentation):
    r"""ITD + ILD + spectral-amplitude representation of [barumerli2026]_.

    Builds the feature vector
    :math:`\mathbf{t} = [x_{\mathrm{itd}}, x_{\mathrm{ild}},
    \mathbf{x}_{L,\mathrm{mon}}, \mathbf{x}_{R,\mathrm{mon}}]`
    (Eq. 1 of [barumerli2023]_) with monaural cues as gammatone-bank
    log-amplitudes.  ``n_features = 2 + 2 * n_freqs`` where ``n_freqs`` is
    typically 28 (ERB-spaced centre frequencies between 700 Hz and 18 kHz).

    Attributes
    ----------
    convention : str
        Fixed to ``'Barumerli2023'``.
    coords : :class:`pyfar.Coordinates`
        Source positions, one per row.
    itd : :class:`numpy.ndarray`
        Warped interaural time differences (signed log of ITD in seconds,
        as produced by :func:`~bayesian_listener.utils.compute_features`),
        shape ``(n_dirs, 1)``.
    ild : :class:`numpy.ndarray`
        Interaural level differences in dB, shape ``(n_dirs, 1)``.
    spectral_cues : :class:`numpy.ndarray`
        Monaural log-amplitude spectra for the left and right ears in dB,
        shape ``(n_dirs, n_freqs, 2)``.
    freqs : :class:`numpy.ndarray`
        Filterbank centre frequencies in Hz, shape ``(n_freqs,)``.
    features : :class:`numpy.ndarray`
        Pre-computed concatenation
        ``[itd, ild, spectral_L, spectral_R]`` of shape
        ``(n_dirs, 2 + 2*n_freqs)``.  Built in ``__post_init__``.

    Examples
    --------
    Construct from raw HRIRs via the helper in ``bayesian_listener.utils``:

    >>> from bayesian_listener.utils import compute_features                # doctest: +SKIP
    >>> itd, ild, spec, freqs = compute_features(hrir, coords, fs)          # doctest: +SKIP
    >>> repr_ = Barumerli2023(coords=coords, itd=itd, ild=ild,
    ...                       spectral_cues=spec, freqs=freqs)              # doctest: +SKIP
    >>> repr_.features.shape[1] == 2 + 2 * freqs.size                       # doctest: +SKIP
    True
    """

    convention: str = 'Barumerli2023'
    coords: pf.Coordinates = None
    itd: np.ndarray = None
    ild: np.ndarray = None
    spectral_cues: np.ndarray = None
    freqs: np.ndarray = None

    def __post_init__(self):
        """Concatenate ITD, ILD, and stacked spectral cues into :attr:`features`."""
        self.features = np.hstack([self.itd,
                                   self.ild,
                                   self.spectral_cues[:, :, 0],
                                   self.spectral_cues[:, :, 1]])

    def __getitem__(self, idx):
        """Return a new :class:`Barumerli2023` with the requested rows.

        Used to subset a template to behavioural target directions before
        fitting (e.g. inside
        :func:`~bayesian_listener.fitting.fit_listener_partial`).
        """
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        return Barumerli2023(
            coords=self.coords[idx],
            itd=self.itd[idx],
            ild=self.ild[idx],
            spectral_cues=self.spectral_cues[idx],
            freqs=self.freqs,
        )

    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        r"""Diagonal sensory covariance (Eq. 2 of [barumerli2023]_).

        Builds
        :math:`\boldsymbol{\Sigma} = \mathrm{diag}(
        \sigma_{\mathrm{itd}}^2,\, \sigma_{\mathrm{ild}}^2,\,
        \sigma_{\mathrm{mon}}^2 \mathbf{I}_{2\,n_{\mathrm{freqs}}})`.

        Parameters
        ----------
        parameters : dict
            Must contain keys ``sigma_itd`` (dimensionless), ``sigma_ild``
            (dB), ``sigma_spectral`` (dB).

        Returns
        -------
        :class:`numpy.ndarray`
            Diagonal covariance of shape ``(n_features, n_features)``.

        Examples
        --------
        >>> Sigma = repr_.sigma_matrix(                                     # doctest: +SKIP
        ...     {'sigma_itd': 0.569, 'sigma_ild': 1.0, 'sigma_spectral': 10.4}
        ... )                                                                # doctest: +SKIP
        >>> Sigma.shape == (repr_.features.shape[1],) * 2                   # doctest: +SKIP
        True
        """
        return np.diag(np.hstack([
            parameters['sigma_itd']**2,
            parameters['sigma_ild']**2,
            np.repeat(parameters['sigma_spectral']**2, self.freqs.shape[0] * 2),
        ]))


@dataclass
class Barumerli2023pge(_AuditoryRepresentation):
    """ITD + ILD + spectral-gradient representation — stub.

    .. warning::

       This convention is registered for future use (issue #22) but is not
       implemented.  Instantiating it raises :class:`NotImplementedError`,
       and :meth:`sigma_matrix` does the same.
    """

    convention: str = 'barumerli2023pge'
    coords: pf.Coordinates = None
    itd: np.ndarray = None
    ild: np.ndarray = None
    spectral_gradient: np.ndarray = None
    freqs: np.ndarray = None

    def __post_init__(self):
        """Not implemented."""
        raise NotImplementedError('barumerli2023pge is not yet implemented.')

    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError('barumerli2023pge is not yet implemented.')


CONVENTIONS = {
    'Barumerli2023': Barumerli2023,
    'barumerli2023pge': Barumerli2023pge,
}
