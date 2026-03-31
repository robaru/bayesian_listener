"""Auditory representation classes for the Bayesian listener model."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pyfar as pf


class AuditoryRepresentation(ABC):
    """Abstract base for all auditory representations.

    Each subclass encapsulates a specific set of spatial cues and the
    corresponding covariance structure used during Bayesian inference.
    The feature matrix is pre-computed once at construction and stored
    as ``self.features``.

    Attributes
    ----------
    convention : str
        Short label identifying the representation, e.g. ``'barumerli2025'``.
    coords : pyfar.Coordinates
        Source positions corresponding to the rows of ``features``.
    freqs : ndarray
        Centre frequencies of the gammatone filterbank, shape ``(n_freqs,)``.
    features : ndarray
        Concatenated feature matrix, shape ``(n_dirs, n_features)``.
        Computed in ``__post_init__`` of each concrete subclass.
    """

    convention: str
    coords: pf.Coordinates
    freqs: np.ndarray
    features: np.ndarray

    @abstractmethod
    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        """Return the covariance matrix for the Gaussian likelihood.

        Parameters
        ----------
        parameters : dict
            Model noise parameters (``sigma_itd``, ``sigma_ild``,
            ``sigma_spectral``, …).

        Returns
        -------
        ndarray
            Square covariance matrix of shape ``(n_features, n_features)``.
        """
        ...


@dataclass
class Barumerli2025(AuditoryRepresentation):
    """ITD + ILD + spectral envelope (Barumerli et al. 2025).

    Attributes
    ----------
    convention : str
        Fixed to ``'barumerli2025'``.
    coords : pyfar.Coordinates
        Source positions, shape ``(n_dirs,)``.
    itd : ndarray
        Interaural time differences, shape ``(n_dirs, 1)``.
    ild : ndarray
        Interaural level differences, shape ``(n_dirs, 1)``.
    spectral_cues : ndarray
        Log-magnitude spectra for left and right ears,
        shape ``(n_dirs, n_freqs, 2)``.
    freqs : ndarray
        Filterbank centre frequencies, shape ``(n_freqs,)``.
    features : ndarray
        Pre-computed concatenation ``[itd, ild, spectral_L, spectral_R]``,
        shape ``(n_dirs, 2 + 2*n_freqs)``.  Set by ``__post_init__``.
    """

    convention: str = 'barumerli2025'
    coords: pf.Coordinates = None
    itd: np.ndarray = None
    ild: np.ndarray = None
    spectral_cues: np.ndarray = None
    freqs: np.ndarray = None

    def __post_init__(self):
        self.features = np.hstack([self.itd,
                                   self.ild,
                                   self.spectral_cues[:, :, 0],
                                   self.spectral_cues[:, :, 1]])

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        return Barumerli2025(
            coords=self.coords[idx],
            itd=self.itd[idx],
            ild=self.ild[idx],
            spectral_cues=self.spectral_cues[idx],
            freqs=self.freqs,
        )

    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        """Diagonal covariance from ``sigma_itd``, ``sigma_ild``, ``sigma_spectral``."""
        return np.diag(np.hstack([
            parameters['sigma_itd']**2,
            parameters['sigma_ild']**2,
            np.repeat(parameters['sigma_spectral']**2, self.freqs.shape[0] * 2),
        ]))


@dataclass
class Barumerli2023pge(AuditoryRepresentation):
    """ITD + ILD + spectral gradient — stub (issue #22).

    Not yet implemented.  Instantiating this class raises
    ``NotImplementedError``.
    """

    convention: str = 'barumerli2023pge'
    coords: pf.Coordinates = None
    itd: np.ndarray = None
    ild: np.ndarray = None
    spectral_gradient: np.ndarray = None
    freqs: np.ndarray = None

    def __post_init__(self):
        raise NotImplementedError('barumerli2023pge is not yet implemented.')

    def sigma_matrix(self, parameters: dict) -> np.ndarray:
        raise NotImplementedError('barumerli2023pge is not yet implemented.')


CONVENTIONS = {
    'barumerli2025': Barumerli2025,
    'barumerli2023pge': Barumerli2023pge,
}
