"""BayesianListener module: core auditory model for sound localisation."""
import sofar
import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from bayesian_listener import utils
from bayesian_listener import resample
from bayesian_listener.auditory_representation import (
    AuditoryRepresentation, Barumerli2025, CONVENTIONS)
from pathlib import Path

class BayesianListener:
    """Bayesian model of human sound localisation using HRTF-derived cues."""

    def __init__(self, sofa,
                 sigma_itd=0.569,
                 sigma_ild=1.0,
                 sigma_spectral=10.4,
                 sigma_prior=69.0,
                 kappa_motor=23.31):
        """Initialize listener from SOFA file or in-memory Sofa object.

        Parameters
        ----------
        sofa : str or sofar.Sofa
            Path to a SOFA file or a pre-loaded ``sofar.Sofa`` object.
            When a file path is given, it is stored as ``self.sofa_file``
            and used as a cache key by ``prepare_features``.
            When a ``sofar.Sofa`` object is given, ``self.sofa_file`` is
            ``None`` and caching is disabled, because ``sofar.Sofa`` does
            not store the original file path.

        Attributes
        ----------
        sofa_file : str or None
            Path to the SOFA file, or ``None`` if a Sofa object was passed.
        hrir : ndarray
            Head-related impulse responses, shape (n_directions, 2, n_samples).
        fs : int
            Sampling rate in Hz.
        coords : pyfar.Coordinates
            Source positions in spherical top-elevation convention (degrees).
        parameters : dict
            Noise and prior parameters for the Bayesian model.
        """
        # handle sofa input
        if isinstance(sofa, str):
            self.sofa_file = sofa
            sofa_data = sofar.read_sofa(sofa, verbose=False)
        elif isinstance(sofa, sofar.Sofa):
            self.sofa_file = None
            sofa_data = sofa
        else:
            raise ValueError('sofa must be a string containing the path to a '
                             'sofa file or a sofar.Sofa object')

        self.hrir = sofa_data.Data_IR
        self.fs = int(sofa_data.Data_SamplingRate)
        sp = sofa_data.SourcePosition
        self.coords = pf.Coordinates(sp[:, 0], sp[:, 1], sp[:, 2],
                                     domain='sph', convention='top_elev',
                                     unit='deg')

        # noise and prior parameters (group average, Barumerli et al. 2025)
        self.parameters = {
            'sigma_itd':      sigma_itd,
            'sigma_ild':      sigma_ild,
            'sigma_spectral': sigma_spectral,
            'sigma_prior':    sigma_prior,
            'kappa_motor':    kappa_motor,
        }

        self._target = None
        self._template = None

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, dict):
            raise ValueError("Parameters must be a dictionary.")
        # check if all parameters are present
        for key in [
            'sigma_itd',
            'sigma_ild',
            'sigma_spectral',
            'sigma_prior',
            'kappa_motor',
            ]:
            if key not in value:
                raise ValueError(f"Missing parameter: {key}")
        self._parameters = value

    @property
    def sigma_itd(self):
        return self.parameters['sigma_itd']

    @sigma_itd.setter
    def sigma_itd(self, v):
        self.parameters['sigma_itd'] = v

    @property
    def sigma_ild(self):
        return self.parameters['sigma_ild']

    @sigma_ild.setter
    def sigma_ild(self, v):
        self.parameters['sigma_ild'] = v

    @property
    def sigma_spectral(self):
        return self.parameters['sigma_spectral']

    @sigma_spectral.setter
    def sigma_spectral(self, v):
        self.parameters['sigma_spectral'] = v

    @property
    def sigma_prior(self):
        return self.parameters['sigma_prior']

    @sigma_prior.setter
    def sigma_prior(self, v):
        self.parameters['sigma_prior'] = v

    @property
    def kappa_motor(self):
        return self.parameters['kappa_motor']

    @kappa_motor.setter
    def kappa_motor(self, v):
        self.parameters['kappa_motor'] = v

    @property
    def sigma_motor(self):
        """Motor noise in degrees (converted from ``kappa_motor``)."""
        from bayesian_listener.fitting import kappa_to_sigma
        return kappa_to_sigma(self.parameters['kappa_motor'])

    @sigma_motor.setter
    def sigma_motor(self, value):
        from bayesian_listener.fitting import sigma_to_kappa
        self.parameters['kappa_motor'] = sigma_to_kappa(value)

    @property
    def target(self):
        """AuditoryRepresentation or None — what the listener is hearing."""
        return self._target

    @target.setter
    def target(self, value):
        if value is not None and not isinstance(value, AuditoryRepresentation):
            raise TypeError('target must be an AuditoryRepresentation.')
        self._target = value

    @property
    def template(self):
        """AuditoryRepresentation or None — the listener's internal model."""
        return self._template

    @template.setter
    def template(self, value):
        if value is not None and not isinstance(value, AuditoryRepresentation):
            raise TypeError('template must be an AuditoryRepresentation.')
        self._template = value

    def _interpolate(self, ar, interpolation='SHMAX', interpolation_grid=None):
        """Resample an AuditoryRepresentation onto a uniform grid.

        Parameters
        ----------
        ar : AuditoryRepresentation
            Source representation to interpolate.
        interpolation : {'SH', 'SHmax', 'barycentric', 'barumerli2023'}, default='SHMAX'
            Interpolation method:
            - 'SH': Spherical harmonics interpolation with SH truncation.
            - 'SHMAX': Spherical harmonics with high SH order.
            - 'barycentric': Barycentric interpolation on triangulated mesh.
            - 'barumerli2023': Method from Barumerli et al. (2023).
        interpolation_grid : pyfar.Coordinates or None
            Target grid; ``None`` uses the method's default uniform grid.

        Returns
        -------
        AuditoryRepresentation
            Same subclass as ``ar``, resampled onto the uniform grid.
        """
        cues_list = [
            ar.itd,
            ar.ild,
            ar.spectral_cues[:, :, 0],
            ar.spectral_cues[:, :, 1],
        ]
        resampled_cues, coords_new = resample.resample(
            cues_list, ar.coords, interpolation_grid, method=interpolation)

        return type(ar)(
            coords=coords_new,
            itd=resampled_cues[0],
            ild=resampled_cues[1],
            spectral_cues=np.stack([resampled_cues[2],
                                    resampled_cues[3]], axis=-1),
            freqs=ar.freqs,
        )

    def compute_target(self, convention='barumerli2025', spectral_range=None):
        """Compute raw auditory representation from ``self.hrir``.

        Sets ``self.target``.  No interpolation is performed.

        Parameters
        ----------
        convention : str, default='barumerli2025'
        spectral_range : list of float or None, default=[700, 18000]
        """
        if spectral_range is None:
            spectral_range = [7e2, 18e3]
        if convention not in CONVENTIONS:
            raise ValueError(
                f"Unknown convention '{convention}'. "
                f"Available: {list(CONVENTIONS)}")
        itd, ild, spectral_cues, freqs = utils.compute_features(
            self.hrir, self.coords, self.fs, spectral_range)
        self.target = CONVENTIONS[convention](
            coords=self.coords,
            itd=itd,
            ild=ild,
            spectral_cues=spectral_cues,
            freqs=freqs,
        )

    def compute_template(self, interpolation='SHMAX', interpolation_grid=None):
        """Interpolate ``self.target`` onto a uniform grid.

        Sets ``self.template``.  Must call ``compute_target()`` first.

        Parameters
        ----------
        interpolation : str, default='SHMAX'
        interpolation_grid : pyfar.Coordinates or None
        """
        if self.target is None:
            raise ValueError(
                'Call compute_target() before compute_template().')
        self.template = self._interpolate(
            self.target, interpolation, interpolation_grid)

    def prepare_features(self,
                         spectral_range=None,
                         interpolation='SHMAX',
                         interpolation_grid=None,
                         use_cache=True,
                         force_recompute=False,
                         cache_dir=None,
                         compute_template=True):
        """Compute features and optionally the template, with caching.

        Calls :meth:`compute_target` then :meth:`compute_template`.
        Set ``compute_template=False`` to skip interpolation (non-individual
        localisation, where only target features are needed).

        Parameters
        ----------
        spectral_range : list of float or None, default=[700, 18000]
        interpolation : str, default='SHMAX'
        interpolation_grid : pyfar.Coordinates or None
        use_cache : bool, default=True
        force_recompute : bool, default=False
        cache_dir : str or Path or None
        compute_template : bool, default=True
        """
        if spectral_range is None:
            spectral_range = [7e2, 18e3]
        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'preprocessed'
        else:
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

        if use_cache and compute_template and not force_recompute \
                and self.sofa_file is not None:
            cached = utils.load_from_cache(
                cache_dir, self.sofa_file,
                ['target', 'template'], interpolation)
            if cached is not None:
                self.target = cached['target']
                self.template = cached['template']
                return
            print('  Cache not found or invalid. Recomputing...')

        print('→ Computing features...')
        self.compute_target(spectral_range=spectral_range)
        if compute_template:
            self.compute_template(interpolation=interpolation,
                                  interpolation_grid=interpolation_grid)
        print('✓ Feature preparation complete')

        if compute_template and self.sofa_file is not None:
            utils.save_to_cache(
                cache_dir, self.sofa_file,
                {'target': self.target, 'template': self.template},
                interpolation)


    def infer(self,
              repetitions=50,
              prior='horizontal',
              store_posterior=False,
              seed=None):
        """Perform Bayesian inference to estimate sound source direction.

        Parameters
        ----------
        repetitions : int, default=50
        prior : {'uniform', 'horizontal'} or ndarray, default='horizontal'
        store_posterior : bool, default=False
        seed : int or None

        Returns
        -------
        ndarray
            MAP indices ``(targets × repetitions)`` or log-posteriors
            ``(targets × repetitions × templates)`` if ``store_posterior=True``.
        """
        if self.target is None:
            raise ValueError(
                'Target not set. Call compute_target() or set self.target.')
        if self.template is None:
            raise ValueError(
                'Template not set. Call compute_template() or set self.template.')
        if type(self.target) is not type(self.template):
            raise ValueError(
                f'Convention mismatch: '
                f'target={self.target.convention!r}, '
                f'template={self.template.convention!r}.')

        rng = np.random.default_rng(seed)

        target_feat  = self.target.features
        target_num   = target_feat.shape[0]
        template_feat = self.template.features

        sigma = self.target.sigma_matrix(self.parameters)

        vals, vecs = np.linalg.eigh(sigma)
        logdet = np.sum(np.log(vals))
        Us = vecs * np.sqrt(1. / vals)[:, None]

        sigmas = self.parameters
        if isinstance(prior, str):
            if prior == 'uniform':
                prior = np.ones(template_feat.shape[0])
            elif prior == 'horizontal':
                sph = self.template.coords.spherical_elevation
                prior = np.exp(
                    -0.5 * (np.rad2deg(sph[:, 1]) / sigmas['sigma_prior'])**2)
            else:
                raise ValueError(
                    f"Unknown prior: {prior!r}. "
                    f"Use 'uniform', 'horizontal', or a numpy array.")
            prior /= np.sum(prior)
        elif isinstance(prior, np.ndarray):
            if prior.shape[0] != template_feat.shape[0]:
                raise ValueError(
                    f'Prior shape mismatch: '
                    f'{prior.shape[0]} vs {template_feat.shape[0]}')
            prior = prior / np.sum(prior)
        else:
            raise TypeError(
                "Prior must be str ('uniform', 'horizontal') or numpy array.")

        template_num = template_feat.shape[0]
        posterior = np.zeros((target_num, repetitions, template_num))
        if not store_posterior:
            posterior_idx = np.zeros(
                (target_num, repetitions), dtype=np.int32)

        if repetitions > 1:
            L = np.linalg.cholesky(sigma)
            for t in range(target_num):
                ts = np.tile(target_feat[t, :], [repetitions, 1])
                xs = ts + rng.normal(size=ts.shape) @ L.T
                loglik = utils.multiple_logpdfs_vec_input_single_cov(
                    xs, template_feat, logdet, Us).squeeze()
                logpost = loglik + np.log(prior)
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)
                logpost = np.logaddexp(
                    logpost, np.log(np.finfo(loglik.dtype).eps))
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)
                if store_posterior:
                    posterior[t, :, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=1)
        else:
            for t in range(target_num):
                # AWGN NOISE
                x = rng.multivariate_normal(target_feat[t, :], sigma)

                # COMPUTE POSTERIOR
                loglik = utils.multiple_logpdfs_vec_input_single_cov(
                    np.expand_dims(x, axis=0),
                    template_feat, logdet, Us).squeeze()
                logpost = loglik + np.log(prior)

                # normalise
                logpost = logpost - logsumexp(logpost)

                # add numerical precision to avoid underflow (i.e. prob = 0)
                # it also function as a negligible lapse rate
                logpost = np.logaddexp(
                    logpost, np.log(np.finfo(loglik.dtype).eps))

                # normalise
                logpost = logpost - logsumexp(logpost)

                # the solution above is faster than the for loop below but I am
                # keeping it for future reference and debugging
                # post = np.zeros(template_num)
                # for tp in range(template_num):
                #     # post[tp] = multivariate_normal.pdf(
                #     #     x,mean=template_feat[tp], cov=sigma) * prior[tp]
                #     # doing this speeds up stuff
                #     u_diff = (x-template_feat[tp])
                #     post[tp] = (
                #         np.exp(-0.5*u_diff @ sigma_inv @ u_diff.T))*prior[tp]
                # post /= np.sum(post)
                # logpost = np.log(post+np.finfo(post.dtype).eps)

                if store_posterior:
                    posterior[t, 0, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=0)

        return posterior if store_posterior else posterior_idx


    def estimate(self, posterior, kappa_motor=None, seed=None):
        """
        Estimate directions from posterior distribution.

        Parameters
        ----------
        posterior : ndarray
            Either full posterior (trials :math:`\times` repetitions :math:`\times` templates)
            OR argmax indices (trials :math:`\times` repetitions)
            if computed with ``store_posterior=False`` (see :py:meth:`~infer` for details)
        kappa_motor : float or None, optional
            Motor noise concentration. The concentration parametrises a von Mises - Fisher distribution
            and can be obtained from a standard deviation in degrees froom fitting.sigma_to_kappa()
            If None, uses self.parameters['kappa_motor'].
            If False or 0, motor noise is disabled.
        seed : int or None, optional
            Fixed random seed for reproducibility.

        Returns
        -------
        estimations : ndarray
            Estimated directions in Cartesian coordinates
            (trials :math:`\times` repetitions :math:`\times` 3)
        """
        repetitions = np.size(posterior, 1)
        trials = np.size(posterior, 0)
        assert(trials > 0)

        coords_temp = self.template.coords.cartesian

        # Shape check: 2D = indices, 3D = full posterior
        if (posterior.ndim == 2):
            # Shape: (trials, repetitions, 3)
            estimations = coords_temp[posterior]
        else:
            estimations = np.zeros((trials, repetitions, 3))
            # loops for full posterior
            for t in range(trials):
                for r in range(repetitions):
                    idx = np.argmax(posterior[t, r, :])
                    estimations[t, r, :] = coords_temp[idx, :]

        # pointing error - apply only if kappa_motor is not disabled
        if kappa_motor is None:
            kappa_motor = self.parameters['kappa_motor']

        if kappa_motor not in [False, 0]:
            for rt in range(repetitions):
                estimations[:, rt, :] = utils.scatter_von_mises(
                    estimations[:, rt, :], kappa_motor, seed=seed)

        return pf.Coordinates.from_cartesian(estimations[..., 0],
                                             estimations[..., 1],
                                             estimations[..., 2])

    def plot_cues(self, title='', fig=None, ax=None, clim=None, elev_min=None):
        """Plot spectral cues on the median plane.

        Parameters
        ----------
        title : str, optional
            Additional title text.
        fig, ax : matplotlib objects, optional
            Existing figure/axes to plot on.
        clim : tuple, optional
            Color limits (min, max) for intensity.
        elev_min : float, optional
            Minimum elevation to display.

        Returns
        -------
        fig, ax : matplotlib objects
        """
        if self.target is None:
            raise ValueError(
                'Target not set. Call compute_target() before plot_cues().')
        side = 0 # left/right channel
        dirs = self.coords.spherical_elevation
        dirs[:, 0:2] = np.rad2deg(dirs[:, 0:2])
        # select directions with azimuth almost zero (median frontal plane)
        median_idx = np.abs(dirs[:, 0] - 0) < 2
        elevations = dirs[median_idx,1]
        amps = self.target.spectral_cues[median_idx, :, side]

        # sort by elevations (this avoids jumps in the plot but might
        #  introduce some artifacts if median plane is not uniformely sampled)
        sorted_indices = np.argsort(elevations)
        elevations = elevations[sorted_indices,]
        amps = amps[sorted_indices, :]

        # Apply elevation cutoff if specified
        if elev_min is not None:
            elev_mask = elevations >= elev_min
            elevations = elevations[elev_mask]
            amps = amps[elev_mask, :]

        # Create new figure/axis if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.pcolormesh(amps, shading='gouraud')
        plt.colorbar(im, ax=ax, label='Intensity [dB]')
        ax.set_title('Spectral cues (azimuth = o) ' + title)
        ax.set_ylabel('Elevation [deg]')
        ax.set_xlabel('Frequency [Hz]')

        # Use provided color limits or compute from data
        if clim is not None:
            im.set_clim(clim[0], clim[1])
        else:
            im.set_clim(np.min(amps), np.max(amps))

        ax.set_xticks(np.interp([100, 1e3, 5e3, 1e4],
                                self.target.freqs,
                                np.arange(len(self.target.freqs))))
        ax.set_xticklabels([f'{freq:.0f}' for freq in [100, 1e3, 5e3, 1e4]])
        ax.set_yticks(np.arange(len(elevations)))
        ax.set_yticklabels([f'{elev:.0f}' for elev in elevations])
        plt.show()

        return fig, ax

    def plot_post(self, posterior, estimations):
        """Plot posterior distribution with estimated direction overlay."""
        amps = posterior.squeeze()

        ax = self.template.coords.show(
                c=np.maximum(amps, np.log(np.finfo(amps.dtype).eps)),
                s=20,
                alpha=.5,
                label='Log posterior')

        ax.plot([0, 1], [0, 0], zs=[0, 0], c='red', label='Front direction')

        if estimations is not None:
            ax.plot(xs=[0, estimations.x.squeeze()],
                    ys=[0, estimations.y.squeeze()],
                    zs=[0, estimations.z.squeeze()],
                    c='blue',
                    label='Estimated direction',
                    )

        ax.view_init(elev=20, azim=35)
        ax.set_box_aspect([1, 1, 1])
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Values')
        ax.legend()
        plt.show()

