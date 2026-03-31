"""BayesianListener module: core auditory model for sound localisation."""
import sofar
import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from bayesian_listener import utils
from bayesian_listener import resample
from pathlib import Path

class BayesianListener:
    """Bayesian model of human sound localisation using HRTF-derived cues."""

    def __init__(self, sofa):
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

        # noise parameters
        # these values are the group average got from the 2026 paper.
        self.parameters = {
            "sigma_itd": 0.569,
            "sigma_ild": 1.0,
            "sigma_spectral": 10.4,
            "sigma_prior": 69.0,
            "kappa_motor": 23.31,  # ~12 deg via Bessel-based conversion
        }

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, dict):
            raise ValueError("Parameters must be a dictionary.")
        # backward compatibility: migrate sigma_motor -> kappa_motor
        if 'sigma_motor' in value and 'kappa_motor' not in value:
            from bayesian_listener.fitting import sigma_to_kappa
            sigma_m = value.pop('sigma_motor')
            value['kappa_motor'] = sigma_to_kappa(sigma_m) if sigma_m else 0
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

    def interpolate(self, interpolation='SH'):
        """Resample cues to a uniform spherical grid for internal templates.

        Templates are used during inference to compare against target features.
        This resampling ensures a consistent spatial resolution.

        Parameters
        ----------
        interpolation : {'SH', 'SHmax', 'barycentric', 'barumerli2023'}, default='SH'
            Interpolation method:
            - 'SH': Spherical harmonics interpolation with SH truncation.
            - 'SHmax': Spherical harmonics with high SH order.
            - 'barycentric': Barycentric interpolation on triangulated mesh.
            - 'barumerli2023': Method from Barumerli et al. (2023).

        Returns
        -------
        BayesianListener
            New instance with resampled cues on uniform grid.
        """
        # Create empty instance
        model = BayesianListener.__new__(BayesianListener)

        # Resample all cues in a single call
        cues_list = [
            self.itd,
            self.ild,
            self.spectral_cues[:, :, 0],
            self.spectral_cues[:, :, 1],
        ]

        resampled_cues, coords_new = resample.resample(cues_list,
                                                       self.coords,
                                                       self.interpolation_grid,
                                                       method=interpolation)

        # Unpack results
        model.itd = resampled_cues[0]
        model.ild = resampled_cues[1]
        model.spectral_cues = np.stack([resampled_cues[2],
                                        resampled_cues[3]],
                                       axis=-1)
        model.coords = coords_new
        model.freqs = self.freqs
        # model.coords.plot(model.spectral_cues[:, 5, 0])
        # self.coords.plot(self.spectral_cues[:, 5, 0])

        return model

    # prepare
    def prepare_features(self,
                         spectral_range=[7e2, 18e3],
                         interpolation='SHMAX',
                         interpolation_grid=None,
                         use_cache=True,
                         force_recompute=False,
                         cache_dir=None):
        """Compute spatial features and templates, with optional caching.

        Extracts ITD, ILD, and spectral cues from the HRIRs via a gammatone
        filterbank, then interpolates them onto a uniform grid to produce the
        inference template.  Results are loaded from disk when available and
        saved after computation (unless ``self.sofa_file`` is ``None``).

        Parameters
        ----------
        spectral_range : list of float, default=[700, 18000]
            Frequency range [low, high] in Hz used for spectral cue extraction.
        interpolation : str, default='SHMAX'
            Interpolation method for template generation.
            One of ``'SH'``, ``'SHMAX'``, ``'barumerli2023'``, ``'barycentric'``.
        interpolation_grid : pyfar.Coordinates or None, default=None
            Target grid for interpolation.  ``None`` uses the default uniform
            grid defined by the chosen interpolation method.
        use_cache : bool, default=True
            If ``True``, attempt to load features from cache before computing.
            Caching is silently skipped when the listener was initialised
            with a ``sofar.Sofa`` object (``self.sofa_file`` is ``None``).
        force_recompute : bool, default=False
            If ``True``, ignore any cached data and recompute from scratch
            (the new result is still written to cache).
        cache_dir : str or Path, optional
            Directory for cached features.  Defaults to
            ``<cwd>/data/preprocessed``.

        Notes
        -----
        After this method returns, the following attributes are set:
        ``itd``, ``ild``, ``spectral_cues``, ``freqs``, ``template``.
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'preprocessed'
        else:
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir

        self.interpolation_grid = interpolation_grid

        if use_cache:
            return self._load_or_compute_features(spectral_range,
                                                  interpolation,
                                                  force_recompute)
        else:
            return self._compute_features(spectral_range, interpolation)

    def _compute_features(self,
                          spectral_range = [7e2, 18e3],
                          interpolation='SHMAX'):
        """Compute ITD, ILD, and spectral cues from HRIRs and build the template.

        Runs the full feature extraction pipeline via a gammatone filterbank
        and interpolates the results onto a uniform grid.  Sets ``self.itd``,
        ``self.ild``, ``self.spectral_cues``, ``self.freqs``, and
        ``self.template`` in place.

        Parameters
        ----------
        spectral_range : list of float, default=[700, 18000]
            Frequency range [low, high] in Hz for spectral cue extraction.
        interpolation : str, default='SH'
            Interpolation method passed to :meth:`interpolate`.
        """
        self.itd, self.ild, self.spectral_cues, self.freqs = \
            utils.compute_features(self.hrir, self.coords, self.fs,
                                   spectral_range)

        # prepare templates on uniform grid
        self.template = self.interpolate(interpolation)

    def _load_or_compute_features(self,
                                  spectral_range=[7e2, 18e3],
                                  interpolation='SHMAX',
                                  force_recompute=False):
        """Load features from cache, or compute and save them.

        Attempts to restore ``itd``, ``ild``, ``spectral_cues``, ``freqs``,
        ``coords``, ``parameters``, and ``template`` from disk.  Falls back to
        :meth:`_compute_features` on a cache miss and writes the result to
        disk afterwards.  Caching is skipped entirely when ``self.sofa_file``
        is ``None`` (i.e. the listener was initialised with a ``sofar.Sofa``
        object).

        Parameters
        ----------
        spectral_range : list of float, default=[700, 18000]
            Frequency range [low, high] in Hz for spectral cue extraction.
        interpolation : str, default='SHMAX'
            Interpolation method passed to :meth:`_compute_features`.
        force_recompute : bool, default=False
            If ``True``, ignore any cached data and recompute from scratch.
            The new result is still written to cache.
        """
        cache_dir = self.cache_dir

        # Define what attributes to cache/restore
        cache_attributes = [
            'itd', 'ild', 'freqs', 'spectral_cues',
            'coords', 'parameters', 'template',
        ]

        # ========== Try to load from cache ==========
        if not force_recompute and self.sofa_file is not None:
            cached_data = utils.load_from_cache(cache_dir,
                                                self.sofa_file,
                                                cache_attributes,
                                                interpolation)

            if cached_data is not None:
                # Restore cached attributes
                for attr in cache_attributes:
                    setattr(self, attr, cached_data[attr])
                return

            print("  Cache not found or invalid. Recomputing...")

        # ========== Compute features ==========
        print("→ Computing features...")
        self._compute_features(spectral_range=spectral_range,
                               interpolation=interpolation)
        print("✓ Feature preparation complete")

        # ========== Save to cache ==========
        if self.sofa_file is not None:
            # Prepare data to cache
            cache_data = {
                attr: getattr(self, attr) for attr in cache_attributes
                }
            utils.save_to_cache(cache_dir,
                                self.sofa_file,
                                cache_data,
                                interpolation)
        # return internal representation

    def represent(self):
        """Return concatenated feature vector [ITD, ILD, spectral_L, spectral_R].

        Returns
        -------
        ndarray
            Feature matrix of shape (n_directions, n_features).
        """
        bcue = np.hstack([self.itd,
                          self.ild])

        scue = np.hstack([self.spectral_cues[:, :, 0],
                          self.spectral_cues[:, :, 1]])

        return np.hstack([bcue,
                          scue])

    def infer(self,
              target = None,
              repetitions = 50,
              prior = 'horizontal',
              store_posterior = False,
              seed = None):
        """Perform Bayesian inference to estimate sound source direction.

        Parameters
        ----------
        target : array-like, optional
            Target spatial features to localise
            (if None, uses features from listener's own HRIR).
        repetitions : int, default=50
            Number of Monte Carlo samples
            (i.e. number of repetitions for each target).
        seed : int, optional
            Random seed for reproducibility.
        prior : {'uniform', 'horizontal'} or ndarray, default='horizontal'
            Prior distribution over directions. 'horizontal' biases toward
            the horizontal plane; 'uniform' weights all directions equally.
            User can provide custom prior as ndarray (templates :math:`\times` 1).
        store_posterior : bool, default=False
            If True, returns full log-posterior
            (warning: this increase memory usage);
            otherwise returns indices of maximum a posteriori estimates
            (see :py:meth:`~estimate` for details).

        Returns
        -------
        ndarray
            If ``store_posterior=True``: log-posteriors of shape
            (targets :math:`\times` repetitions :math:`\times` templates).
            Otherwise: Estimated template indices of shape
            (targets :math:`\times` repetitions).
        """

        if not hasattr(self, 'itd'):
            raise ValueError(
                'Features not computed. Call prepare_features() before infer().'
            )

        rng = np.random.default_rng(seed)

        # prepare features
        # use original HRIR if no target is provided
        if target is None:
            target_feat = self.represent()
            target_num = target_feat.shape[0]
        else:
            target_feat = target
            if target_feat.ndim == 1:
                target_feat = np.expand_dims(target_feat, axis=0)
            target_num = np.size(target_feat, 0)

        # prepare template features - horrible concatenation but it works
        template_feat = self.template.represent()

        sigmas = self.parameters
        sigma = np.block(np.diag(np.hstack(
            [sigmas["sigma_itd"]**2,
             sigmas["sigma_ild"]**2,
             np.repeat(sigmas["sigma_spectral"]**2, self.freqs.shape[0]*2),
            ])))

        # the following code is needed to speed up multiple_logpdfs_vec_input
        # since here the covariance matrix is constant NumPy broadcasts `eigh`.
        vals, vecs = np.linalg.eigh(sigma)

        # Compute the log determinants across the second axis.
        logdet = np.sum(np.log(vals))
        # Invert the eigenvalues and add a dimension to `valsinvs`
        # so that NumPy broadcasts appropriately.
        Us  = vecs * np.sqrt(1./vals)[:, None]

        # Prior computation
        if isinstance(prior, str):
            if prior == 'uniform':
                # Uniform prior: all directions equally likely
                prior = np.ones(template_feat.shape[0])
            elif prior == 'horizontal':
                # Horizontal bias prior:
                # Gaussian centered on horizontal plane (elevation = 0°)
                sph = self.template.coords.spherical_elevation
                prior = np.exp(
                    -0.5 * (np.rad2deg(sph[:, 1]) / sigmas["sigma_prior"])**2,
                    )
            else:
                raise ValueError(
                    f"Unknown prior: {prior}. "
                    f"Use 'uniform', 'horizontal', or numpy array")
            # Normalize to sum to 1 (valid probability distribution)
            prior /= np.sum(prior)
        elif isinstance(prior, np.ndarray):
            # Custom prior provided as array
            if prior.shape[0] != template_feat.shape[0]:
                raise ValueError(
                    f"Prior shape mismatch: "
                    f"{prior.shape[0]} vs {template_feat.shape[0]}")
            prior = prior / np.sum(prior)  # normalize
        else:
            raise TypeError(
                "Prior must be str ('uniform', 'horizontal') or numpy array")

        # self.template.coords.plot(prior)

        # Internal belief computation
        template_num = template_feat.shape[0]
        if store_posterior:
            posterior = np.zeros((target_num, repetitions, template_num))
        else:
            posterior_idx = np.zeros((target_num, repetitions), dtype=np.int32)

        posterior = np.zeros((target_num, repetitions, template_num))

        if repetitions > 1:
            L = np.linalg.cholesky(sigma)  # L @ L.T = sigma
            for t in range(target_num):
                ts = np.tile(target_feat[t,:], [repetitions, 1])
                xs = ts + rng.normal(size=ts.shape) @ L.T
                loglik = utils.multiple_logpdfs_vec_input_single_cov(
                    xs,template_feat, logdet, Us).squeeze()
                logpost = loglik + np.log(prior)
                # normalise in log space for numerical stability
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)
                # add numerical precision to avoid underflow (i.e. prob = 0)
                # it also function as a negligible lapse rate
                logpost = np.logaddexp(
                    logpost, np.log(np.finfo(loglik.dtype).eps))
                # normalise again
                # (there is a better way but this is ok for now)
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)

                if store_posterior:
                    posterior[t, :, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=1)
        else:
            for t in range(target_num):
                # for ta in range(target_num):
                # AWGN NOISE
                x = rng.multivariate_normal(target_feat[t,:], sigma)

                # COMPUTE POSTERIOR
                # using vectorised solution
                loglik = utils.multiple_logpdfs_vec_input_single_cov(
                    np.expand_dims(x, axis=0),
                    template_feat,
                    logdet,
                    Us,
                    ).squeeze()
                # post = np.exp(loglik+np.log(prior))
                logpost = loglik + np.log(prior)
                # normalise
                logpost = logpost - logsumexp(logpost)
                # add numerical precision to avoid underflow (i.e. prob = 0)
                # it also function as a negligible lapse rate
                logpost = np.logaddexp(logpost,
                                       np.log(np.finfo(loglik.dtype).eps))
                # normalise again
                # (there is a better way but this is ok for now)
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

                # Store posterior -
                if store_posterior:
                    posterior[t, 0, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=0)

        # Results
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
        side = 0 # left/right channel
        dirs = self.coords.spherical_elevation
        dirs[:, 0:2] = np.rad2deg(dirs[:, 0:2])
        # select directions with azimuth almost zero (median frontal plane)
        median_idx = np.abs(dirs[:, 0] - 0) < 2
        elevations = dirs[median_idx,1]
        amps = self.spectral_cues[median_idx, :, side]

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
                                self.freqs,
                                np.arange(len(self.freqs))))
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

