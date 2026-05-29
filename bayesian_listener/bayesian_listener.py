"""BayesianListener module: core auditory model for sound localisation."""
import warnings
import sofar
import numpy as np
import pyfar as pf
from scipy.special import logsumexp
from bayesian_listener import utils
from bayesian_listener import resample
from bayesian_listener.auditory_representation import (
    _AuditoryRepresentation, CONVENTIONS)
from pathlib import Path

class BayesianListener:
    """Bayesian model of human static sound localisation from HRTFs.

    Implements the generative pipeline of :footcite:t:`barumerli2023`, validated and
    extended in :footcite:t:`barumerli2026`:  noisy spatial features (ITD, ILD, monaural
    spectra) are extracted from a binaural stimulus, compared against
    direction-labelled HRTF templates, combined with an elevation prior, and
    finally perturbed by motor noise to yield a directional response.

    See :ref:`background` for the full equations and parameter definitions.

    References
    ----------
    .. :footcite:t:`barumerli2023`  Eqs. 1–7 (model formulation).
    .. :footcite:t:`barumerli2026`  Eqs. 8–14 (likelihood, fitting, BIC).

    Examples
    --------
    Load an HRTF, prepare features, and infer a single direction:

    >>> from bayesian_listener import BayesianListener   # doctest: +SKIP
    >>> bl = BayesianListener('subject01.sofa')          # doctest: +SKIP
    >>> bl.compute_template(interpolation='SHMAX')       # doctest: +SKIP
    >>> posterior = bl.infer(repetitions=50, seed=0)     # doctest: +SKIP
    >>> response  = bl.estimate(posterior, seed=0)       # doctest: +SKIP
    """

    def __init__(self, sofa,
                 sigma_itd=0.569,
                 sigma_ild=1.0,
                 sigma_spectral=10.4,
                 sigma_prior=69.0,
                 kappa_motor=23.31):
        r"""Initialise the listener from a SOFA file or an in-memory Sofa object.

        Parameters
        ----------
        sofa : str or :class:`sofar.Sofa`
            Path to a SOFA file or a pre-loaded :class:`sofar.Sofa` object.
            When a file path is given, it is stored as ``self.sofa_file`` and
            used as a cache key by :meth:`compute_template`.  When a
            :class:`sofar.Sofa` object is given, ``self.sofa_file`` is
            ``None`` and caching is disabled.
        sigma_itd : float, default=0.569
            ITD perceptual noise :math:`\sigma_{\mathrm{itd}}` (dimensionless,
            applied to the warped ITD feature of Eq. 1, :footcite:t:`barumerli2023`).
            Fixed at the literature value during the two-stage fit.
        sigma_ild : float, default=1.0
            ILD perceptual noise :math:`\sigma_{\mathrm{ild}}` in dB.
            Fixed at the literature value during the two-stage fit.
        sigma_spectral : float, default=10.4
            Monaural spectral noise :math:`\sigma_{\mathrm{mon}}` in dB
            (paper symbol ``sigma_mon``, Eq. 2 of :footcite:t:`barumerli2023`).  Fitted
            in stage 2 of the procedure described in
            :func:`~bayesian_listener.fitting.fit_listener`.
        sigma_prior : float, default=69.0
            Elevation prior width :math:`\sigma_{\mathrm{prior}}` in degrees
            (Eq. 5 of :footcite:t:`barumerli2023`).  Fitted in stage 2.  Group-average
            value from :footcite:t:`barumerli2026`, Table 1.
        kappa_motor : float, default=23.31
            Motor-noise concentration :math:`\kappa_m` of the von
            Mises–Fisher response distribution (Eq. 7 of :footcite:t:`barumerli2023`).
            Convert to a circular standard deviation in degrees with
            :func:`~bayesian_listener.fitting.kappa_to_sigma`.

        Attributes
        ----------
        sofa_file : str or None
            Path to the SOFA file, or ``None`` if a Sofa object was passed.
        hrir : :class:`numpy.ndarray`
            Head-related impulse responses, shape ``(n_directions, 2, n_samples)``.
        fs : int
            Sampling rate in Hz.
        coords : :class:`pyfar.Coordinates`
            Source positions, one per HRIR row.
        parameters : dict
            Mapping with keys ``sigma_itd``, ``sigma_ild``, ``sigma_spectral``,
            ``sigma_prior``, ``kappa_motor``.

        Raises
        ------
        ValueError
            If ``sofa`` is neither a string path nor a :class:`sofar.Sofa`
            instance.

        Examples
        --------
        >>> bl = BayesianListener('subject01.sofa',
        ...                       sigma_spectral=8.0,
        ...                       sigma_prior=55.0)        # doctest: +SKIP
        >>> bl.fs                                          # doctest: +SKIP
        48000
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
        self.coords = pf.Coordinates.from_spherical_elevation(
            np.deg2rad(sp[:, 0]), np.deg2rad(sp[:, 1]), sp[:, 2])

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
        """Noise and prior parameters as a dict.

        Required keys: ``sigma_itd``, ``sigma_ild``, ``sigma_spectral``,
        ``sigma_prior``, ``kappa_motor``.  The setter raises
        :class:`ValueError` if any key is missing.
        """
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
        r""":math:`\sigma_{\mathrm{itd}}`, the warped-ITD noise (dimensionless)."""
        return self.parameters['sigma_itd']

    @sigma_itd.setter
    def sigma_itd(self, v):
        self.parameters['sigma_itd'] = v

    @property
    def sigma_ild(self):
        r""":math:`\sigma_{\mathrm{ild}}`, the ILD noise in dB."""
        return self.parameters['sigma_ild']

    @sigma_ild.setter
    def sigma_ild(self, v):
        self.parameters['sigma_ild'] = v

    @property
    def sigma_spectral(self):
        r""":math:`\sigma_{\mathrm{mon}}`, the monaural spectral noise in dB."""
        return self.parameters['sigma_spectral']

    @sigma_spectral.setter
    def sigma_spectral(self, v):
        self.parameters['sigma_spectral'] = v

    @property
    def sigma_prior(self):
        r""":math:`\sigma_{\mathrm{prior}}`, the elevation prior width in degrees."""
        return self.parameters['sigma_prior']

    @sigma_prior.setter
    def sigma_prior(self, v):
        self.parameters['sigma_prior'] = v

    @property
    def kappa_motor(self):
        r""":math:`\kappa_m`, the von Mises–Fisher motor-noise concentration."""
        return self.parameters['kappa_motor']

    @kappa_motor.setter
    def kappa_motor(self, v):
        self.parameters['kappa_motor'] = v

    @property
    def sigma_motor(self):
        r"""Motor-noise circular standard deviation :math:`\sigma_m` in degrees.

        Computed from :attr:`kappa_motor` via the Bessel-ratio identity
        :math:`R = I_1(\kappa_m)/I_0(\kappa_m) = \exp(-\sigma_m^2/2)`.
        See :func:`~bayesian_listener.fitting.kappa_to_sigma`.
        """
        from bayesian_listener.fitting import kappa_to_sigma
        return kappa_to_sigma(self.parameters['kappa_motor'])

    @sigma_motor.setter
    def sigma_motor(self, value):
        from bayesian_listener.fitting import sigma_to_kappa
        self.parameters['kappa_motor'] = sigma_to_kappa(value)

    @property
    def target(self):
        """Auditory representation of the stimulus, or ``None`` if not computed.

        Set by :meth:`compute_target`.  Assigning a value validates that it is
        a :class:`~bayesian_listener.auditory_representation.Barumerli2023` instance
        (or subclass, or ``None``); otherwise raises :class:`TypeError`.
        """
        return self._target

    @target.setter
    def target(self, value):
        if value is not None and not isinstance(value, _AuditoryRepresentation):
            raise TypeError('target must be a Barumerli2023 (or subclass).')
        self._target = value

    @property
    def template(self):
        """Listener's internal template, or ``None`` if not computed.

        Set by :meth:`compute_template`.  Assigning a value validates that it is
        a :class:`~bayesian_listener.auditory_representation.Barumerli2023` instance
        (or subclass, or ``None``); otherwise raises :class:`TypeError`.
        """
        return self._template

    @template.setter
    def template(self, value):
        if value is not None and not isinstance(value, _AuditoryRepresentation):
            raise TypeError('template must be a Barumerli2023 (or subclass).')
        self._template = value

    def _interpolate(self, ar, interpolation='SHMAX', interpolation_grid=None):
        """Resample a :class:`~bayesian_listener.auditory_representation.Barumerli2023`
        onto a uniform grid.

        Parameters
        ----------
        ar : Barumerli2023
            Source representation to interpolate.
        interpolation : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'}, default='SHMAX'
            Interpolation method:

            - ``'SH'`` — regularised spherical-harmonic interpolation with
              the maximum stable order chosen by
              :func:`~bayesian_listener.resample.find_max_order`.
            - ``'SHMAX'`` — spherical-harmonic interpolation at fixed order 44
              with Tikhonov regularisation.
            - ``'barycentric'`` — VBAP/barycentric weights on the convex hull
              of the sampling grid.
            - ``'barumerli2023'`` — original method of :footcite:t:`barumerli2023`,
              order-15 SH; truncated below the lowest measured elevation.
        interpolation_grid : :class:`pyfar.Coordinates` or None, default=None
            Target grid.  If ``None``, uses a 64th-degree spherical t-design
            (2,112 quasi-uniform points).

        Returns
        -------
        Barumerli2023
            Same subclass as ``ar``, resampled onto the target grid.
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

    def compute_target(self, convention='Barumerli2023', spectral_range=None,
                       use_cache=True, force_recompute=False, cache_dir=None):
        """Extract auditory features from ``self.hrir`` and store them as :attr:`target`.

        The target is the stimulus-side representation: ITD, ILD, and monaural
        spectral features computed at the measured HRTF directions.  Each row
        of ``target.coords`` corresponds to one measured direction,
        preserving the original HRTF measurement grid. No interpolation is applied.

        Call this method before :meth:`compute_template` when you need control
        over feature extraction parameters such as ``spectral_range`` or
        ``convention``.  If :attr:`target` is already set when
        :meth:`compute_template` is called, it will not be recomputed.
        Caching is available only when the listener was initialised with a SOFA
        file path; it is automatically disabled otherwise.

        Parameters
        ----------
        convention : str, default='Barumerli2023'
            Auditory representation to use.  Must be a key in
            ``bayesian_listener.auditory_representation.CONVENTIONS``.
            Currently ``'Barumerli2023'`` (ITD + ILD + spectral amplitudes) is
            the only fully implemented convention.
        spectral_range : list of float or None, default=None
            ``[low_Hz, high_Hz]`` frequency limits of the gammatone filterbank
            used for the monaural cues.  ``None`` selects ``[700.0, 18000.0]``.
        use_cache : bool, default=True
            Load from cache if available and save after computing.
        force_recompute : bool, default=False
            If ``True``, ignore any cached target and recompute from scratch.
        cache_dir : str or :class:`pathlib.Path` or None, default=None
            Cache directory.  ``None`` selects
            ``Path.cwd() / 'data' / 'preprocessed'``.

        Raises
        ------
        ValueError
            If ``convention`` is not a registered key, or if ``sofa_file`` is
            ``None`` (listener was initialised with a :class:`sofar.Sofa`
            object rather than a file path).

        Examples
        --------
        >>> bl = BayesianListener('subject01.sofa')           # doctest: +SKIP
        >>> bl.compute_target(spectral_range=[500, 16000])    # doctest: +SKIP
        >>> bl.target.features.shape                          # doctest: +SKIP
        (793, 58)
        """
        if use_cache and self.sofa_file is None:
            use_cache = False
            UserWarning('Caching disabled since sofa file path unavailable')

        if spectral_range is None:
            spectral_range = [7e2, 18e3]
        if convention not in CONVENTIONS:
            raise ValueError(
                f"Unknown convention '{convention}'. "
                f"Available: {list(CONVENTIONS)}")
        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'preprocessed'
        else:
            cache_dir = Path(cache_dir)

        if use_cache and not force_recompute:
            target = utils.cache_load_target(cache_dir, self.sofa_file)
            if target is not None:
                self.target = target
                return

        itd, ild, spectral_cues, freqs = utils.compute_features(
            self.hrir, self.coords, self.fs, spectral_range)
        self.target = CONVENTIONS[convention](
            coords=self.coords,
            itd=itd,
            ild=ild,
            spectral_cues=spectral_cues,
            freqs=freqs,
        )

        if use_cache:
            utils.cache_save_target(cache_dir, self.sofa_file, self.target)

    def compute_template(self, interpolation='SHMAX', interpolation_grid=None,
                         use_cache=True, force_recompute=False, cache_dir=None):
        """Build the template: the listener's learned mapping from features to directions.

        The template is constructed from the individual HRTF by extracting
        auditory features and interpolating them onto a quasi-uniform spherical
        grid (default: 2,112 directions at ~4° spacing).  It represents the
        internal model the listener uses to infer source directions during
        :meth:`infer`.

        If :attr:`target` has not been set, :meth:`compute_target` is called
        automatically with default parameters.  For finer control over feature
        extraction (e.g. spectral range, convention), call :meth:`compute_target`
        explicitly before calling this method.

        This function does not modify :attr:`target` if already set.
        Caching is available only when the listener was initialised with a SOFA
        file path; it is automatically disabled otherwise.

        Parameters
        ----------
        interpolation : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'} or None, default='SHMAX'
            Interpolation method used to resample features onto the uniform grid.
            Pass ``None`` to skip interpolation and set :attr:`template` equal
            to :attr:`target` (useful when the HRTF is already on a uniform grid).
        interpolation_grid : :class:`pyfar.Coordinates` or None, default=None
            Directions of the template grid.  ``None`` selects a 64th-degree
            spherical t-design (2,112 directions, ~4° average spacing).
        use_cache : bool, default=True
            Load from cache if available and save after computing.  Requires a
            SOFA file path; raises :class:`ValueError` if ``sofa_file`` is ``None``.
        force_recompute : bool, default=False
            If ``True``, ignore any cached template and recompute from scratch.
        cache_dir : str or :class:`pathlib.Path` or None, default=None
            Cache directory.  ``None`` selects
            ``Path.cwd() / 'data' / 'preprocessed'``.

        Raises
        ------
        ValueError
            If ``sofa_file`` is ``None`` (i.e. listener was initialised with a
            :class:`sofar.Sofa` object rather than a file path).

        Examples
        --------
        >>> bl = BayesianListener('subject01.sofa')                  # doctest: +SKIP
        >>> bl.compute_template(interpolation='SHMAX')               # doctest: +SKIP
        >>> bl.template.features.shape[0]                            # doctest: +SKIP
        2112

        To control feature extraction before building the template:

        >>> bl.compute_target(spectral_range=[500, 16000])           # doctest: +SKIP
        >>> bl.compute_template()                                     # doctest: +SKIP
        """
        if use_cache and self.sofa_file is None:
            use_cache = False
            UserWarning('Caching disabled since sofa file path unavailable')

        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'preprocessed'
        else:
            cache_dir = Path(cache_dir)

        # step 1: ensure target is available without modifying it if already set
        if self.target is None:
            self.compute_target(use_cache=use_cache,
                                cache_dir=cache_dir)

        # step 2: if no interpolation, template == target
        if interpolation is None:
            self.template = self.target
            return

        # step 3: try cache, else interpolate and cache
        if use_cache and not force_recompute:
            template = utils.cache_load_template(
                cache_dir, self.sofa_file, interpolation)
            if template is not None:
                self.template = template
                return

        self.template = self._interpolate(
            self.target, interpolation, interpolation_grid)

        if use_cache:
            utils.cache_save_template(
                cache_dir, self.sofa_file, interpolation, self.template)

    def infer(self,
              repetitions=50,
              prior='horizontal',
              store_posterior=False,
              seed=None):
        r"""Run Monte Carlo Bayesian inference of source direction.

        For each target direction, draw ``repetitions`` noisy feature samples
        :math:`\mathbf{t}` from
        :math:`\mathcal{N}(\mathbf{s}(\boldsymbol{\varphi}), \boldsymbol{\Sigma})`,
        then compute the log-posterior
        :math:`p(\boldsymbol{\varphi} \mid \mathbf{t}) \propto
        p(\mathbf{t} \mid \boldsymbol{\varphi})\, p(\boldsymbol{\varphi})`
        (Eq. 3 of :footcite:t:`barumerli2023`) over all template directions.

        Parameters
        ----------
        repetitions : int, default=50
            Number of Monte Carlo trials per target direction.
        prior : {'uniform', 'horizontal'} or :class:`numpy.ndarray`, default='horizontal'
            Direction prior :math:`p(\boldsymbol{\varphi})`:

            - ``'uniform'`` — flat prior over template directions.
            - ``'horizontal'`` — Gaussian prior over elevation with width
              :attr:`sigma_prior` (Eq. 5 of :footcite:t:`barumerli2023`).
            - array of shape ``(n_templates,)`` — custom unnormalised
              prior; normalised internally.
        store_posterior : bool, default=False
            If ``True``, return the full normalised log-posterior over
            templates.  If ``False``, return only argmax indices (memory-light).
        seed : int or None, default=None
            Seed for :func:`numpy.random.default_rng`.  ``None`` yields a
            non-reproducible run.

        Returns
        -------
        :class:`numpy.ndarray`
            If ``store_posterior=True``: log-posterior of shape
            ``(n_targets, repetitions, n_templates)`` (rows sum to 1 in
            linear domain).  Otherwise: integer template indices of shape
            ``(n_targets, repetitions)``.

        Raises
        ------
        ValueError
            If :attr:`target` or :attr:`template` is ``None``, if their
            conventions differ, if ``prior`` is an array of the wrong shape,
            or if ``prior`` is an unknown string.
        TypeError
            If ``prior`` is neither a string nor a :class:`numpy.ndarray`.

        Examples
        --------
        >>> bl.compute_target(); bl.compute_template()       # doctest: +SKIP
        >>> idx = bl.infer(repetitions=200, seed=0)          # doctest: +SKIP
        >>> idx.shape                                        # doctest: +SKIP
        (793, 200)
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
                # post = np.zeros(template_num)                                 # noqa: ERA001
                # for tp in range(template_num):                                # noqa: ERA001
                #     # post[tp] = multivariate_normal.pdf(                     # noqa: ERA001
                #     #     x,mean=template_feat[tp], cov=sigma) * prior[tp]    # noqa: ERA001
                #     # doing this speeds up stuff                              # noqa: ERA001
                #     u_diff = (x-template_feat[tp])                            # noqa: ERA001
                #     post[tp] = (                                              # noqa: ERA001
                #         np.exp(-0.5*u_diff @ sigma_inv @ u_diff.T))*prior[tp] # noqa: ERA001
                # post /= np.sum(post)                                          # noqa: ERA001
                # logpost = np.log(post+np.finfo(post.dtype).eps)               # noqa: ERA001

                if store_posterior:
                    posterior[t, 0, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=0)

        return posterior if store_posterior else posterior_idx


    def estimate(self, posterior, kappa_motor=None, seed=None):
        r"""Convert posterior(s) into pointing responses with motor noise applied.

        Selects the MAP template direction for each (trial, repetition) pair
        and perturbs it with isotropic motor noise drawn from
        :math:`\mathrm{vMF}(\mathbf{0}, \kappa_m)` (Eq. 7 of :footcite:t:`barumerli2023`).

        Parameters
        ----------
        posterior : :class:`numpy.ndarray`
            Output of :meth:`infer`.  Either:

            - log-posteriors of shape ``(n_targets, repetitions, n_templates)``
              (when :meth:`infer` was called with ``store_posterior=True``), or
            - argmax indices of shape ``(n_targets, repetitions)``.
        kappa_motor : float, ``False``, ``0`` or None, default=None
            von Mises–Fisher concentration for the motor-noise step.

            - ``None`` — use ``self.parameters['kappa_motor']``.
            - ``False`` or ``0`` — disable motor noise (return raw MAP
              directions).
            - any positive float — explicit concentration.  Convert from a
              circular SD with
              :func:`~bayesian_listener.fitting.sigma_to_kappa`.
        seed : int or None, default=None
            Seed for the motor-noise RNG.

        Returns
        -------
        :class:`pyfar.Coordinates`
            Pointing responses with cartesian array shape
            ``(n_targets, repetitions, 3)``.

        Raises
        ------
        ValueError
            If ``posterior`` has zero rows.

        Examples
        --------
        >>> idx       = bl.infer(repetitions=50, seed=0)        # doctest: +SKIP
        >>> responses = bl.estimate(idx, seed=0)                # doctest: +SKIP
        >>> responses.cartesian.shape                           # doctest: +SKIP
        (793, 50, 3)
        """
        repetitions = np.size(posterior, 1)
        trials = np.size(posterior, 0)
        if trials <= 0:
            raise ValueError('posterior must have at least one trial.')

        coords_temp = self.template.coords.cartesian

        # Shape check: 2D = indices, 3D = full posterior
        if (posterior.ndim == 2):
            # Shape: (trials, repetitions, 3)  # noqa: ERA001
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

    def localise(self, target=None, directions=None, repetitions=None,
                 seed=None):
        """Run the full localisation pipeline and return pointing responses.

        Convenience wrapper around :meth:`infer` and :meth:`estimate`.
        Ensures the template (the listener's internal directional model) and
        the target (the stimulus features to be localised) are both available,
        then draws ``repetitions`` noisy percepts via Monte Carlo inference and
        converts each MAP estimate into a pointing response perturbed by motor
        noise.

        Parameters
        ----------
        target : :class:`~bayesian_listener.auditory_representation._AuditoryRepresentation` or None, default=None
            Auditory representation of the source directions to localise.
            Must be the same subclass as :attr:`template`.  ``None`` reuses
            :attr:`target`, computing it via :meth:`compute_target` if not
            already set.  When combined with ``directions``, acts as the
            search pool from which directions are selected.
        directions : :class:`pyfar.Coordinates` or None, default=None
            Desired source directions.  The nearest neighbours in the
            resolved ``target`` are selected via great-circle distance.  A
            :class:`UserWarning` is raised for any match farther than 10°.
            Raises :class:`ValueError` if the intersection is empty.
        repetitions : int or None, default=None
            Number of Monte Carlo trials passed to :meth:`infer`.
            ``None`` uses the default of :meth:`infer` (50).
        seed : int or None, default=None
            Seed for :func:`numpy.random.default_rng`.  ``None`` yields a
            non-reproducible run.

        Returns
        -------
        :class:`pyfar.Coordinates`
            Pointing responses of shape ``(n_targets, repetitions, 3)``.

        Raises
        ------
        ValueError
            If ``target`` and ``directions`` are both provided.
        ValueError
            If ``target`` has a different convention than :attr:`template`.
        ValueError
            If ``directions`` finds no match within the provided or stored
            ``target`` (empty intersection).

        Examples
        --------
        >>> responses = bl.localise(repetitions=1, seed=0)      # doctest: +SKIP
        >>> responses.cartesian.shape                           # doctest: +SKIP
        (793, 1, 3)
        """
        if self.template is None:
            self.compute_template()

        if target is not None:
            if type(target) is not type(self.template):
                raise ValueError(
                    f'Convention mismatch: '
                    f'target={target.convention!r}, '
                    f'template={self.template.convention!r}.')
            self.target = target

        if self.target is None:
            self.compute_target()

        if directions is not None:
            # rescale to match the target grid radius so find_nearest doesn't reject
            target_radius = float(self.target.coords.radius.flat[0])
            unit = directions.cartesian / np.linalg.norm(
                directions.cartesian, axis=-1, keepdims=True)
            directions_scaled = pf.Coordinates.from_cartesian(
                *(unit * target_radius).T)
            indices, distances = self.target.coords.find_nearest(
                directions_scaled, distance_measure='spherical_radians')
            indices = list(np.atleast_1d(indices))
            if len(indices) == 0:
                raise ValueError(
                    "No directions matched: the intersection of 'directions' "
                    "and 'target' is empty.")
            distances_deg = np.rad2deg(np.atleast_1d(distances))
            far = distances_deg > 10
            if np.any(far):
                warnings.warn(
                    f"{np.sum(far)} requested direction(s) snapped to a grid "
                    f"point farther than 10°  (max {distances_deg.max():.1f}°).",
                    UserWarning, stacklevel=2)
            self.target = self.target[indices]

        infer_kwargs = {"seed": seed}
        if repetitions is not None:
            infer_kwargs["repetitions"] = repetitions
        estimates = self.estimate(self.infer(**infer_kwargs), seed=seed)

        return estimates



