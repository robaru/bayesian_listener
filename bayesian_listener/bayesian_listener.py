"""BayesianListener module: core auditory model for sound localisation."""
import sofar
import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from bayesian_listener import utils
from bayesian_listener import resample
from bayesian_listener.auditory_representation import (
    _AuditoryRepresentation, CONVENTIONS)
from pathlib import Path

class BayesianListener:
    """Bayesian model of human static sound localisation from HRTFs.

    Implements the generative pipeline of [barumerli2023]_, validated and
    extended in [barumerli2026]_:  noisy spatial features (ITD, ILD, monaural
    spectra) are extracted from a binaural stimulus, compared against
    direction-labelled HRTF templates, combined with an elevation prior, and
    finally perturbed by motor noise to yield a directional response.

    See :ref:`background` for the full equations and parameter definitions.

    References
    ----------
    .. [barumerli2023]_  Eqs. 1–7 (model formulation).
    .. [barumerli2026]_  Eqs. 8–14 (likelihood, fitting, BIC).

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
            applied to the warped ITD feature of Eq. 1, [barumerli2023]_).
            Fixed at the literature value during the two-stage fit.
        sigma_ild : float, default=1.0
            ILD perceptual noise :math:`\sigma_{\mathrm{ild}}` in dB.
            Fixed at the literature value during the two-stage fit.
        sigma_spectral : float, default=10.4
            Monaural spectral noise :math:`\sigma_{\mathrm{mon}}` in dB
            (paper symbol ``sigma_mon``, Eq. 2 of [barumerli2023]_).  Fitted
            in stage 2 of the procedure described in
            :func:`~bayesian_listener.fitting.fit_listener`.
        sigma_prior : float, default=69.0
            Elevation prior width :math:`\sigma_{\mathrm{prior}}` in degrees
            (Eq. 5 of [barumerli2023]_).  Fitted in stage 2.  Group-average
            value from [barumerli2026]_, Table 1.
        kappa_motor : float, default=23.31
            Motor-noise concentration :math:`\kappa_m` of the von
            Mises–Fisher response distribution (Eq. 7 of [barumerli2023]_).
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
            - ``'barumerli2023'`` — original method of [barumerli2023]_,
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
        """Compute the auditory representation of the stimulus from ``self.hrir``.

        Sets :attr:`target` to a fresh
        :class:`~bayesian_listener.auditory_representation.Barumerli2023`
        instance.  No interpolation is performed; rows of ``target.coords``
        match the measured HRTF directions one-to-one.  When a SOFA file path
        is known and ``use_cache=True``, the result is stored in an on-disk
        pickle keyed by the file hash and reused on subsequent calls.

        Parameters
        ----------
        convention : str, default='Barumerli2023'
            Key into ``bayesian_listener.auditory_representation.CONVENTIONS``
            selecting the representation subclass.  Currently
            ``'Barumerli2023'`` (ITD + ILD + spectral amplitudes) is the only
            implemented convention; ``'barumerli2023pge'`` is a stub.
        spectral_range : list of float or None, default=None
            ``[low_Hz, high_Hz]`` limits of the gammatone filterbank used for
            the monaural cues.  ``None`` selects ``[700.0, 18000.0]``.
        use_cache : bool, default=True
            If ``True``, attempt to load a previously computed target from
            ``cache_dir`` and save after computing.
        force_recompute : bool, default=False
            If ``True``, recompute even when a cache hit is found.
        cache_dir : str or :class:`pathlib.Path` or None, default=None
            Cache directory.  ``None`` selects
            ``Path.cwd() / 'data' / 'preprocessed'``.

        Raises
        ------
        ValueError
            If ``convention`` is not a registered convention key.

        Examples
        --------
        >>> bl = BayesianListener('subject01.sofa')           # doctest: +SKIP
        >>> bl.compute_target(spectral_range=[500, 16000])    # doctest: +SKIP
        >>> bl.target.features.shape                          # doctest: +SKIP
        (793, 58)
        """
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

        if use_cache and not force_recompute and self.sofa_file is not None:
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

        if self.sofa_file is not None:
            utils.cache_save_target(cache_dir, self.sofa_file, self.target)

    def compute_template(self, interpolation='SHMAX', interpolation_grid=None,
                         spectral_range=None, use_cache=True,
                         force_recompute=False, cache_dir=None):
        """Interpolate :attr:`target` onto a quasi-uniform grid and store it as :attr:`template`.

        If :attr:`target` has not been set, :meth:`compute_target` is called
        automatically with matching ``spectral_range``, ``use_cache``, and
        ``cache_dir`` arguments.  When a SOFA file path is known and
        ``use_cache=True``, both target and template are cached in a single
        per-HRTF pickle so that repeated calls (including with different
        interpolation methods) skip the expensive gammatone filterbank step.

        Parameters
        ----------
        interpolation : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'} or None, default='SHMAX'
            Interpolation method; ``'SH'``, ``'SHMAX'``, ``'barycentric'``, or ``'barumerli2023'``.
            Pass ``None`` to skip interpolation and set :attr:`template` equal
            to :attr:`target` (useful when the HRTF is already on a uniform
            grid, or for the non-individual workflow where only the target is
            needed).
        interpolation_grid : :class:`pyfar.Coordinates` or None, default=None
            Target directions.  ``None`` selects a 64th-degree spherical
            t-design (2,112 directions, 4° average spacing).
        spectral_range : list of float or None, default=None
            ``[low_Hz, high_Hz]`` forwarded to the auto :meth:`compute_target`
            call when :attr:`target` is ``None``.
        use_cache : bool, default=True
            If ``True``, attempt to load previously computed features from
            ``cache_dir`` and save after computing.
        force_recompute : bool, default=False
            If ``True``, recompute even when a cache hit is found.
        cache_dir : str or :class:`pathlib.Path` or None, default=None
            Cache directory.  ``None`` selects
            ``Path.cwd() / 'data' / 'preprocessed'``.

        Examples
        --------
        >>> bl = BayesianListener('subject01.sofa')                  # doctest: +SKIP
        >>> bl.compute_template(interpolation='SHMAX')               # doctest: +SKIP
        >>> bl.template.features.shape[0]                            # doctest: +SKIP
        2112
        """
        if interpolation is None:
            if self.target is None:
                self.compute_target(spectral_range=spectral_range,
                                    use_cache=use_cache,
                                    force_recompute=force_recompute,
                                    cache_dir=cache_dir)
            self.template = self.target
            return

        if cache_dir is None:
            cache_dir = Path.cwd() / 'data' / 'preprocessed'
        else:
            cache_dir = Path(cache_dir)

        if use_cache and not force_recompute and self.sofa_file is not None:
            template = utils.cache_load_template(
                cache_dir, self.sofa_file, interpolation)
            if template is not None:
                self.template = template
                if self.target is None:
                    target = utils.cache_load_target(cache_dir, self.sofa_file)
                    if target is not None:
                        self.target = target
                return

        if self.target is None:
            self.compute_target(spectral_range=spectral_range,
                                use_cache=use_cache,
                                force_recompute=force_recompute,
                                cache_dir=cache_dir)

        self.template = self._interpolate(
            self.target, interpolation, interpolation_grid)

        if self.sofa_file is not None:
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
        (Eq. 3 of [barumerli2023]_) over all template directions.

        Parameters
        ----------
        repetitions : int, default=50
            Number of Monte Carlo trials per target direction.
        prior : {'uniform', 'horizontal'} or :class:`numpy.ndarray`, default='horizontal'
            Direction prior :math:`p(\boldsymbol{\varphi})`:

            - ``'uniform'`` — flat prior over template directions.
            - ``'horizontal'`` — Gaussian prior over elevation with width
              :attr:`sigma_prior` (Eq. 5 of [barumerli2023]_).
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
                # post = np.zeros(template_num)  # noqa: ERA001
                # for tp in range(template_num):  # noqa: ERA001
                #     # post[tp] = multivariate_normal.pdf(  # noqa: ERA001
                #     #     x,mean=template_feat[tp], cov=sigma) * prior[tp]  # noqa: ERA001
                #     # doing this speeds up stuff  # noqa: ERA001
                #     u_diff = (x-template_feat[tp])  # noqa: ERA001
                #     post[tp] = (  # noqa: ERA001
                #         np.exp(-0.5*u_diff @ sigma_inv @ u_diff.T))*prior[tp]  # noqa: ERA001
                # post /= np.sum(post)  # noqa: ERA001
                # logpost = np.log(post+np.finfo(post.dtype).eps)  # noqa: ERA001

                if store_posterior:
                    posterior[t, 0, :] = logpost
                else:
                    posterior_idx[t, :] = np.argmax(logpost, axis=0)

        return posterior if store_posterior else posterior_idx


    def estimate(self, posterior, kappa_motor=None, seed=None):
        r"""Convert posterior(s) into pointing responses with motor noise applied.

        Selects the MAP template direction for each (trial, repetition) pair
        and perturbs it with isotropic motor noise drawn from
        :math:`\mathrm{vMF}(\mathbf{0}, \kappa_m)` (Eq. 7 of [barumerli2023]_).

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

    def plot_cues(self, title='', fig=None, ax=None, clim=None, elev_min=None):
        """Plot spectral cues on the median plane.

        Parameters
        ----------
        title : str, default=''
            String appended to the default subplot title.
        fig : :class:`matplotlib.figure.Figure` or None, default=None
            Existing figure to plot on.  ``None`` creates a fresh figure.
        ax : :class:`matplotlib.axes.Axes` or None, default=None
            Existing axes to plot on.  ``None`` creates fresh axes inside ``fig``.
        clim : tuple of float or None, default=None
            ``(vmin, vmax)`` for the colour scale in dB.  ``None`` uses
            ``(amps.min(), amps.max())``.
        elev_min : float or None, default=None
            Drop directions with elevation below this threshold (degrees).
            ``None`` keeps all directions on the median plane.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
        ax : :class:`matplotlib.axes.Axes`

        Raises
        ------
        ValueError
            If :attr:`target` is ``None``.
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
        """Plot the posterior distribution on the sphere with response overlay.

        Parameters
        ----------
        posterior : :class:`numpy.ndarray`
            Log-posterior of shape ``(n_templates,)``, e.g. one slice of
            :meth:`infer` output with ``store_posterior=True``.
        estimations : :class:`pyfar.Coordinates` or None
            Optional pointing-response coordinate (single direction).
            If ``None``, only the posterior is drawn.

        Returns
        -------
        ax : :class:`matplotlib.axes.Axes`
            3-D axes with the posterior scatter and overlay arrows.
        """
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

