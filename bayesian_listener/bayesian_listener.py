import sofar
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import logsumexp
from bayesian_listener import utils
from bayesian_listener import resample
from bayesian_listener.coordinates import Coordinates
from joblib import Parallel, delayed
from pathlib import Path

class BayesianListener:
    def __init__(self, sofa):
        # handle sofa input
        if isinstance(sofa, str):
            self.sofa_file = sofa
            self.sofa_data = sofar.read_sofa(sofa, verbose = False)
        elif isinstance(sofa, sofar.Sofa):
            self.sofa_file = None
            self.sofa_data = sofa
        else:
            raise ValueError('sofa must be a string containing the path to a '
                             'sofa file or a sofar.Sofa object')

        self.hrir = self.sofa_data.Data_IR
        self.fs = int(self.sofa_data.Data_SamplingRate)
        self.coords = Coordinates(sofa_file = sofa)
        # noise parameters
        self.parameters = dict(sigma_itd = .569,
                                    sigma_ild = 0.75,
                                    sigma_spectral = 4,
                                    sigma_prior = 11.5,
                                    sigma_motor = 12)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if not isinstance(value, dict):
            raise ValueError("Parameters must be a dictionary.")
        # check if all parameters are present
        for key in ['sigma_itd', 'sigma_ild', 'sigma_spectral', 'sigma_prior', 'sigma_motor']:
            if key not in value:
                raise ValueError(f"Missing parameter: {key}")
        self._parameters = value

    # interpolate to a "uniform" spherical grid coz usually HRTF do not have a uniform grid
    def interpolate(self, interpolation='SH'):
        model = BayesianListener.__new__(BayesianListener)  # Create empty instance

        # Resample all cues in a single call
        cues_list = [
            self.itd,
            self.ild,
            self.spectral_cues[:, :, 0],
            self.spectral_cues[:, :, 1]
        ]

        resampled_cues, coords_new = resample.resample(cues_list, self.coords, method=interpolation)

        # Unpack results
        model.itd = resampled_cues[0]
        model.ild = resampled_cues[1]
        model.spectral_cues = np.stack([resampled_cues[2], resampled_cues[3]], axis=-1)
        model.coords = coords_new
        model.freqs = self.freqs
        # model.coords.plot(model.spectral_cues[:, 5, 0])
        # self.coords.plot(self.spectral_cues[:, 5, 0])

        return model

    # prepare
    def prepare_features(self, spectral_range=[7e2, 18e3], interpolation='SH', use_cache=True, force_recompute=False):
        """delegates to cached or direct computation of spatial features and templates"""
        assert(self.sofa_file is not None)

        if use_cache:
            return self._load_or_compute_features(spectral_range, interpolation, force_recompute)
        else:
            return self._compute_features(spectral_range, interpolation)

    # pre-compute features for faster inference
    def _compute_features(self, spectral_range = [7e2, 18e3], interpolation='SH'):
        # normalize hrirs to frontal position
        _, idx = self.coords.find(Coordinates(positions=np.array([1, 0, 0])))
        hrirs_temp = self.hrir / np.max(np.abs(self.hrir[idx]))

        a = 32.5e-6
        b = 0.095

        # ITD
        itd = utils.itdestimator(hrirs_temp, fs=self.fs)
        self.itd = np.sign(itd) * ((np.log(a + b * np.abs(itd)) - np.log(a)) / b)

        # ILD
        self.ild = np.ones_like(self.itd)
        self.ild[:, 0] = (utils.mag2db(np.sqrt(np.mean(hrirs_temp[:, 0, :]**2, axis=1))) -
            utils.mag2db(np.sqrt(np.mean(hrirs_temp[:, 1, :]**2, axis=1))))

        # compute spatial features
        # -------- padding to account for longer filter responses --------
        pad_len_sec = 0.05  # 50 ms (same idea as the MATLAB code)
        time_len = hrirs_temp.shape[2]    # samples along time (last axis)
        dir_len  = hrirs_temp.shape[0]
        ear_len  = hrirs_temp.shape[1]
        target_samples = int(round(pad_len_sec * self.fs))

        if time_len < target_samples:
            pad_samples = target_samples - time_len
            pad_mat = np.zeros((dir_len, ear_len, pad_samples), dtype=hrirs_temp.dtype)
            hrirs_temp = np.concatenate([hrirs_temp, pad_mat], axis=2)

        # generate gammatone filterbank
        self.freqs = utils.erb_space(spectral_range)
        B, A, *_ = utils.gammatone(self.freqs, fs=self.fs)

        # Preallocate output array (float, since we take 2*real(...))
        hrirs_filt = np.zeros((len(self.freqs), *hrirs_temp.shape), dtype=float)

        # Parallel gammatone filtering
        def apply_filter(i):
            return 2 * np.real(lfilter([B[i]], A[i], hrirs_temp, axis=-1))

        # Use all available cores for parallel processing
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(apply_filter)(i) for i in range(len(self.freqs))
        )

        # results = [
        #     2 * np.real(lfilter([B[i]], A[i], hrirs_temp, axis=-1))
        #     for i in range(len(self.freqs))
        # ]
        #
        for i, result in enumerate(results):
            hrirs_filt[i] = result

        # Rectify + sqrt (rectification and compression for hair cell's firing)
        # removed because skews features and makes worse fits
        hrirs_filt = np.sqrt(np.maximum(hrirs_filt, 0))

        # average over time -> spectral amplitude
        rms = np.sqrt(np.mean(hrirs_filt**2, axis=-1))                 # (n_freqs, n_dirs, n_ears)
        spectral_amps = utils.mag2db(rms).transpose(1, 0, 2)           # -> (n_dirs, n_freqs, n_ears)

        self.spectral_cues = spectral_amps

        # prepare templates on uniform grid
        self.template = self.interpolate(interpolation)

    def _load_or_compute_features(self, spectral_range=[7e2, 18e3], interpolation='SH', force_recompute=False):
        """
        Preprocess with caching: load from cache if available, otherwise compute and save.

        Parameters
        ----------
        spectral_range : list
            Frequency range for spectral cues [low, high] in Hz
        interpolation : str
            Interpolation method to use (e.g., 'SH', 'barumerli2023')
        force_recompute : bool
            If True, ignores cache and recomputes (but still saves to cache)

        Returns
        -------
        None
            Modifies object in place
        """
        # Get path to repo root (parent of model directory)
        repo_root = Path(__file__).parent.parent
        cache_dir = repo_root / 'data' / 'preprocessed'

        # Define what attributes to cache/restore
        cache_attributes = [
            'itd', 'ild', 'freqs', 'spectral_cues',
            'coords', 'parameters', 'template'
        ]

        # ========== Try to load from cache ==========
        if not force_recompute and self.sofa_file is not None:
            cached_data = utils.load_from_cache(cache_dir, self.sofa_file, cache_attributes, interpolation)

            if cached_data is not None:
                # Restore cached attributes
                for attr in cache_attributes:
                    setattr(self, attr, cached_data[attr])
                return

            print("  Cache not found or invalid. Recomputing...")

        # ========== Compute features ==========
        print("→ Computing features...")
        self._compute_features(spectral_range=spectral_range, interpolation=interpolation)
        print("✓ Feature preparation complete")

        # ========== Save to cache ==========
        if self.sofa_file is not None:
            # Prepare data to cache
            cache_data = {attr: getattr(self, attr) for attr in cache_attributes}
            utils.save_to_cache(cache_dir, self.sofa_file, cache_data, interpolation)
        # return internal representation

    def represent(self):
        bcue = np.hstack([self.itd, self.ild])
        scue = np.hstack([self.spectral_cues[:, :, 0], self.spectral_cues[:, :, 1]])
        return np.hstack([bcue, scue])

    def infer(self, target = None, repetitions = 50, seed = None, prior = 'horizontal'):
        np.random.seed(seed)

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
        sigma = np.block(np.diag(np.hstack([sigmas["sigma_itd"]**2, sigmas["sigma_ild"]**2,
                                            np.repeat(sigmas["sigma_spectral"]**2, self.freqs.shape[0]*2)])))

        # the following code is needed to speed up multiple_logpdfs_vec_input
        # since here the covariance matrix is constant NumPy broadcasts `eigh`.
        vals, vecs = np.linalg.eigh(sigma)

        # Compute the log determinants across the second axis.
        logdet = np.sum(np.log(vals))
        # Invert the eigenvalues and add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
        Us  = vecs * np.sqrt(1./vals)[:, None]

        # Prior computation
        if isinstance(prior, str):
            if prior == 'uniform':
                # Uniform prior: all directions equally likely
                prior = np.ones(template_feat.shape[0])
            elif prior == 'horizontal':
                # Horizontal bias prior: Gaussian centered on horizontal plane (elevation = 0°)
                sph = self.template.coords.convert('spherical')
                prior = np.exp(-0.5 * (np.rad2deg(sph[:, 1]) / sigmas["sigma_prior"])**2)
            else:
                raise ValueError(f"Unknown prior: {prior}. Use 'uniform', 'horizontal', or numpy array")
            # Normalize to sum to 1 (valid probability distribution)
            prior /= np.sum(prior)
        elif isinstance(prior, np.ndarray):
            # Custom prior provided as array
            if prior.shape[0] != template_feat.shape[0]:
                raise ValueError(f"Prior shape mismatch: {prior.shape[0]} vs {template_feat.shape[0]}")
            prior = prior / np.sum(prior)  # normalize
        else:
            raise TypeError("Prior must be str ('uniform', 'horizontal') or numpy array")

        # self.template.coords.plot(prior)

        # Internal belief computation
        template_num = template_feat.shape[0]
        posterior = np.zeros((target_num, repetitions, template_num))

        if repetitions > 1:
            L = np.linalg.cholesky(sigma)  # L @ L.T = sigma
            for t in range(target_num):
                ts = np.tile(target_feat[t,:], [repetitions, 1])
                xs = ts + np.random.normal(size=ts.shape) @ L.T
                loglik = utils.multiple_logpdfs_vec_input_single_cov(xs, template_feat, logdet, Us).squeeze()
                logpost = loglik + np.log(prior)
                # normalise in log space for numerical stability
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)
                # add numerical precision to avoid underflow (i.e. prob = 0)
                # it also function as a negligible lapse rate
                logpost = np.logaddexp(logpost, np.log(np.finfo(loglik.dtype).eps))
                # normalise again (there is a better way but this is ok for now)
                logpost = logpost - logsumexp(logpost, axis=1, keepdims=True)
                posterior[t, :,:] = logpost
        else:
            for t in range(target_num):
                # for ta in range(target_num):
                # AWGN NOISE
                x = np.random.multivariate_normal(target_feat[t,:], sigma)

                # COMPUTE POSTERIOR
                # using vectorised solution
                loglik = utils.multiple_logpdfs_vec_input_single_cov(np.expand_dims(x, axis=0), template_feat, logdet, Us).squeeze()
                # post = np.exp(loglik+np.log(prior))
                logpost = loglik + np.log(prior)
                # normalise
                logpost = logpost - logsumexp(logpost)
                # add numerical precision to avoid underflow (i.e. prob = 0)
                # it also function as a negligible lapse rate
                logpost = np.logaddexp(logpost, np.log(np.finfo(loglik.dtype).eps))
                # normalise again (there is a better way but this is ok for now)
                logpost = logpost - logsumexp(logpost)

                # the solution above is faster than the for loop below but I am
                # keeping it for future reference and debugging
                # post = np.zeros(template_num)
                # for tp in range(template_num):
                #     # post[tp] = multivariate_normal.pdf(x, mean=template_feat[tp], cov=sigma) * prior[tp]
                #     # doing this speeds up stuff
                #     u_diff = (x-template_feat[tp])
                #     post[tp] = (np.exp(-0.5*u_diff @ sigma_inv @ u_diff.T))*prior[tp]
                # post /= np.sum(post)
                # logpost = np.log(post+np.finfo(post.dtype).eps)

                # Store posterior -
                posterior[t, 0,: ] = logpost

        # Results
        return posterior

    def estimate(self, posterior, sigma_motor=None):
        """
        Estimate directions from posterior distribution.

        Parameters
        ----------
        posterior : ndarray
            Posterior distribution (trials x repetitions x templates)
        sigma_motor : float or None, optional
            Motor noise standard deviation in degrees.
            If None, uses self.parameters['sigma_motor'].
            If False or 0, motor noise is disabled.

        Returns
        -------
        estimations : ndarray
            Estimated directions in Cartesian coordinates (trials x repetitions x 3)
        """
        repetitions = np.size(posterior, 1)
        trials = np.size(posterior, 0)

        assert(trials > 0)
        estimations = np.zeros((trials, repetitions, 3))
        coords_temp = self.template.coords.convert('cartesian')
        for t in range(trials):
            for r in range(repetitions):
                # Decision stage
                idx = np.argmax(posterior[t, r, :])
                estimations[t, r, :] = coords_temp[idx,:]

        # pointing error - apply only if sigma_motor is not disabled
        if sigma_motor is None:
            sigma_motor = self.parameters['sigma_motor']

        if sigma_motor not in [False, 0]:
            for rt in range(repetitions):
                estimations[:, rt, :] = utils.scatter_von_mises(estimations[:, rt, :], sigma_motor)

        return estimations

    def plot_cues(self, title='', fig=None, ax=None, clim=None, elev_min=None):
        side = 0 # left/right channel
        dirs = self.coords.sph()
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

        ax.set_xticks(np.interp([100, 1e3, 5e3, 1e4], self.freqs, np.arange(len(self.freqs))))
        ax.set_xticklabels([f'{freq:.0f}' for freq in [100, 1e3, 5e3, 1e4]])
        ax.set_yticks(np.arange(len(elevations)))
        ax.set_yticklabels([f'{elev:.0f}' for elev in elevations])

        return fig, ax

    def plot_post(self, posterior, estimations):
        amps = posterior.squeeze()
        self.template.coords.plot(np.maximum(amps, np.log(np.finfo(amps.dtype).eps)), estimations.squeeze())

def test_interp():
    # doing it from scratch
    sofa_file = 'data/P0001_FreeFieldCompMinPhase_48kHz.sofa'
    am = BayesianListener(sofa_file)
    am.prepare_features(use_cache=False, interpolation='SHMAX')

    # Get target spectral cues to compute color limits
    side = 0
    dirs = am.coords.sph()
    median_idx = np.abs(dirs[:, 0] - 0) < 2
    elevations = dirs[median_idx, 1]
    amps_target = am.spectral_cues[median_idx, :, side]
    sorted_indices = np.argsort(elevations)
    amps_target = amps_target[sorted_indices, :]

    # Compute color limits from target
    clim = (np.min(amps_target), np.max(amps_target))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot target and template features side by side with same color limits
    am.plot_cues('- Target features (i.e. original)', fig=fig, ax=axes[0], clim=clim)
    am.template.plot_cues('- Template features (i.e. interpolated)', fig=fig, ax=axes[1], clim=clim, elev_min=-45)

    plt.tight_layout()
    plt.show()




