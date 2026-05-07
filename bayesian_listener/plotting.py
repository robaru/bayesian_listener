"""Plotting functions for the BayesianListener class."""
import numpy as np
import matplotlib.pyplot as plt
from bayesian_listener import BayesianListener

def plot_cues(bl: BayesianListener, title='', fig=None, ax=None, clim=None, elev_min=None):
    """Plot spectral cues on the median plane.

    Parameters
    ----------
    bl : :class:`~bayesian_listener.BayesianListener`
        Listener instance with :attr:`target` already set.
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
    if bl.target is None:
        raise ValueError(
            'Target not set. Call compute_target() before plot_cues().')
    side = 0 # left/right channel
    dirs = bl.coords.spherical_elevation
    dirs[:, 0:2] = np.rad2deg(dirs[:, 0:2])
    # select directions with azimuth almost zero (median frontal plane)
    median_idx = np.abs(dirs[:, 0] - 0) < 2
    elevations = dirs[median_idx,1]
    amps = bl.target.spectral_cues[median_idx, :, side]

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
                            bl.target.freqs,
                            np.arange(len(bl.target.freqs))))
    ax.set_xticklabels([f'{freq:.0f}' for freq in [100, 1e3, 5e3, 1e4]])
    ax.set_yticks(np.arange(len(elevations)))
    ax.set_yticklabels([f'{elev:.0f}' for elev in elevations])
    plt.show()

    return fig, ax

def plot_post(bl: BayesianListener, posterior, estimations):
    """Plot the posterior distribution on the sphere with response overlay.

    Parameters
    ----------
    bl : :class:`~bayesian_listener.BayesianListener`
        Listener instance with :attr:`template` already set.
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

    ax = bl.template.coords.show(
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

