"""Spatial resampling of ITD, ILD, and spectral cues onto a quasi-uniform grid.

Four interpolation methods are exposed via :func:`resample`:

- ``'SH'``      — regularised spherical-harmonic (SH) interpolation at
  the maximum stable order for the input grid.
- ``'SHMAX'``   — SH interpolation at fixed order 44 with Tikhonov
  regularisation (Bau damping, [bau2022]_).
- ``'barycentric'`` — VBAP weights on the convex hull of the sampling grid
  ([pulkki1997]_).
- ``'barumerli2023'`` — order-15 SH interpolation from [barumerli2023]_;
  retained for backward compatibility.

Methods are compared on the SONICOM dataset in [barumerli2026]_, §2.5.
"""
import numpy as np
import spharpy as sy
import spaudiopy
import pyfar as pf
import warnings
from bayesian_listener import utils

# -----------------------------------------------------------------------------
# SPHERICAL INTERPOLATION
# -----------------------------------------------------------------------------

# helpers
def build_Y(dirs, N):
    r"""Build the matrix of real spherical-harmonic basis functions at each direction.

    Parameters
    ----------
    dirs : :class:`numpy.ndarray`
        Direction array of shape ``(n_dirs, 2)`` containing
        ``(azimuth, elevation)`` in **radians**.
    N : int
        SH order; the basis has :math:`(N+1)^2` columns.

    Returns
    -------
    :class:`numpy.ndarray`
        Real SH basis :math:`\mathbf{Y}` of shape
        ``(n_dirs, (N + 1) ** 2)``.
    """
    Y = sy.sph.sh_matrix(N, dirs[:, 0], dirs[:, 1], sh_type='real')
    return Y

def build_bau_damping(N):
    r"""Bau et al. damping matrix for Tikhonov-regularised SH inversion.

    Builds the diagonal damping
    :math:`D_{ii} = 1 + n(n+1)` indexed in SH order :math:`n` (each :math:`n`
    repeats :math:`2n+1` times).

    Parameters
    ----------
    N : int
        Maximum SH order.

    Returns
    -------
    :class:`numpy.ndarray`
        Diagonal damping matrix of shape ``((N + 1) ** 2, (N + 1) ** 2)``.

    References
    ----------
    [bau2022]_.
    """
    num_coeffs = (N + 1) ** 2
    D = np.zeros((num_coeffs, num_coeffs))
    idx = 0
    for n in range(N + 1):
        for _ in range(-n, n + 1):
            D[idx, idx] = 1 + n * (n + 1)
            idx += 1
    return D

def find_max_order(dirs,
                   condition_threshold=12.25,
                   N_max=35,
                   regularised=True,
                   regularisation_coefficient=1e-2):
    r"""Return the largest SH order :math:`N \le N_{\max}` with stable conditioning.

    Iterates from :math:`N = 1` upward and returns the highest order whose
    (optionally Tikhonov-regularised) Gram matrix
    :math:`\mathbf{Y}^\top\mathbf{Y}` has condition number below
    ``condition_threshold``.

    Parameters
    ----------
    dirs : :class:`pyfar.Coordinates`
        Sampling grid.
    condition_threshold : float, default=12.25
        Upper bound on :math:`\kappa(\mathbf{Y}^\top\mathbf{Y})`.  The default
        follows [bau2022]_, equivalent to :math:`\kappa(\mathbf{Y}) < 3.5`
        ([benhur2019]_).
    N_max : int, default=35
        Maximum SH order to test.
    regularised : bool, default=True
        If ``True``, add the Bau damping matrix
        :func:`build_bau_damping` scaled by ``regularisation_coefficient``
        before computing the condition number.
    regularisation_coefficient : float, default=1e-2
        Tikhonov weight applied to the damping matrix.

    Returns
    -------
    int
        Largest admissible SH order.  Returns ``1`` if no order satisfies
        the threshold.
    """
    Y = sy.spherical.spherical_harmonic_basis_real(N_max, dirs)

    for N in range(1, N_max+1):
        Y_N = Y[:, :(N+1)**2]
        YtY = Y_N.T @ Y_N

        D_bau = build_bau_damping(N)

        if regularised:
            YY = YtY + regularisation_coefficient * D_bau
        else:
            YY = YtY

        if np.linalg.cond(YY) > condition_threshold:
                return N - 1
    return N_max

def solve_sh(Y, H):
    r"""Solve :math:`\mathbf{Y}\mathbf{C} \approx \mathbf{H}` by Moore–Penrose pseudo-inverse.

    Non-regularised least-squares.  Use :func:`build_bau_damping` and a
    direct solve when regularisation is required (as in :func:`resample_two_step`).

    Parameters
    ----------
    Y : :class:`numpy.ndarray`
        SH basis matrix of shape ``(n_dirs, (N + 1) ** 2)``.
    H : :class:`numpy.ndarray`
        Cue matrix of shape ``(n_dirs, n_features)``.

    Returns
    -------
    :class:`numpy.ndarray`
        Coefficient matrix of shape ``((N + 1) ** 2, n_features)``.
    """
    return np.linalg.pinv(Y) @ H

def interpolate_HRTF(query_dirs, C, N):
    r"""Evaluate an SH expansion :math:`\mathbf{Y}(\mathbf{q})\mathbf{C}` at query directions.

    Parameters
    ----------
    query_dirs : :class:`numpy.ndarray`
        Query directions of shape ``(n_query, 2)`` containing
        ``(azimuth, elevation)`` in radians.
    C : :class:`numpy.ndarray`
        SH coefficient matrix of shape ``((N + 1) ** 2, n_features)``.
    N : int
        SH order matching the columns of ``C``.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolated cues of shape ``(n_query, n_features)``.
    """
    Yq = build_Y(query_dirs, N)
    return Yq @ C

# method from Fabian in test_resampling
def complement_sampling(coordinates):
    """
    Complement sampling grid.

    The sampling grid is complemented by detecting the minimum elevation and
    adding points below. For example, if the minimum elevation is -30 degree,
    the grid is complemented by finding sampling points above 30 degree and
    mirroring them downward by flipping the sign of their elevation angles.

    Note this method works for Gaussian-like sampling grids but might not
    work well in other cases.

    Parameters
    ----------
    coordinates : pyfar.Coordinates
        The incomplete sampling grid.

    Returns
    -------
    complemented : :class:`pyfar.Coordinates`
        The complemented sampling grid.
    mask : :class:`numpy.ndarray`
        Boolean array of shape ``(complemented.csize,)``; ``True`` at
        positions corresponding to mirrored (complemented) points and
        ``False`` for the original measurement directions.
    """

    # detect and check minimum elevation
    min_elevation = np.min(coordinates.elevation)
    if min_elevation > 0:
        warnings.warn(
            'Detected positive minimum elevation during resampling.'
            'Manual resampling might be required',
            stacklevel=2)

    # find and add mirror points with added 0.1 degree safety margin
    mask = coordinates.elevation > -min_elevation + 0.0017
    complement = coordinates.spherical_elevation[mask]
    complement[:, 1] *= -1
    complemented = coordinates.copy()
    complemented.spherical_elevation = np.concatenate(
        (complemented.spherical_elevation, complement), axis=0)

    # mask for selecting complemented points
    mask = np.zeros(complemented.csize, dtype=bool)
    mask[coordinates.csize:] = True

    return complemented, mask

# method from Fabian in test_resampling
def resample_two_step(cues, coordinates, template, second_step, **kwargs):
    """Resample localisation cues using the two-step procedure of [ahrens2012]_.

    Stage 1: low-order SH extrapolation completes missing low-elevation
    directions by mirroring measured points across the horizontal plane
    (see :func:`complement_sampling`).  Stage 2: high-order interpolation
    of the complemented cues onto ``template``.

    Parameters
    ----------
    cues : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Cues as a single array or a list of arrays.  For each array,
        ``shape[-2]`` must equal the number of source positions in
        ``coordinates``.  When a list is passed, the result is also a list
        in the same order.
    coordinates : :class:`pyfar.Coordinates`
        Source coordinates corresponding to the cue rows.
    template : :class:`pyfar.Coordinates` or None, default=None
        Output directions.  ``None`` selects a 64th-degree spherical
        t-design (2,112 directions).
    second_step : {'SH', 'SHMAX', 'barycentric'}
        Stage-2 interpolator (case-insensitive).

        - ``'SH'``: regularised SH interpolation at the maximum stable order
          for ``coordinates_complemented``.
        - ``'SHMAX'``: regularised SH interpolation at fixed order 44.
        - ``'barycentric'``: VBAP weights on the convex hull.
    **kwargs
        Forwarded options:

        - ``regularisation_coefficient`` (float, default 1e-2) — Tikhonov
          weight on the Bau damping matrix.
        - ``condition_threshold`` (float, default 12.25) — condition-number
          bound forwarded to :func:`find_max_order`.
        - ``norm`` (``{1, 2}``, default 1) — gain normalisation for
          ``'barycentric'``; see :func:`~bayesian_listener.utils.vbap_interpolate`.

    Returns
    -------
    cues : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Resampled cues.  ``shape[-2]`` of each array equals
        ``template.csize``.  Same container type as the input.
    template_coords : :class:`pyfar.Coordinates`
        Output directions.

    Raises
    ------
    TypeError
        If ``coordinates`` or ``template`` is not :class:`pyfar.Coordinates`.
    ValueError
        If ``second_step`` is not one of the three accepted values.
    """
    regularisation_coefficient = kwargs.get('regularisation_coefficient', 1e-2)
    condition_threshold = kwargs.get('condition_threshold', 12.25)
    norm = kwargs.get('norm', 1)

    # check input format
    if not isinstance(cues, (list, tuple)):
        cues = [cues]
        passed_list = False
    else:
        passed_list = True

    # Check if coordinates is a pf.Coordinates object
    if not isinstance(coordinates, pf.Coordinates):
        raise TypeError('`coordinates` must be a pyfar.Coordinates object')

    # Check if template is a pf.Coordinates object or `None
    if template is not None and not isinstance(template, pf.Coordinates):
        raise TypeError(
            '`template` must be a pyfar.Coordinates object or `None`')

    if template is None:
        # %% Generate t-design points
        # the advantage of using this is that
        # the weights are equal when integrating
        template = utils.load_n_design(64)  # 2112 equally distant points
        template = pf.Coordinates.from_cartesian(template[:, 0],
                                                 template[:, 1],
                                                 template[:, 2])

    # HRTF measurement grids usually lack the bottom, so we add it
    coordinates_complemented, mask = complement_sampling(coordinates)

    # perform first interpolation step only if grid could be complemented
    if np.any(mask):
        # low order SH transform
        n_max = find_max_order(coordinates, N_max=5, condition_threshold=condition_threshold)
        print("Low order {}".format(n_max))
        Y = sy.spherical.spherical_harmonic_basis_real(n_max, coordinates)
        Y_inv = np.linalg.pinv(Y)
        cues_low = [Y_inv @ c for c in cues]

        # low order inverse SH transform to complemented grid
        Y_complemented = sy.spherical.spherical_harmonic_basis_real(
            n_max, coordinates_complemented[mask])
        cues_low = [np.matmul(Y_complemented, c) for c in cues_low]

        cues = [np.concat((c, c_low), -2) for c, c_low in zip(cues, cues_low)]

    # perform second interpolation step
    if second_step.lower() == 'barycentric':
        # compute interpolation weights
        weights = utils.vbap_interpolate(template.cartesian,
                                         coordinates_complemented.cartesian,
                                         norm=norm)

        # apply as matrix multiplication
        cues = [weights @ c for c in cues]

    elif second_step.lower() == 'sh':
        # high(er) order SH transform
        n_max = find_max_order(coordinates_complemented, condition_threshold=condition_threshold)
        print("High order {}".format(n_max))
        Y = sy.spherical.spherical_harmonic_basis_real(
            n_max, coordinates_complemented)

        # Apply Tikhonov regularization with Bau damping
        D_bau = build_bau_damping(n_max)
        YtY = Y.T @ Y
        Y_inv = np.linalg.solve(YtY + regularisation_coefficient * D_bau, Y.T)
        cues = [Y_inv @ c for c in cues]

        # high(er) order inverse SH transform to template grid
        Y_template = sy.spherical.spherical_harmonic_basis_real(
            n_max, template)
        cues = [np.matmul(Y_template, c) for c in cues]

    elif second_step.lower() == 'shmax':
        # high(er) order SH transform
        n_max = 44
        print("High order {}".format(n_max))
        Y = sy.spherical.spherical_harmonic_basis_real(
            n_max, coordinates_complemented)

        # Apply Tikhonov regularization with Bau damping
        D_bau = build_bau_damping(n_max)
        YtY = Y.T @ Y
        Y_inv = np.linalg.solve(YtY + regularisation_coefficient * D_bau, Y.T)
        cues = [Y_inv @ c for c in cues]

        # high(er) order inverse SH transform to template grid
        Y_template = sy.spherical.spherical_harmonic_basis_real(
            n_max, template)
        cues = [np.matmul(Y_template, c) for c in cues]

    else:
        raise ValueError(
            "second_step must be 'SH', 'SHMAX', or 'barycentric' "
            "(case-insensitive)")

    if not passed_list:
        cues = cues[0]

    return cues, template

# as in barumerli2023
def resample_barumerli2023(values,
                           coords_in,
                           template=None,
                           flag_regularisation = True):
    """Resample with order-15 SH interpolation, as in [barumerli2023]_.

    Single-step SH interpolation at order :math:`N = 15` with optional
    Tikhonov regularisation.  Retained for backward compatibility with the
    original MATLAB implementation; assigns no probability mass to
    directions below the lowest measured elevation (see :ref:`background_limitations`).

    Parameters
    ----------
    values : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Single cue of shape ``(n_dirs, ...)`` or a list of cues whose first
        dimension matches.
    coords_in : :class:`pyfar.Coordinates`
        Source coordinates of the input cues.
    template : :class:`pyfar.Coordinates` or None, default=None
        Output directions.  ``None`` selects a 64th-degree spherical t-design.
    flag_regularisation : bool, default=True
        If ``True``, apply a fixed Tikhonov regulariser
        (:math:`\\lambda = 4`) ignoring the first three SH orders.

    Returns
    -------
    values_out : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Resampled cues; same container type as ``values``.
    template_coords : :class:`pyfar.Coordinates`
        Output directions.
    """
    N_sph = 15

    # Check if input is a list of cues
    if isinstance(values, (list, tuple)):
        cues = values
        passed_list = True
    else:
        cues = [values]
        passed_list = False

    # Check if coordinates is a pf.Coordinates object
    if not isinstance(coords_in, pf.Coordinates):
        raise TypeError('`coordinates` must be a pyfar.Coordinates object')

    # Check if template is a pf.Coordinates object or `None``
    if template is not None and not isinstance(template, pf.Coordinates):
        raise TypeError(
            '`template` must be a pyfar.Coordinates object or `None`')

    if template is None:
        # %% Generate t-design points
        # the advantage of using this is that
        # the weights are equal when integrating
        dirs = utils.load_n_design(64)  # 2112 equally distant points
        template = pf.Coordinates.from_cartesian(dirs[:, 0],
                                                 dirs[:, 1],
                                                 dirs[:, 2])

    dirs_sph = template.spherical_elevation
    azimuth = dirs_sph[:, 0]
    colatidude = dirs_sph[:, 1]

    # assert(N_SH < N_dirs, ...
    #     ['Spherical harmonics: beware that the number of provided ',...
    #     'coordinates is too low to obtain a precise interpolation'])

    dirs_SH = np.transpose([azimuth, colatidude])

    # transform signal to SH domain
    c = coords_in.spherical_colatitude
    azi = c[..., 0]
    zen = c[..., 1]

    # get SH basis on new directions
    int_new = spaudiopy.sph.sh_matrix(N_sph, dirs_SH[:, 0],
                                      dirs_SH[:, 1],
                                      sh_type='real')

    # Ensure all cues are at least 2D
    cues = [c[:, np.newaxis] if c.ndim == 1 else c for c in cues]

    if not flag_regularisation:
        # get SH matrix for input positions and transform to SH domain
        cues_SH = [spaudiopy.sph.sht(c, N_sph, azi, zen, 'real') for c in cues]
    else:
        # regularization
        lambda_val = 4.0
        SIG = np.eye((N_sph+1)**2)
        SIG[1:(2+1)**2,1:(2+1)**2] = 0

        # get SH basis on old directions
        Y_N_tik = spaudiopy.sph.sh_matrix(N_sph, azi, zen, 'real')
        # Compute regularized inverse once
        Y_inv_reg = np.linalg.solve(
            np.transpose(Y_N_tik)@Y_N_tik+lambda_val*SIG,
            np.transpose(Y_N_tik))
        # Transform all cues to SH domain
        cues_SH = [Y_inv_reg @ c for c in cues]

    # interpolate all cues
    cues_out = [int_new @ c_SH for c_SH in cues_SH]

    # remove bottom as done in AMT model
    # commented out otherwise fitting does not work
    # idx = dirs[:, 2] > -.5
    # dirs = dirs[idx, :]
    # cues_out = [c[idx,:] for c in cues_out]

    # Return in same format as input
    if not passed_list:
        cues_out = cues_out[0]

    return cues_out, template

def resample(cues, coordinates, template=None, method='SH', **kwargs):
    """Unified entry point for the four resampling methods.

    Parameters
    ----------
    cues : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Single cue of shape ``(n_dirs, ...)`` or a list of cues whose first
        dimension matches.
    coordinates : :class:`pyfar.Coordinates`
        Source coordinates of the input cues.
    template : :class:`pyfar.Coordinates` or None, default=None
        Output directions.  ``None`` selects a 64th-degree spherical t-design
        for every method.
    method : {'SH', 'SHMAX', 'barycentric', 'barumerli2023'}, default='SH'
        Resampling method (case-insensitive).
    **kwargs
        Forwarded to :func:`resample_two_step` for ``'SH'``, ``'SHMAX'``,
        and ``'barycentric'``: ``regularisation_coefficient``,
        ``condition_threshold``, ``norm``.

    Returns
    -------
    result : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
        Resampled cues with first dimension equal to ``template.csize``;
        same container type as ``cues``.
    template_coords : :class:`pyfar.Coordinates`
        Output directions.

    Raises
    ------
    ValueError
        If ``method`` is none of the four accepted values.
    """
    if method.lower() == 'barycentric':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'barycentric',
                                                    **kwargs)
    elif method.lower() == 'sh':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'sh', **kwargs)
    elif method.lower() == 'shmax':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'shmax', **kwargs)
    elif method.lower() == 'barumerli2023':
        result, template_coords = resample_barumerli2023(cues,
                                                         coordinates,
                                                         template)
    else:
        raise ValueError(f"Unknown resample method: {method}")

    return result, template_coords

def plot_resampling_grid(coords_meas_cart,
                         dirs_virt,
                         missing_mask,
                         z_min_meas):
    """Plot measured (black) and added (red) directions on a 3-D + 2-D figure.

    Parameters
    ----------
    coords_meas_cart : :class:`numpy.ndarray`
        Measured directions in Cartesian coordinates, shape ``(n_meas, 3)``.
    dirs_virt : :class:`numpy.ndarray`
        Virtual grid directions in Cartesian coordinates, shape
        ``(n_virt, 3)``.
    missing_mask : :class:`numpy.ndarray`
        Boolean mask of shape ``(n_virt,)`` selecting which virtual
        directions were added (i.e. fall in the unmeasured region).
    z_min_meas : float
        Minimum z-coordinate of the measured directions; the horizontal
        plane at this height is drawn as a translucent reference surface.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot measured directions (black)
    ax1.scatter(coords_meas_cart[:, 0],
                coords_meas_cart[:, 1],
                coords_meas_cart[:, 2],
                c='black',
                s=20,
                alpha=0.6,
                label=f'Measured (n={len(coords_meas_cart)})',
                )

    # Plot added directions (red)
    dirs_added = dirs_virt[missing_mask]
    ax1.scatter(dirs_added[:, 0],
                dirs_added[:, 1],
                dirs_added[:, 2],
                c='red',
                s=20,
                alpha=0.6,
                label=f'Added (n={np.sum(missing_mask)})',
                )

    # Draw horizontal plane at z_min_meas
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    zz = np.ones_like(xx) * z_min_meas
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='blue')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Resampling Grid')
    ax1.legend()
    ax1.set_box_aspect([1,1,1])

    # 2D projection (top view)
    ax2 = fig.add_subplot(122)
    cm = pf.Coordinates(coords_meas_cart[:, 0], coords_meas_cart[:, 1], coords_meas_cart[:, 2])
    cd = pf.Coordinates(dirs_added[:, 0], dirs_added[:, 1], dirs_added[:, 2])

    ax2.scatter(cm.spherical_elevation[:, 0],
                cm.spherical_elevation[:, 1],
                c='black',
                s=20,
                alpha=0.6,
                label=f'Measured (n={len(coords_meas_cart)})',
                )
    ax2.scatter(cd.spherical_elevation[:, 0],
                cd.spherical_elevation[:, 1],
                c='red',
                s=20, alpha=0.6,
                label=f'Added (n={np.sum(missing_mask)})',
                )
    ax2.set_xlabel('Azimuth (rad)')
    ax2.set_ylabel('Elevation (rad)')
    ax2.set_title('Top View: Resampling Grid')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Grid statistics:")
    print(f"  Measured directions: {len(coords_meas_cart)}")
    print(f"  Added directions (z < {z_min_meas:.3f}): {np.sum(missing_mask)}")
    print(f"  Total directions for interpolation: "
          f"{len(coords_meas_cart) + np.sum(missing_mask)}")
