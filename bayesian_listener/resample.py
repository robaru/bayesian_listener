"""
This module contains functions to spatially resample ITD, ILD and spectral
cues.
"""
import numpy as np
import spharpy as sy
import spaudiopy
import pyfar as pf
import warnings

# -----------------------------------------------------------------------------
# SPHERICAL INTERPOLATION
# -----------------------------------------------------------------------------

# helpers
def build_Y(dirs, N):
    """
    Build the (len(dirs) × (N+1)^2) matrix of real SH basis functions
    evaluated at each direction in `dirs`.
    """
    Y = sy.sph.sh_matrix(N, dirs[:, 0], dirs[:, 1], sh_type='real')
    return Y

def build_bau_damping(N):
    """Bau et al. damping: D_ii = 1 + n(n+1)."""
    num_coeffs = (N + 1) ** 2
    D = np.zeros((num_coeffs, num_coeffs))
    idx = 0
    for n in range(N + 1):
        for _ in range(-n, n + 1):
            D[idx, idx] = 1 + n * (n + 1)
            idx += 1
    return D

def find_max_order(dirs,
                   thresh=12.25,
                   N_max=35,
                   regularised=True,
                   epsilon=1e-2):
    """Return the largest SH order N ≤ N_max with acceptable conditioning.

    Iterates from N=1 upward and returns the highest order whose
    (optionally regularised) Gram matrix ``Y^T Y`` has a condition number
    below *thresh*.

    Parameters
    ----------
    dirs : pyfar.Coordinates
        The sampling grid.
    thresh : float
        Upper bound on κ(Y^T Y).  Default 12.25 follows Bau et al. (2022),
        corresponding to κ(Y) < 3.5 (Ben-Hur et al., 2019).
    N_max : int
        Maximum SH order to consider.
    regularised : bool
        If True, apply Tikhonov regularisation with Bau damping matrix
        before evaluating the condition number.
    epsilon : float
        Regularisation weight (Bau et al., 2022, use 10e-2 = 0.1;
        current default is 1e-2 — see issue #37).

    Returns
    -------
    N : int
        Largest admissible SH order (1-based). Returns 1 if no order
        satisfies the threshold.
    """
    Y = sy.spherical.spherical_harmonic_basis_real(N_max, dirs)

    for N in range(1, N_max+1):
        Y_N = Y[:, :(N+1)**2]
        YtY = Y_N.T @ Y_N

        D_bau = build_bau_damping(N)

        if regularised:
            YY = YtY + epsilon * D_bau
        else:
            YY = YtY

        if np.linalg.cond(YY) > thresh:
                return N - 1
    return N_max

def solve_sh(Y, H):
    """
    Non-regularized least-squares: return coefficient matrix C so that Y·C ≈ H.
    H is (len(dirs) × F), C is ((N+1)^2 × F).
    """
    return np.linalg.pinv(Y) @ H

def interpolate_HRTF(query_dirs, C, N):
    """
    query_dirs: (Q,2) array of (az,el).
    returns: (Q,F) matrix of interpolated HRTF magnitudes.
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
    complemented : pyfar.Coordinates
        The complemented sampling grid.
    mask : np.ndarray
        Boolean array. ``True`` at the position of complemented points
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
def resample_two_step(cues, coordinates, template, second_step):
    """
    Resample localization cues.

    Parameters
    ----------
    cues : array, list of arrays
        Cues as an array or list of arrays. For each array, ``shape[-2]`` must
        equal the number of source positions in `coordinates`.
    coordinates : pyfar.Coordinates
        Coordinates of the cues
    template : pyfar.Coordinates or `None`, optional
        Coordinates to which the cues are interpolated to. If `None` (default),
        uses spherical n-design of 64th degree.
    second_step : string
        'SH' or 'Barycentric' (case insensitive)

    Returns
    -------
    cues : array, list of arrays
        Resampled cues. For each array, ``shape[-2]`` equal the number of
        source positions in `template`.
    template_coords : pyfar.Coordinates
        Output coordinates of resampled cues.
    """

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
        template = spaudiopy.grids.load_n_design(64)# 2112 equally distant points
        template = pf.Coordinates.from_cartesian(template[:, 0],
                                                 template[:, 1],
                                                 template[:, 2])

    # HRTF measurement grids usually lack the bottom, so we add it
    coordinates_complemented, mask = complement_sampling(coordinates)

    # perform first interpolation step only if grid could be complemented
    if np.any(mask):
        # low order SH transform
        n_max = find_max_order(coordinates, N_max=5)
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
        convex_hull = spaudiopy.decoder.LoudspeakerSetup(
            coordinates_complemented.x,
            coordinates_complemented.y,
            coordinates_complemented.z)
        weights = spaudiopy.decoder.vbap(
            template.cartesian, convex_hull, norm=1)

        # apply as matrix multiplication
        cues = [weights @ c for c in cues]

    elif second_step.lower() == 'sh':
        # high(er) order SH transform
        n_max = find_max_order(coordinates_complemented)
        print("High order {}".format(n_max))
        Y = sy.spherical.spherical_harmonic_basis_real(
            n_max, coordinates_complemented)

        # Apply Tikhonov regularization with Bau damping
        epsilon = 1e-2
        D_bau = build_bau_damping(n_max)
        YtY = Y.T @ Y
        Y_inv = np.linalg.solve(YtY + epsilon * D_bau, Y.T)
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
        epsilon = 1e-2
        D_bau = build_bau_damping(n_max)
        YtY = Y.T @ Y
        Y_inv = np.linalg.solve(YtY + epsilon * D_bau, Y.T)
        cues = [Y_inv @ c for c in cues]

        # high(er) order inverse SH transform to template grid
        Y_template = sy.spherical.spherical_harmonic_basis_real(
            n_max, template)
        cues = [np.matmul(Y_template, c) for c in cues]

    else:
        raise ValueError("second step must be 'barycentric' or 'sh'")

    if not passed_list:
        cues = cues[0]

    return cues, template

# as in barumerli2023
def resample_barumerli2023(values,
                           coords_in,
                           template=None,
                           flag_regularisation = True):
    """
    Resample using spherical harmonics as in Barumerli et al. 2023.

    Parameters
    ----------
    values : array or list of arrays
        Single cue array of shape (n_dirs, ...) or list of cue arrays.
        If list, each array must have shape (n_dirs, ...)
        where first dimension matches.
    coords_in : pyfar.Coordinates
        Source coordinates
    template : pyfar.Coordinates or `None`, optional
        Coordinates to which the cues are interpolated to. If `None` (default),
        uses spherical n-design of 64th degree.
    flag_regularisation : bool
        Whether to use Tikhonov regularization

    Returns
    -------
    values_out : array or list of arrays
        Resampled cues. Returns same type (single array or list) as input.
    template_coords : pyfar.Coordinates
        Output coordinates of resampled cues.
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
        dirs = spaudiopy.grids.load_n_design(64)# 2112 equally distant points
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

def resample(cues, coordinates, template=None, method='SH'):
    """
    Unified resample interface that handles both single and multiple cues.

    Parameters
    ----------
    cues : array or list of arrays
        Single cue array of shape (n_dirs, ...) or list of cue arrays.
        If list, each array must have shape (n_dirs, ...)
        where first dimension matches.
    coordinates : pf.Coordinates
        Source coordinates
    termplate : pyfar.Coordinates or `None`, optional
        Coordinates to which the cues are interpolated to. If `None` (default),
        uses 64th degree spherical n-design for methods 'SH', 'barycentric', and
        'barumerli2023'.
    method : str
        Resampling method: 'SH', 'barycentric', or 'barumerli2023'

    Returns
    -------
    result : array or list of arrays
        Resampled cues. Returns same type (single array or list) as input.
        Each array has shape (n_template_dirs, ...)
        matching input except first dimension.
    template_coords : pyfar.Coordinates
        Output coordinates
    """
    if method.lower() == 'barycentric':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'barycentric')
    elif method.lower() == 'sh':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'sh')
    elif method.lower() == 'shmax':
        result, template_coords = resample_two_step(cues, coordinates,
                                                    template, 'shmax')
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
    """
    Plot the resampling grid showing measured directions (black)
    and added directions (red).

    Parameters
    ----------
    coords_meas_cart : ndarray, shape (M, 3)
        Measured directions in Cartesian coordinates
    dirs_virt : ndarray, shape (V, 3)
        Virtual grid directions in Cartesian coordinates
    missing_mask : ndarray, shape (V,)
        Boolean mask indicating which virtual directions were added
    z_min_meas : float
        Minimum z-coordinate of measured directions
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
