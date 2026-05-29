"""Guide tests: How to use the model (understand.rst)."""
import numpy as np
from bayesian_listener import BayesianListener


def test_individual(sofa_path):
    # [individual]
    import pyfar as pf

    bl = BayesianListener(sofa_path)
    frontal = pf.Coordinates.from_spherical_elevation(0, 0, 1)
    estimates = bl.localise(directions=frontal, repetitions=1)
    lateral_deg = np.rad2deg(estimates.lateral.squeeze())
    polar_deg = np.rad2deg(estimates.polar.squeeze())
    print(f"\nFrontal estimate — lateral: {lateral_deg:.1f}°, polar: {polar_deg:.1f}°")
    # [/individual]
    assert estimates is not None


def test_nonindividual(sofa_path, sofa_path_non_individual):
    # [nonindividual]
    import pyfar as pf
    from bayesian_listener.metrics import angular_error

    frontal = pf.Coordinates.from_spherical_elevation(0, 0, 1)

    # Individual condition: P0001 template with P0001 target
    repetitions = 50

    bl_ind = BayesianListener(sofa_path)
    est_ind = bl_ind.localise(directions=frontal, repetitions=repetitions, seed=0)

    # Non-individual condition: P0001 template with P0002 target
    foreign = BayesianListener(sofa_path_non_individual)
    foreign.compute_target()
    bl_non = BayesianListener(sofa_path)
    est_non = bl_non.localise(target=foreign.target,
                               directions=frontal, repetitions=repetitions, seed=0)

    err_ind = np.rad2deg(angular_error(frontal.cartesian, est_ind.cartesian.squeeze())[0])
    err_non = np.rad2deg(angular_error(frontal.cartesian, est_non.cartesian.squeeze())[0])

    print(f"\nFrontal great-circle error — individual: {err_ind:.1f}°, "
          f"non-individual: {err_non:.1f}°, "
          f"difference: {err_non - err_ind:.1f}°")
    # [/nonindividual]
    assert est_ind is not None
    assert est_non is not None


def test_dynamic(sofa_path):
    # [dynamic]
    import numpy as np

    bl = BayesianListener(sofa_path)
    bl.compute_template()
    bl.compute_target()

    # Select directions on the horizontal plane, azimuth 0–90°
    sph = bl.target.coords.spherical_elevation
    azi_deg = np.rad2deg(sph[:, 0])
    ele_deg = np.rad2deg(sph[:, 1])
    mask = (ele_deg == 0) & (azi_deg >= 0) & (azi_deg <= 90)

    # Build trajectory: one single-direction target per step, sorted by azimuth
    trajectory_indices = np.where(mask)[0][np.argsort(azi_deg[mask])]
    trajectory = [bl.target[i] for i in trajectory_indices]
    actual_azimuths = azi_deg[trajectory_indices]

    prior = "horizontal"  # initialise with the default elevation prior
    estimates = []
    for step_target in trajectory:
        bl.target = step_target
        posterior = bl.infer(repetitions=1, prior=prior, store_posterior=True)
        estimates.append(bl.estimate(posterior))
        # squeeze (1, 1, n_templates) → (n_templates,) and convert to linear domain
        prior = np.exp(posterior.squeeze())

    print("\nTrajectory (actual → estimated azimuth):")
    for actual, est in zip(actual_azimuths, estimates):
        est_azi = np.rad2deg(est.spherical_elevation.squeeze()[0])
        print(f"  {actual:5.1f}° → {est_azi:5.1f}°")
    # [/dynamic]
    assert len(estimates) == len(trajectory)


def test_parameters(sofa_path):
    # [parameters]
    import pyfar as pf
    from bayesian_listener.metrics import localization_error

    frontal = pf.Coordinates.from_spherical_elevation(0, 0, 1)

    # Normal parameters with non-individual target
    bl = BayesianListener(sofa_path,
                          sigma_spectral=8.2,
                          sigma_prior=45.0,
                          kappa_motor=30.0)
    est_normal = bl.localise(directions=frontal, repetitions=50, seed=0)

    # Impaired hearing (higher spectral uncertainty)
    bl.sigma_spectral = 15.0
    est_impaired = bl.localise(directions=frontal, repetitions=50, seed=0)

    err_normal = localization_error(
        frontal, est_normal, metric='angular_error', degrees=True)
    err_impaired = localization_error(
        frontal, est_impaired, metric='angular_error', degrees=True)

    print(f"\nNon-individual frontal error — normal: {err_normal:.1f}°, "
          f"impaired: {err_impaired:.1f}°, "
          f"difference: {err_impaired - err_normal:.1f}°")
    # [/parameters]
    assert est_normal is not None
    assert est_impaired is not None
