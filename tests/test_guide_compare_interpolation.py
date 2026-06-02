"""Guide tests: Compare Interpolation Methods (compare_interpolation.rst)."""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.guide

DATA_CSV = Path(__file__).parent.parent / "data" / "responses_P0001.csv"
pytestmark = pytest.mark.skipif(
    not DATA_CSV.exists(),
    reason="Response CSV not available (data/responses_P0001.csv)",
)


def test_compare_interpolation(sofa_path):
    # [fit_methods]
    import pandas as pd
    import pyfar as pf
    from bayesian_listener.fitting import fit_listener, fit_listener_partial

    obs_tbl       = pd.read_csv(DATA_CSV)
    targets        = obs_tbl[["azi_target", "ele_target"]].drop_duplicates()
    targets_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(targets["azi_target"].values),
        np.deg2rad(targets["ele_target"].values),
        np.ones(len(targets)),
    )

    results = fit_listener(
        sofa_path, obs_tbl, targets_coords,
        interpolation_method="SHMAX", num_repetitions=1)

    # fit only sigma_spectral (i.e. motor noise and prior fixed to default)
    results_noprior = fit_listener_partial(
        sofa_path, obs_tbl, targets_coords,
        interpolation_method="SHMAX",
        params_to_fit=["sigma_spectral"],
        num_repetitions=1,
        verbose=False,
    )
    # [/fit_methods]

    # [bic]
    n_trials  = results["n_trials"]
    k_full    = 2  # sigma_spectral + sigma_prior (kappa_motor fixed in stage 1)
    k_noprior = 1  # sigma_spectral only

    bic_full = k_full * np.log(n_trials) + 2 * results["nll"]

    bic_noprior = k_noprior * np.log(n_trials) + 2 * results_noprior["nll"]

    delta_bic = bic_noprior - bic_full
    print(f"\nBIC (with prior): {bic_full:.1f}  BIC (no prior): {bic_noprior:.1f}  "
          f"ΔBIC = {delta_bic:.1f}")
    # [/bic]

    assert np.isfinite(bic_full)
    assert np.isfinite(bic_noprior)
