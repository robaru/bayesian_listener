"""Guide tests: Fit the Model to Your Own HRTF (fit_model.rst)."""
import pytest
from pathlib import Path

pytestmark = pytest.mark.guide

DATA_CSV = Path(__file__).parent.parent / "data" / "responses_P0001.csv"
pytestmark = pytest.mark.skipif(
    not DATA_CSV.exists(),
    reason="Response CSV not available (data/responses_P0001.csv)",
)


def test_prepare_data():
    # [prepare]
    import pandas as pd
    import pyfar as pf
    import numpy as np

    obs_tbl = pd.read_csv(DATA_CSV)

    targets = obs_tbl[["azi_target", "ele_target"]].drop_duplicates()
    targets_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(targets["azi_target"].values),
        np.deg2rad(targets["ele_target"].values),
        np.ones(len(targets)),
    )
    # [/prepare]
    assert targets_coords is not None


def test_fit_and_inspect(sofa_path):
    import pandas as pd
    import pyfar as pf
    import numpy as np
    from bayesian_listener.fitting import fit_listener

    obs_tbl = pd.read_csv(DATA_CSV)
    targets = obs_tbl[["azi_target", "ele_target"]].drop_duplicates()
    targets_coords = pf.Coordinates.from_spherical_elevation(
        np.deg2rad(targets["azi_target"].values),
        np.deg2rad(targets["ele_target"].values),
        np.ones(len(targets)),
    )

    # [fit]
    result = fit_listener(
        sofa_path=sofa_path,
        obs_tbl=obs_tbl,
        num_repetitions=1,
        targets_coords=targets_coords,
        interpolation_method="SHMAX",
    )

    print(f"sigma_motor   = {result['sigma_motor']:.2f} deg")
    print(f"sigma_spectral = {result['sigma_spectral']:.2f} dB")
    print(f"sigma_prior   = {result['sigma_prior']:.2f} deg")
    print(f"NLL           = {result['nll']:.2f}")
    # [/fit]

    # [inspect]
    print(result["sigma_itd"])    # 0.569 (ITD noise, fixed)
    print(result["sigma_ild"])    # 1.0   (ILD noise, fixed by default)
    print(result["sigma_spectral"])
    print(result["sigma_prior"])
    print(result["kappa_motor"])  # concentration form of sigma_motor
    print(result["n_trials"])     # number of responses used
    print(result["time_total"])   # wall-clock seconds
    # [/inspect]

    assert result["success"]
