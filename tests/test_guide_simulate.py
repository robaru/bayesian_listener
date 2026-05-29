"""Guide tests: Simulate Localization Responses (simulate_responses.rst)."""
import numpy as np


def test_simulate_full_pipeline(sofa_path):
    # [prepare]
    from bayesian_listener import BayesianListener

    listener = BayesianListener(sofa_path)
    listener.compute_template(interpolation="SHMAX")
    # [/prepare]

    # [infer]
    posterior = listener.infer(repetitions=1, prior="horizontal", seed=0)
    # posterior shape: (n_targets, repetitions) — argmax indices into template grid
    # [/infer]

    # [estimate]
    responses = listener.estimate(posterior, seed=0)
    # responses: pyfar.Coordinates, shape (n_targets, repetitions, 3)
    # [/estimate]

    # [metrics]
    from bayesian_listener.metrics import localization_error

    # Ground-truth target directions (same order as listener.target.coords)
    targets = listener.target.coords   # pyfar.Coordinates, shape (n_targets,)

    le = localization_error(targets, responses, metric="rmsL", degrees = True)
    pe = localization_error(targets, responses, metric="rmsPmedianlocal", degrees = True)
    qe = localization_error(targets, responses, metric="querrMiddlebrooks", degrees = True)
    print(f"LE={le:.1f}°  PE={pe:.1f}°  QE={qe:.1f}%")
    # [/metrics]

    assert np.isfinite(le)
    assert np.isfinite(pe)
    assert 0.0 <= qe <= 100.0
