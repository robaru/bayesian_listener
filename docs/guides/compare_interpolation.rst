.. _guide_interpolation:

.. meta::
   :keywords: HRTF, interpolation, spherical harmonics, SHMAX, barycentric,
              BIC, model selection, evaluating HRTF, individual acoustics,
              auditory model, model-based analysis, computational modelling,
              Bayes inference, spatial hearing,
              head-related transfer function, sound localization, MLE

Compare Interpolation Methods
==============================

This guide shows how to evaluate the four HRTF template interpolation methods
— ``SHMAX``, ``SH``, ``barycentric``, and ``barumerli2023`` — using
likelihood-based model comparison (BIC).  The output is a per-listener
:math:`\Delta\mathrm{BIC}` table that identifies which method best accounts
for the observed localisation behaviour.

Full-sphere coverage and high-frequency spectral fidelity are the primary
determinants of template quality; the specific algorithm is secondary once
those conditions are met (see :ref:`background_interpolation`).

.. note::

   ``barumerli2023`` retains template features only above the minimum HRTF
   measurement elevation and assigns zero probability mass to directions below
   it.  This can bias the posterior distribution for sources near the floor
   and is the reason full-sphere methods are preferred.

Fit all four methods
--------------------

Run :func:`~bayesian_listener.fitting.fit_listener` once per method and
collect the negative log-likelihood (NLL) from each result.

.. code-block:: python

   import pandas as pd
   import pyfar as pf
   import numpy as np
   from bayesian_listener.fitting import fit_listener

   obs_tbl       = pd.read_csv("responses_P0001.csv")
   targets        = obs_tbl[["azi_target", "ele_target"]].drop_duplicates()
   targets_coords = pf.Coordinates.from_spherical_elevation(
       np.deg2rad(targets["azi_target"].values),
       np.deg2rad(targets["ele_target"].values),
       np.ones(len(targets)),
   )
   sofa_path = "P0001_FreeFieldCompMinPhase_48kHz.sofa"

   methods = ["SHMAX", "SH", "barycentric", "barumerli2023"]
   results = {
       m: fit_listener(sofa_path, obs_tbl, targets_coords, interpolation_method=m)
       for m in methods
   }

Compute BIC and delta-BIC
--------------------------

BIC penalises model complexity; lower is better.  :math:`\Delta\mathrm{BIC}`
is computed relative to the best-fitting method for each listener.

.. code-block:: python

   n_trials   = results["SHMAX"]["n_trials"]
   k_params   = 2  # sigma_spectral and sigma_prior (sigma_motor fixed in stage 1)

   bic = {
       m: k_params * np.log(n_trials) + 2 * results[m]["nll"]
       for m in methods
   }
   best_bic = min(bic.values())
   delta_bic = {m: bic[m] - best_bic for m in methods}

   for m, d in sorted(delta_bic.items(), key=lambda x: x[1]):
       print(f"{m:16s}  ΔBIC = {d:.1f}")

.. note::

   :math:`|\Delta\mathrm{BIC}| > 10` is conventionally regarded as strong
   evidence in favour of the lower-BIC method (see [barumerli2026]_ in the
   :ref:`Background <background>` page).

What to do next
---------------

Use the winning interpolation method in :doc:`fit_model` for all listeners.
The :func:`~bayesian_listener.resample.resample` function exposes all four
methods and can be called directly if you need the interpolated feature arrays.
Full interpolation API is in :doc:`../api/hrtf`.
