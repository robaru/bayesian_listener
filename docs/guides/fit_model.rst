.. _guide_fit_model:

.. meta::
   :keywords: HRTF, MLE, fitting, Bayesian, motor noise, sigma, two-stage,
              auditory model, model-based analysis, computational modelling,
              evaluating HRTF, individual acoustics, Bayes inference,
              spatial hearing, head-related transfer function,
              psychoacoustics, sound localization

Fit the Model to Your Own HRTF
================================

This guide shows how to estimate listener-specific model parameters from
measured sound-localisation responses.  The output is a dictionary of fitted
noise parameters — :attr:`~bayesian_listener.BayesianListener.sigma_spectral`,
:attr:`~bayesian_listener.BayesianListener.sigma_prior`, and
:attr:`~bayesian_listener.BayesianListener.sigma_motor` — that characterise
that listener's sensory precision and spatial expectations.

The fitting procedure follows a two-stage profile-likelihood approach
(see :ref:`background_likelihood` for the equations):

1. **Stage 1 (lateral only):** estimate motor noise :math:`\sigma_\mathrm{m}`
   from lateral responses using ITD and ILD cues, which dominate horizontal
   localisation and are largely independent of spectral processing.
2. **Stage 2 (full sphere):** fix :math:`\sigma_\mathrm{m}` and fit
   :math:`\sigma_\mathrm{mon}` (spectral noise) and :math:`\sigma_\mathrm{prior}`
   (prior width) by maximising the full-sphere likelihood.

.. note::

   :attr:`~bayesian_listener.BayesianListener.sigma_itd` (0.569) and
   :attr:`~bayesian_listener.BayesianListener.sigma_ild` (1.0 dB) are fixed
   at literature values throughout — they cannot be separated from
   :math:`\kappa_\mathrm{m}` along the same spatial dimension.
   See :ref:`background_parameters` for the identifiability analysis.

Prepare your data
-----------------

You need:

- A SOFA file for the listener's HRTF.
- A :class:`pandas.DataFrame` of measured responses with columns
  ``azi_target``, ``ele_target``, ``azi_response``, ``ele_response``
  (all in degrees, spherical-elevation convention).
- A :class:`pyfar.Coordinates` object with the target directions.

.. literalinclude:: ../../tests/test_guide_fit_model.py
   :language: python
   :start-after: # [prepare]
   :end-before: # [/prepare]

Run the two-stage fit
---------------------

:func:`~bayesian_listener.fitting.fit_listener` runs both stages and returns
a results dictionary.

.. literalinclude:: ../../tests/test_guide_fit_model.py
   :language: python
   :start-after: # [fit]
   :end-before: # [/fit]

.. note::

   Fitting a single listener takes roughly 5–15 minutes depending on the
   number of trials and Monte Carlo repetitions.  Set ``num_repetitions=50``
   for a quick exploratory fit; use the default ``num_repetitions=200``
   for publishable results.

Inspect the result
------------------

The returned dictionary contains all fitted and fixed parameter values, timing
information, and the final negative log-likelihood:

.. literalinclude:: ../../tests/test_guide_fit_model.py
   :language: python
   :start-after: # [inspect]
   :end-before: # [/inspect]

What to do next
---------------

Use the fitted parameters to simulate responses with
:doc:`simulate_responses`, or compare fits across interpolation methods with
:doc:`compare_interpolation`.  Full parameter documentation is in
:func:`~bayesian_listener.fitting.fit_listener`.
