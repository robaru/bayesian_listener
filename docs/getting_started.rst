.. _getting_started:

.. meta::
   :keywords: HRTF, sound localization, Bayesian, auditory model, install,
              AMT, MLE, model-based analysis, computational modelling,
              evaluating HRTF, individual acoustics, Bayes inference,
              spatial hearing, head-related transfer function,
              sagittal-plane localisation, monaural cues, ITD, ILD,
              sound source localization, psychoacoustics

Getting Started
===============

:mod:`bayesian_listener` simulates and fits a Bayesian model of human sound
localisation from individual head-related transfer functions (HRTFs).

With this package you can:

- **Simulate** predicted sound-localisation responses for any HRTF, returning a
  full response distribution over all source directions.
- **Fit** the model to measured behavioural data to estimate listener-specific
  noise parameters via maximum likelihood.

Installation
------------

.. code-block:: bash

   pip install bayesian_listener

.. note::

   Requires Python 3.10 or higher.
   For the statistical framework see :doc:`background`.

Minimal working example
-----------------------

The example below downloads a single listener's HRTF from the
`SONICOM dataset <https://doi.org/10.17605/OSF.IO/M36C2>`__, computes the
auditory features, runs Bayesian inference across all measured source
directions, and adds motor noise to produce simulated pointing responses.

.. code-block:: python

   import urllib.request
   from bayesian_listener import BayesianListener

   # Download one SONICOM HRTF (≈ 10 MB; runs once).
   sofa_path = "P0001_FreeFieldCompMinPhase_48kHz.sofa"
   urllib.request.urlretrieve(
       "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/"
       "P0001/HRTF/HRTF/48kHz/" + sofa_path,
       sofa_path,
   )

   listener = BayesianListener(sofa_path)
   listener.compute_template()               # extract features + build template

   posterior  = listener.infer(repetitions=10)   # Bayesian inference
   responses  = listener.estimate(posterior)     # add motor noise

   # responses is a pyfar.Coordinates object (azimuth, elevation, radius).
   print(responses.spherical_elevation[:5])

.. note::

   :meth:`~bayesian_listener.BayesianListener.compute_template` can take a few
   seconds the first time; results are cached automatically so subsequent
   calls return immediately.

What to do next
---------------

- :doc:`guides/fit_model` — estimate noise parameters from measured pointing
  responses.
- :doc:`guides/simulate_responses` — generate full response distributions and
  compute standard localisation metrics.
- :doc:`guides/compare_interpolation` — compare HRTF interpolation methods
  using likelihood-based model selection.
- :class:`~bayesian_listener.BayesianListener` — full API reference for the
  core class.
- :doc:`background` — statistical framework and model equations.
