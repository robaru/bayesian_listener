.. _guide_simulate:

.. meta::
   :keywords: HRTF, sound localization, Bayesian, simulate, infer, estimate,
              metrics, von Mises, MAP, motor noise, auditory model,
              model-based analysis, computational modelling, individual
              acoustics, Bayes inference, spatial hearing,
              head-related transfer function, psychoacoustics, ITD, ILD

Simulate Localization Responses
================================

This guide shows how to generate predicted sound-localisation responses
for a given HRTF and set of model parameters.  The output is a
:class:`pyfar.Coordinates` object of simulated pointing directions, from
which standard localisation metrics (lateral RMS error, polar RMS error,
quadrant error rate) can be computed.

The workflow has three steps: feature preparation, Bayesian inference, and
motor noise sampling.

Prepare features
----------------

:meth:`~bayesian_listener.BayesianListener.compute_template` computes ITD,
ILD, and spectral cues from the HRTF and interpolates them onto a uniform
spherical template grid.

.. literalinclude:: ../../tests/test_guide_simulate.py
   :language: python
   :start-after: # [prepare]
   :end-before: # [/prepare]

Run inference
-------------

:meth:`~bayesian_listener.BayesianListener.infer` computes the MAP direction
for each source position, repeated ``repetitions`` times to account for
sensory noise.

.. literalinclude:: ../../tests/test_guide_simulate.py
   :language: python
   :start-after: # [infer]
   :end-before: # [/infer]

Sample motor responses
----------------------

:meth:`~bayesian_listener.BayesianListener.estimate` adds von Mises–Fisher
motor noise to each MAP estimate and returns pointing directions.

.. literalinclude:: ../../tests/test_guide_simulate.py
   :language: python
   :start-after: # [estimate]
   :end-before: # [/estimate]

.. note::

   Pass ``kappa_motor=False`` to disable motor noise and recover the pure
   MAP estimate — useful for debugging or comparing against noiseless
   model predictions.

Compute localisation metrics
----------------------------

Use :func:`~bayesian_listener.metrics.localization_error` to compute standard
metrics one at a time.  The function expects flat :class:`pyfar.Coordinates`
of the same length, so repeat the target directions to match the response array.

.. literalinclude:: ../../tests/test_guide_simulate.py
   :language: python
   :start-after: # [metrics]
   :end-before: # [/metrics]

What to do next
---------------

Use fitted parameters from :doc:`fit_model` to make the simulation
listener-specific.  Full API documentation is in
:class:`~bayesian_listener.BayesianListener` and
:func:`~bayesian_listener.metrics.localization_error`.
