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

.. code-block:: python

   from bayesian_listener import BayesianListener

   listener = BayesianListener("P0001_FreeFieldCompMinPhase_48kHz.sofa")
   listener.compute_template(interpolation="SHMAX")

Run inference
-------------

:meth:`~bayesian_listener.BayesianListener.infer` computes the MAP direction
for each source position, repeated ``repetitions`` times to account for
sensory noise.

.. code-block:: python

   posterior = listener.infer(repetitions=200, prior="horizontal", seed=0)
   # posterior shape: (n_targets, repetitions) — argmax indices into template grid

Sample motor responses
----------------------

:meth:`~bayesian_listener.BayesianListener.estimate` adds von Mises–Fisher
motor noise to each MAP estimate and returns pointing directions.

.. code-block:: python

   responses = listener.estimate(posterior, seed=0)
   # responses: pyfar.Coordinates, shape (n_targets, repetitions)

.. note::

   Pass ``kappa_motor=False`` to disable motor noise and recover the pure
   MAP estimate — useful for debugging or comparing against noiseless
   model predictions.

Compute localisation metrics
----------------------------

Use :func:`~bayesian_listener.metrics.localization_error` to compute lateral
RMS error (LE), polar RMS error (PE), and quadrant error rate (QE) from the
simulated responses.

.. code-block:: python

   from bayesian_listener.metrics import localization_error
   import numpy as np

   # Ground-truth target directions (same order as listener.coords)
   targets = listener.coords   # pyfar.Coordinates

   # Flatten repetitions into a single response list
   n_targets, n_reps = posterior.shape
   targets_repeated = targets[np.repeat(np.arange(n_targets), n_reps)]
   responses_flat   = responses.reshape(n_targets * n_reps)

   errors = localization_error(
       targets_repeated,
       responses_flat,
       metrics=["rmsL", "rmsPmedianlocal", "querrMiddlebrooks"],
   )
   print(errors)

What to do next
---------------

Use fitted parameters from :doc:`fit_model` to make the simulation
listener-specific.  Full API documentation is in
:class:`~bayesian_listener.BayesianListener` and
:func:`~bayesian_listener.metrics.localization_error`.
