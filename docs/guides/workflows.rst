.. _guide_workflows:

.. meta::
   :keywords: HRTF, sound localization, Bayesian, auditory model,
              individual, non-individual, dynamic, adaptation,
              workflow, use case, compute_target, compute_template,
              spatial hearing, head-related transfer function,
              psychoacoustics, ITD, ILD

Common Workflows
================

This guide shows four typical usage patterns.  Each builds on the same
:class:`~bayesian_listener.BayesianListener` API; the differences are in
which features are computed and how ``target`` and ``template`` are combined
before calling :meth:`~bayesian_listener.BayesianListener.infer`.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Workflow
     - When to use
   * - :ref:`workflow_individual`
     - Default: simulate responses for a listener's own HRTF.
   * - :ref:`workflow_nonindividual`
     - Evaluate a non-individual HRTF against an individual template.
   * - :ref:`workflow_dynamic`
     - Loop over many targets without re-computing the template.
   * - :ref:`workflow_adaptation`
     - Update noise parameters trial-by-trial to model adaptation.

.. _workflow_individual:

Individual
----------

The simplest case: one listener, one HRTF.
:meth:`~bayesian_listener.BayesianListener.compute_template` extracts
features from ``sofa`` and interpolates them onto a uniform grid.
It also calls :meth:`~bayesian_listener.BayesianListener.compute_target`
automatically, so a single line is sufficient.

.. code-block:: python

   from bayesian_listener import BayesianListener

   bl = BayesianListener("P0001_FreeFieldCompMinPhase_48kHz.sofa",
                         sigma_spectral=10.4)
   bl.compute_template()               # computes target + template

   posterior = bl.infer(repetitions=50, seed=0)
   estimates = bl.estimate(posterior)  # pyfar.Coordinates of predicted directions

Results are cached on disk after the first call, so subsequent calls to
:meth:`~bayesian_listener.BayesianListener.compute_template` return
immediately.

.. _workflow_nonindividual:

Non-individual (template mismatch)
-----------------------------------

Simulate how a listener performs when fitted with a non-individual HRTF.
The individual listener's **template** is retained; only the **target**
(the stimulus representation) is replaced with the foreign HRTF.

.. code-block:: python

   from bayesian_listener import BayesianListener

   # Build template from the individual's own HRTF
   individual = BayesianListener("individual.sofa")
   individual.compute_template()                # computes target + template

   # Extract target features from the foreign HRTF (no interpolation needed)
   foreign = BayesianListener("foreign.sofa")
   foreign.compute_target()                     # computes target only

   # Swap the individual's target for the foreign one
   individual.target = foreign.target
   estimates = individual.estimate(individual.infer())

:meth:`~bayesian_listener.BayesianListener.infer` compares
``individual.target`` (foreign HRTF features) against
``individual.template`` (individual template grid), modelling the
template-mismatch scenario.

.. _workflow_dynamic:

Dynamic (loop over targets)
----------------------------

When evaluating many stimuli against the same template — e.g. sweeping
over source positions or comparing HRTF databases — compute the template
once and swap only the target inside the loop.

.. code-block:: python

   from bayesian_listener import BayesianListener

   # Build the template once
   bl = BayesianListener("subject.sofa")
   bl.compute_template()

   # Pre-compute targets for each condition
   targets = []
   for sofa_path in sofa_paths:
       listener = BayesianListener(sofa_path)
       listener.compute_target()
       targets.append(listener.target)

   # Loop: only infer() is repeated; the expensive template stays fixed
   estimates_list = []
   for target in targets:
       bl.target = target
       estimates_list.append(bl.estimate(bl.infer(seed=0)))

.. _workflow_adaptation:

Adaptation (trial-by-trial parameter update)
---------------------------------------------

Model perceptual adaptation by updating noise parameters after each trial.
All parameter properties have setters so they can be changed in place.

.. code-block:: python

   from bayesian_listener import BayesianListener

   bl = BayesianListener("subject.sofa")
   bl.compute_template()

   estimates_list = []
   for trial in range(n_trials):
       posterior = bl.infer(repetitions=50, seed=trial)
       estimates = bl.estimate(posterior)
       estimates_list.append(estimates)

       # Update parameters based on the trial outcome
       bl.sigma_spectral = my_update_spectral(estimates)
       bl.sigma_prior    = my_update_prior(estimates)

The properties :attr:`~bayesian_listener.BayesianListener.sigma_itd`,
:attr:`~bayesian_listener.BayesianListener.sigma_ild`,
:attr:`~bayesian_listener.BayesianListener.sigma_spectral`,
:attr:`~bayesian_listener.BayesianListener.sigma_prior`, and
:attr:`~bayesian_listener.BayesianListener.sigma_motor` (which writes
:attr:`~bayesian_listener.BayesianListener.kappa_motor`) can all be
updated between calls to :meth:`~bayesian_listener.BayesianListener.infer`.

See also
--------

- :class:`~bayesian_listener.BayesianListener` — full API reference.
- :doc:`simulate_responses` — metrics and response analysis.
- :doc:`fit_model` — estimating noise parameters from behavioural data.
