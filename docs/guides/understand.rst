.. _guide_understand:

.. meta::
   :keywords: HRTF, sound localization, Bayesian, auditory model,
              individual, non-individual, dynamic, adaptation,
              workflow, use case, compute_target, compute_template,
              spatial hearing, head-related transfer function,
              psychoacoustics, ITD, ILD

How to use the model
=======================

:class:`~bayesian_listener.BayesianListener` simulates a listener
behavior in a sound localisation task. This class is a stateful object
that holds three things: a **template**, a **target**, and **noise parameters**.

- The **template** is the listener's internal model of the acoustic world represented as
  auditory features extracted from their own HRTF and interpolated onto a
  quasi-uniform spherical grid. It is expensive to compute and rarely changes.
- The **target** contains the auditory features of a single or multiple binaural stimuli to be localised.
  It can be swapped cheaply to change the sound source or test a non-individual HRTF.
- The **noise parameters** (e.g. :attr:`~bayesian_listener.BayesianListener.sigma_spectral`,
  :attr:`~bayesian_listener.BayesianListener.sigma_prior`) control perceputal and behavioral
  uncertainties and can be updated between calls.

:meth:`~bayesian_listener.BayesianListener.localise` is the one-call shortcut.
This design lets you swap just the target without recomputing everything.

If you want more control on the model internal workings,
then :meth:`~bayesian_listener.BayesianListener.infer` always operates on whatever
``template`` and ``target`` are currently stored.

To give an idea of what this model can do, this guide shows four typical usage patterns.
Each builds on the same :class:`~bayesian_listener.BayesianListener` API; the differences are in
which features are computed and how ``target`` and ``template`` are combined
before calling :meth:`~bayesian_listener.BayesianListener.infer`.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Workflow
     - When to use
   * - :ref:`workflow_individual`
     - Simulate responses for a listener's localising with their own HRTF.
   * - :ref:`workflow_nonindividual`
     - Simulate localisation with a non-individual HRTF.
   * - :ref:`workflow_dynamic`
     - Track moving sound source over time.
   * - :ref:`workflow_parameters`
     - Update noise parameters.

.. _workflow_individual:

Individual
----------

The simplest case: one listener, one HRTF.
:meth:`~bayesian_listener.BayesianListener.localise` handles feature
extraction, template interpolation, Bayesian inference, and motor noise in
a single call.

.. literalinclude:: ../../tests/test_guide_understand.py
   :language: python
   :start-after: # [individual]
   :end-before: # [/individual]

Results are cached on disk after the first call, so subsequent calls to
:meth:`~bayesian_listener.BayesianListener.compute_template` return
immediately.

.. _workflow_nonindividual:

Non-individual HRTF
-----------------------------------

Simulate how a listener performs when fitted with a non-individual HRTF.
The individual listener's **template** is retained; only the **target**
(the stimulus representation) is replaced with the foreign HRTF.

.. literalinclude:: ../../tests/test_guide_understand.py
   :language: python
   :start-after: # [nonindividual]
   :end-before: # [/nonindividual]

:meth:`~bayesian_listener.BayesianListener.localise` passes ``target`` to
:meth:`~bayesian_listener.BayesianListener.infer`, which compares the foreign
HRTF features against the individual template grid, modelling the
template-mismatch scenario.

.. _workflow_dynamic:

Dynamic (loop over targets)
----------------------------

When comparing multiple stimulus conditions against the same template —
e.g. different HRTFs or source sets — compute the template once and swap
only the target inside the loop.  Each ``target`` already contains
features for *all* source directions in its SOFA file, so a single swap
covers the full set of stimuli for that condition.  This pattern is the
basis of the FrAMBI framework [barumerli2025]_, which uses repeated
inference over a sequence of conditions to model dynamic auditory tasks.
[llado2024]_ applied this approach to predict how headphone
HRTFs affect the time to localise a target in an auditory-guided visual
search task.

A moving source can be modelled by feeding the posterior of each step as
the prior for the next.  The template stays fixed; only the single-direction
target changes at each step.  With one repetition and ``store_posterior=True``,
:meth:`~bayesian_listener.BayesianListener.infer` returns an array of shape
``(1, 1, n_templates)`` — squeezed to ``(n_templates,)`` it is a valid prior
for the following call.

.. literalinclude:: ../../tests/test_guide_understand.py
   :language: python
   :start-after: # [dynamic]
   :end-before: # [/dynamic]

.. _workflow_parameters:

Setting uncertainty parameters
-------------------------------

The default parameter values are group averages from Barumerli et al. (2023)
and are a reasonable starting point, but individual listeners differ
substantially in their perceptual uncertainties.  For accurate per-listener
simulation, set parameters explicitly before calling
:meth:`~bayesian_listener.BayesianListener.localise` or
:meth:`~bayesian_listener.BayesianListener.infer`.

Each parameter has a specific physical role in the static localisation task:

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Default
     - What it controls
   * - :attr:`~bayesian_listener.BayesianListener.sigma_spectral`
     - 10.4 dB
     - Reliability of monaural spectral cues (elevation, front/back).
       Higher values flatten the elevation response and increase
       front-back reversals.
   * - :attr:`~bayesian_listener.BayesianListener.sigma_prior`
     - 69.0 deg
     - Width of the elevation prior. Lower values pull responses towards
       the horizontal plane regardless of the stimulus.
   * - :attr:`~bayesian_listener.BayesianListener.kappa_motor`
     - 23.31
     - Concentration of the motor-noise distribution.  Lower values
       produce more scattered pointing responses independent of sensory
       processing.
   * - :attr:`~bayesian_listener.BayesianListener.sigma_itd`
     - 0.569
     - ITD perceptual noise (lateral localisation). Fixed at the
       literature value in most analyses — poorly identifiable from
       spatial data alone.
   * - :attr:`~bayesian_listener.BayesianListener.sigma_ild`
     - 1.0 dB
     - ILD perceptual noise (lateral localisation). Fixed at the
       literature value in most analyses.

.. note::

   :attr:`~bayesian_listener.BayesianListener.sigma_itd` and
   :attr:`~bayesian_listener.BayesianListener.sigma_ild` cannot be
   separated from :attr:`~bayesian_listener.BayesianListener.kappa_motor`
   along the lateral dimension and should be left at their defaults unless
   you have strong prior justification.  See :ref:`background_parameters`
   for the identifiability analysis.

All parameters can be set at construction time or updated in place:

.. literalinclude:: ../../tests/test_guide_understand.py
   :language: python
   :start-after: # [parameters]
   :end-before: # [/parameters]

If you have measured behavioural responses from a listener, individual
parameters can be estimated objectively via maximum likelihood optimisation
rather than set by hand.  See :doc:`fit_model` for the two-stage fitting
procedure.

See also
--------

- :class:`~bayesian_listener.BayesianListener` — full API reference.
- :doc:`simulate_responses` — metrics and response analysis.
- :doc:`fit_model` — estimating noise parameters from behavioural data.

References
----------

.. [barumerli2025] R. Barumerli and P. Majdak, "FrAMBI: A Software Framework
   for Auditory Modeling Based on Bayesian Inference," *Neuroinformatics*,
   vol. 23, no. 2, 2025. https://doi.org/10.1007/s12021-024-09702-5

.. [llado2024] P. Lladó, R. Barumerli, R. Baumgartner, and P. Majdak,
   "Predicting the effect of headphones on the time to localize a target
   in an auditory-guided visual search task," *Frontiers in Virtual Reality*,
   vol. 5, 2024. https://doi.org/10.3389/frvir.2024.1359987
