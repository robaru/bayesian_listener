.. _api_model:

Model
=====

The core class and auditory representation types.

BayesianListener
----------------

The single entry point for all inference and simulation workflows.

.. autoclass:: bayesian_listener.BayesianListener
   :members:
   :inherited-members:

AuditoryRepresentation
-----------------------

Abstract base class for all auditory representation types.  Subclass this
to implement a new set of spatial cues.

.. autoclass:: bayesian_listener.AuditoryRepresentation
   :members:

Barumerli2025
-------------

The concrete representation used by the default workflow: ITD + ILD +
monaural spectral amplitude envelopes ([barumerli2026]_).

.. autoclass:: bayesian_listener.Barumerli2025
   :members:
