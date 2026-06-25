.. _api_bayesian_listener:

Bayesian Listener
=================

The core class and auditory representation.

BayesianListener
----------------

The single entry point for all inference and simulation workflows.

.. autoclass:: bayesian_listener.BayesianListener
   :members:
   :inherited-members:

Auditory Representation
-----------------------

The concrete auditory representation used by the default workflow: ITD + ILD +
monaural spectral amplitude envelopes (:footcite:t:`barumerli2026`).

.. autoclass:: bayesian_listener.Barumerli2023
   :members:

.. footbibliography::
