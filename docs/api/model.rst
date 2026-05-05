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

Barumerli2023
-------------

The concrete representation used by the default workflow: ITD + ILD +
monaural spectral amplitude envelopes ([barumerli2026]_).

.. autoclass:: bayesian_listener.Barumerli2023
   :members:
