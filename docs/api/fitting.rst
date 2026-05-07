.. _api_fitting:

Fitting
=======

Functions for maximum-likelihood parameter estimation.  The recommended
entry point is :func:`~bayesian_listener.fitting.fit_listener`, which
implements the full two-stage profile-likelihood procedure described in
:footcite:t:`barumerli2026`.

Two-stage fitting
-----------------

.. autofunction:: bayesian_listener.fitting.fit_listener

.. autofunction:: bayesian_listener.fitting.fit_listener_partial

.. autofunction:: bayesian_listener.fitting.estimate_motor_noise

Likelihood objective
--------------------

.. autofunction:: bayesian_listener.fitting.negloglik

Parameter bounds
----------------

Default parameter values and search bounds are defined in the module-level
constants ``bayesian_listener.fitting.DEFAULT_PARAMS`` and
``bayesian_listener.fitting.PARAM_BOUNDS``.  Consult the source via the
:guilabel:`[source]` link on :func:`~bayesian_listener.fitting.fit_listener`
for the exact values.

.. footbibliography::
