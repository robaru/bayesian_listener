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

Likelihood
----------

.. autofunction:: bayesian_listener.fitting.negloglik

.. autofunction:: bayesian_listener.fitting.von_mises_loglik_mc

.. autofunction:: bayesian_listener.fitting.fit_kappa_ml

Parameter conversion
--------------------

.. autofunction:: bayesian_listener.fitting.sigma_to_kappa

.. autofunction:: bayesian_listener.fitting.kappa_to_sigma

Default parameters and bounds
-----------------------------

.. autodata:: bayesian_listener.fitting.DEFAULT_PARAMS

.. autodata:: bayesian_listener.fitting.DEFAULT_PARAM_BOUNDS

.. footbibliography::
