.. _api_statistics:

Statistics
==========

Distribution helpers for the von Mises and von Mises–Fisher distributions
used throughout the model and fitting pipeline.

Parameter conversion
--------------------

.. autofunction:: bayesian_listener.fitting.sigma_to_kappa

.. autofunction:: bayesian_listener.fitting.kappa_to_sigma

Sampling
--------

.. autofunction:: bayesian_listener.utils.scatter_von_mises

Log-probability
---------------

The Monte Carlo von Mises log-likelihood used internally by the fitter is
documented under :ref:`api_evaluation`
(:func:`~bayesian_listener.fitting.von_mises_loglik_mc`).
