.. _api_evaluation:

Evaluation
==========

Functions for computing localisation metrics and evaluating likelihood.

Localisation metrics
--------------------

.. autofunction:: bayesian_listener.metrics.localization_error

.. autofunction:: bayesian_listener.metrics.wrap_polar_angle

Metric registry
---------------

Custom metrics can be registered and queried via the decorator API.

.. autofunction:: bayesian_listener.metrics.register_metric

.. autofunction:: bayesian_listener.metrics.get_metric_metadata

.. autofunction:: bayesian_listener.metrics.describe_metrics

Likelihood evaluation
---------------------

.. autofunction:: bayesian_listener.fitting.von_mises_loglik_mc

.. autofunction:: bayesian_listener.fitting.fit_kappa_ml
