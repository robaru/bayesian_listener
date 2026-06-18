.. _api_metrics:

Metrics
=======

Functions for computing localisation metrics and registering custom ones.

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

Built-in metrics
----------------

Lateral error
~~~~~~~~~~~~~

.. autofunction:: bayesian_listener.metrics.sdL

.. autofunction:: bayesian_listener.metrics.rmsL

.. autofunction:: bayesian_listener.metrics.accL_cutoff

Polar error
~~~~~~~~~~~

.. autofunction:: bayesian_listener.metrics.rmsPmedianlocal

.. autofunction:: bayesian_listener.metrics.accP_cutoff

Global error
~~~~~~~~~~~~

.. autofunction:: bayesian_listener.metrics.querrMiddlebrooks

.. autofunction:: bayesian_listener.metrics.angular_error

.. footbibliography::
