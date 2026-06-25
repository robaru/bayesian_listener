.. _api_utils:

Utils
=====

Feature extraction, spherical grid utilities, caching helpers, and sampling.
The recommended entry point for feature extraction is
:func:`~bayesian_listener.utils.compute_features`.

Feature extraction
------------------

.. autofunction:: bayesian_listener.utils.compute_features

.. autofunction:: bayesian_listener.utils.gammatone

.. autofunction:: bayesian_listener.utils.minimum_ir_length

.. autofunction:: bayesian_listener.utils.itdestimator

Spherical grid
--------------

.. autofunction:: bayesian_listener.utils.load_n_design

.. autofunction:: bayesian_listener.utils.vbap_interpolate

Sampling
--------

.. autofunction:: bayesian_listener.utils.scatter_von_mises

Caching
-------

.. autofunction:: bayesian_listener.utils.cache_load_target

.. autofunction:: bayesian_listener.utils.cache_load_template

.. autofunction:: bayesian_listener.utils.cache_save_target

.. autofunction:: bayesian_listener.utils.cache_save_template

.. autofunction:: bayesian_listener.utils.clear_cache
