.. _api_hrtf:

HRTF Utilities
==============

Functions for HRTF interpolation, feature extraction, and spherical grid
management.  The recommended entry points are
:func:`~bayesian_listener.utils.compute_features` (feature extraction) and
:func:`~bayesian_listener.resample.resample` (template interpolation).

Feature extraction
------------------

.. autofunction:: bayesian_listener.utils.compute_features

.. autofunction:: bayesian_listener.utils.gammatone

.. autofunction:: bayesian_listener.utils.itdestimator

Interpolation
-------------

.. autofunction:: bayesian_listener.resample.resample

.. autofunction:: bayesian_listener.resample.resample_two_step

.. autofunction:: bayesian_listener.resample.resample_barumerli2023

.. autofunction:: bayesian_listener.resample.complement_sampling

.. autofunction:: bayesian_listener.resample.interpolate_HRTF

Spherical harmonics helpers
----------------------------

.. autofunction:: bayesian_listener.resample.find_max_order

.. autofunction:: bayesian_listener.resample.solve_sh

.. autofunction:: bayesian_listener.resample.build_Y

.. autofunction:: bayesian_listener.resample.build_bau_damping

Spherical grid
--------------

.. autofunction:: bayesian_listener.utils.load_n_design

.. autofunction:: bayesian_listener.utils.vbap_interpolate

Visualisation
-------------

.. autofunction:: bayesian_listener.resample.plot_resampling_grid

Caching
-------

.. autofunction:: bayesian_listener.utils.cache_load_target

.. autofunction:: bayesian_listener.utils.cache_load_template

.. autofunction:: bayesian_listener.utils.cache_save_target

.. autofunction:: bayesian_listener.utils.cache_save_template

.. autofunction:: bayesian_listener.utils.clear_cache
