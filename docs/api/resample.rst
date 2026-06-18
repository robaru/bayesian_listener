.. _api_resample:

Resample
========

Functions for HRTF template interpolation and spherical grid management.
The recommended entry point is :func:`~bayesian_listener.resample.resample`.

Interpolation
-------------

.. autofunction:: bayesian_listener.resample.resample

.. autofunction:: bayesian_listener.resample.resample_two_step

.. autofunction:: bayesian_listener.resample.resample_barumerli2023

.. autofunction:: bayesian_listener.resample.complement_sampling

.. autofunction:: bayesian_listener.resample.interpolate_HRTF

Spherical harmonics
-------------------

.. autofunction:: bayesian_listener.resample.find_max_order

.. autofunction:: bayesian_listener.resample.solve_sh

.. autofunction:: bayesian_listener.resample.build_Y

.. autofunction:: bayesian_listener.resample.build_bau_damping

Visualisation
-------------

.. autofunction:: bayesian_listener.resample.plot_resampling_grid

.. footbibliography::
