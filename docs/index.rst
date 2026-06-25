.. meta::
   :keywords: HRTF, sound localization, Bayesian, von Mises, AMT,
              auditory model, MLE, model-based analysis,
              computational modelling, evaluating HRTF, individual acoustics,
              Bayes inference, spatial hearing, head-related transfer function,
              sagittal-plane localisation, monaural cues, ITD, ILD,
              sound source localization, psychoacoustics

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   getting_started
   guides/index
   api/index
   background

An auditory model for simulating human sound localisation
=========================================================

.. image:: https://img.shields.io/pypi/v/bayesian_listener
   :target: https://pypi.org/project/bayesian_listener/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/bayesian_listener
   :target: https://pypi.org/project/bayesian_listener/
   :alt: Python versions

.. image:: https://img.shields.io/badge/license-EUPL%201.2-blue
   :target: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
   :alt: EUPL 1.2 license

.. image:: https://img.shields.io/badge/DOI-10.1051%2Faacus%2F2023006-blue
   :target: https://doi.org/10.1051/aacus/2023006
   :alt: DOI

.. image:: https://dl.circleci.com/status-badge/img/circleci/BeUU72xrVhZQnvbQs1NtoY/RS9s4ZVXdbRqp4wzT27zLN/tree/main.svg?style=shield&circle-token=CCIPRJ_KhBs7PQLxsRpyKQGpnW17H_8532931196534425ee26a2489db41df66a1ad231
   :target: https://dl.circleci.com/status-badge/redirect/circleci/BeUU72xrVhZQnvbQs1NtoY/RS9s4ZVXdbRqp4wzT27zLN/tree/main
   :alt: CircleCI

|

:mod:`bayesian_listener` is a Python package for simulating and fitting a
Bayesian model of human sound localisation.  Given an individual's
head-related transfer functions (HRTFs) and a binaural sound, it predicts full response
distributions over all source directions (accounting for spectral,
binaural, and motor-noise uncertainties).  Listener-specific noise parameters
can be estimated from measured pointing data via maximum-likelihood
optimisation.

Where to start
--------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Get Started
      :link: getting_started
      :link-type: doc

      Install the package and run your first localization simulation in under
      five minutes.  Start here if you are new to :mod:`bayesian_listener`.

   .. grid-item-card:: Guides
      :link: guides/index
      :link-type: doc

      Task-oriented walkthroughs: simulate localization responses,
      fit the model to your own HRTF, and compare interpolation methods.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete documentation of every public class, method, and function.
      Jump here if you know what you need and want parameter details.

   .. grid-item-card:: Background
      :link: background
      :link-type: doc

      The statistical framework, likelihood equations, noise-parameter table,
      and known limitations.  Start here if you are reading the paper.

Citing this work
----------------

If you use :mod:`bayesian_listener` in your research, please cite the
original model paper and its statistical validation:

.. code-block:: bibtex

   @article{barumerli2026,
      author  = {R. Barumerli and F. Brinkmann and E. Zanoni and A. Hoyer
                  and L. Picinali and M. Geronazzo},
      title   = {Statistical validation and full-sphere extension of a {Bayesian}
                  model for human static sound localisation},
      journal = {Submitted to Acta Acustica},
      year    = {2026},
      url = {https://arxiv.org/abs/2606.24367}
   }

   @article{barumerli2023,
     author  = {Barumerli, Roberto and Majdak, Piotr and Geronazzo, Michele
                and Meijer, Demi and Avanzini, Federico and Baumgartner, Robert},
     title   = {A {Bayesian} model for human directional localization of
                broadband static sound sources},
     journal = {Acta Acustica},
     volume  = {7},
     pages   = {12},
     year    = {2023},
     doi     = {10.1051/aacus/2023006},
   }


