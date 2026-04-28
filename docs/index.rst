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

bayesian_listener
=================

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

:mod:`bayesian_listener` simulates and fits a Bayesian model of human sound
localisation from individual head-related transfer functions (HRTFs).
It is the open-source Python implementation of the model introduced in
[barumerli2023]_ and statistically validated in [barumerli2026]_; see
:ref:`background` for the equations, parameter table, and known limitations.

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

      Task-oriented walkthroughs: fit the model to your own HRTF, simulate
      localization responses, and compare interpolation methods.

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
original model paper:

.. code-block:: bibtex

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
