.. _guide_interpolation:

.. meta::
   :keywords: HRTF, interpolation, spherical harmonics, SHMAX, barycentric,
              BIC, model selection, evaluating HRTF, individual acoustics,
              auditory model, model-based analysis, computational modelling,
              Bayes inference, spatial hearing,
              head-related transfer function, sound localization, MLE

Compare Interpolation Methods
==============================

This guide shows how to evaluate HRTF template interpolation and the
contribution of the elevation prior using likelihood-based model comparison
(BIC).  Two models are compared: a full model (``sigma_spectral`` +
``sigma_prior`` free) and a reduced model (``sigma_spectral`` only,
``sigma_prior`` fixed to a uniform distribution).  A positive
:math:`\Delta\mathrm{BIC}` indicates that the elevation prior improves the
fit beyond what is expected from the added parameter complexity.

Fit the full model and the no-prior model
-----------------------------------------

Run :func:`~bayesian_listener.fitting.fit_listener` for the full model, then
:func:`~bayesian_listener.fitting.fit_listener_partial` with ``sigma_prior``
fixed to a very large value (effectively uniform) for the no-prior model.

.. literalinclude:: ../../tests/test_guide_compare_interpolation.py
   :language: python
   :start-after: # [fit_methods]
   :end-before: # [/fit_methods]

Compute BIC and delta-BIC
--------------------------

BIC penalises model complexity; lower is better.
:math:`\Delta\mathrm{BIC} = \mathrm{BIC}_{\mathrm{no\,prior}} - \mathrm{BIC}_{\mathrm{full}}`
is positive when the elevation prior is justified.

.. literalinclude:: ../../tests/test_guide_compare_interpolation.py
   :language: python
   :start-after: # [bic]
   :end-before: # [/bic]

.. note::

   :math:`|\Delta\mathrm{BIC}| > 10` is conventionally regarded as strong
   evidence in favour of the lower-BIC model (see [barumerli2026]_ in the
   :ref:`Background <background>` page).

References
----------

.. [barumerli2026] R. Barumerli, F. Brinkmann, E. Zanoni, A. Hoyer,
   L. Picinali, and M. Geronazzo, "Statistical validation and full-sphere
   extension of a Bayesian model for human static sound localisation,"
   *Submitted to Acta Acustica*, 2026.
