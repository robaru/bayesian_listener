=======
History
=======

0.2.0 (2026-09-18)
------------------

Added
^^^^^
* ``reference`` argument for the level normalisation of the spatial features
  (``'frontal'``, ``'global'`` or ``'none'``), exposed through
  ``BayesianListener.compute_target``. The default (``'frontal'``) is
  unchanged; ``'global'`` references the whole HRTF set rather than one
  direction, which matters when target and template come from different
  SOFA files.
* Release procedure in ``CONTRIBUTING.rst``.

Changed
^^^^^^^
* The feature cache is keyed on convention, spectral range, half-wave
  rectification and level reference, and the pickle format is stamped, so
  targets that differ only in those parameters no longer collide and caches
  written by earlier versions are treated as a miss.
* ``utils.randvmf`` is now documented as the scalar reference implementation
  of the von Mises-Fisher sampler.
* ``barumerli2026`` reference updated to the published Acta Acustica paper.

Fixed
^^^^^
* Motor noise collapsed to a single fixed rotation in seeded simulations:
  the random generator was re-seeded per direction and per repetition, so
  every response received the same perturbation. The scatter is now drawn
  once per call, restoring the expected trial-to-trial variability and
  matching the AMT MATLAB reference. Seeded simulation results change with
  this release.
* ``resample(method='barumerli2023')`` read the input directions as
  colatitude while the spherical-harmonics basis expects elevation, warping
  the interpolated cues.
* ``angular_error`` normalises the input vectors, so the metric no longer
  depends on the measurement radius and no longer clips large errors to
  zero.
* ``localization_error`` forwards ``degrees`` when a list of metrics is
  requested; it previously always returned radians in that case.
* Degenerate rotation axis at the south pole in the von Mises-Fisher sampler
  no longer produces NaN.

0.1.1 (2026-06-25)
------------------

Changed
^^^^^^^
* Removing PDF format when building documentation

0.1.0 (2026-06-25)
------------------
* First release on PyPI
