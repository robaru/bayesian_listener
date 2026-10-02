# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`bayesian_listener` — a Python package implementing a Bayesian model of human directional sound localisation from HRTFs (Barumerli et al. 2023, extended in Barumerli et al. 2026). Used for simulating listener responses, maximum-likelihood parameter fitting, and quantitative comparison of HRTF interpolation methods.

## Commands

```bash
pip install -e ".[dev]"        # dev install (deploy + tests + docs extras)

pytest                          # run tests (guide tests excluded via addopts "-m 'not guide'")
pytest -m guide                 # run documentation guide tests only
pytest tests/test_metrics.py::test_name   # single test
pytest tests -W error::DeprecationWarning # CI also runs this

ruff check                      # lint — must pass with zero warnings (CI gate)

# docs build, mirrors CI (-W promotes warnings to errors)
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

CI is CircleCI (`.circleci/config.yml`): tests, ruff, strict docs build, deprecation-warning run. Pushing a `vX.Y.Z` tag triggers PyPI publish. Releases use `bump-my-version` (config in `pyproject.toml`; updates `pyproject.toml` and `bayesian_listener/__init__.py`, commits and tags).

Tests download SONICOM SOFA files (P0001/P0002) into `data/` on first run via `conftest.py` fixtures (`sofa_path`, `sofa_path_non_individual`); they skip if the download fails.

## Architecture

The model pipeline lives in `BayesianListener` (`bayesian_listener/bayesian_listener.py`):

1. `compute_target()` — extract spatial features (ITD, ILD, monaural spectra) from the SOFA file at measured directions, wrapped in an auditory representation.
2. `compute_template()` — resample those cues onto a quasi-uniform grid (spherical t-design) via `resample`; this is the listener's internal directional model.
3. `infer()` — Monte Carlo Bayesian inference: noisy target features compared against template features under a Gaussian sensory likelihood plus an elevation prior; returns posterior MAP indices.
4. `estimate()` — perturb MAP directions with von Mises motor noise → pointing responses (`pyfar.Coordinates`).
5. `localise()` — convenience wrapper running the whole chain.

Supporting modules:

- `auditory_representation.py` — `_AuditoryRepresentation` ABC + concrete `Barumerli2023` (ITD + ILD + spectral envelope). Each subclass defines the feature concatenation and its diagonal covariance (`sigma_matrix`). The `CONVENTIONS` dict at the bottom registers subclasses by name; `compute_target(convention=...)` looks them up there.
- `resample.py` — four interpolation methods dispatched by `resample(method=...)`: `'SH'`, `'SHMAX'` (order-44 SH with Bau/Tikhonov damping, the default), `'barycentric'` (VBAP), `'barumerli2023'` (legacy order-15). Direction arrays here are `(azimuth, elevation)` in radians; coordinate-convention mismatches between pyfar conventions have caused real bugs, so be careful with `spherical_elevation` vs. `spherical_colatitude` when touching this code.
- `fitting.py` — two-stage ML fit (`fit_listener`): stage 1 fits motor noise κ from a lateral-only von Mises likelihood (ITD+ILD cues only); stage 2 holds κ fixed and fits spectral/prior sigmas with BADS. `negloglik` is the full-sphere likelihood.
- `metrics.py` — `localization_error(targets, estimations, metric)` is the single entry point; metrics live in a registry populated by the `@register_metric` decorator (Middlebrooks 1999 interaural-polar metrics). Add new metrics via the decorator, not new public functions.
- `utils.py` — gammatone/ERB feature extraction (ported from AMT MATLAB), numba-jitted vectorised Gaussian log-pdfs used by `infer`, and the pickle cache.

**Caching:** `compute_target`/`compute_template` cache results as pickles in `data/preprocessed/` with a `cache_index.csv`, keyed by the SOFA file path/hash and computation parameters. Caching is automatically disabled when the listener is constructed from an in-memory `sofar.Sofa` object instead of a file path. `utils.clear_cache()` resets it.

## Docs ↔ guide tests coupling

Every code example in `docs/guides/*.rst` is backed by a test in `tests/test_guide_<name>.py`, marked `pytestmark = pytest.mark.guide`. Sphinx pulls the snippets directly from the test files via `literalinclude` with `# [section]` / `# [/section]` marker comments. When changing a guide's code, edit the test, keep the markers intact, and verify with `pytest -m guide` plus the strict docs build. The `ERA001` ruff ignore for `tests/*` exists precisely for these marker comments.

## Conventions

- Ruff config in `pyproject.toml`: line length 100, numpy docstring convention, strict rule selection (docstrings required in package code, absolute imports only, no commented-out code, no TODO/FIXME).
- Docstrings use Sphinx roles heavily (`:class:`, `:meth:`, `:footcite:t:`) with references resolved from `docs/refs.bib` — new citations go there.
- PRs target the `develop` branch; keep branches current via rebase (not merge) onto `origin/develop`.
