# Bayesian Listener

Tired to run experiments with real listeners? This model simulates their behavior in a sound localisation task by offering a Bayesian model reproducing human directional sound localization performances.

Moreover, this package offers an explicit likelihood function for parameter estimation (how noisy is this participant in comparison to another one, or how do parameters change over listening conditions?) based on principled maximum likelihood estimation, and statistical model comparison (is this spatial feature better than this other one?).

The model combines an observer's prior expectations about source locations with spectral cues (via individual HRTF) and interaural timing/level differences, producing likelihood surfaces that characterize listener behavior in 3D space.

Key Features:

- **Fit localization behavior** from behavioral data
- **Recover and validate model parameters** with maximum likelihood estimation
- **Compare HRTF interpolation methods** quantitatively
- **Generate synthetic responses** for validation and analysis
- **Accelerated computation** via just-in-time compilation with `numba`

## Installation

```bash
pip install bayesian_listener
```

Requires Python 3.10 or higher.

## Quick Start

```python
from bayesian_listener import BayesianListener

hrtf_data = 'hrtf.sofa'

listener = BayesianListener(hrtf=hrtf_data)
listener.compute_template()
estimations = listener.localise()

print(estimations.spherical_elevation[..., 0:2])
```

See [Getting Started](https://bayesian_listener.readthedocs.io/en/latest/getting_started.html) for more detailed examples.

## Documentation

- **[Getting Started](https://bayesian_listener.readthedocs.io/)** — Installation and first example
- **[Guides](https://bayesian_listener.readthedocs.io/en/latest/guides/)** — Task-oriented walkthroughs (fitting, simulation, HRTF comparison)
- **[API Reference](https://bayesian_listener.readthedocs.io/en/latest/api/)** — Complete class and function documentation
- **[Background](https://bayesian_listener.readthedocs.io/en/latest/background.html)** — Statistical framework and equations

## References

Barumerli, R., Majdak, P., Geronazzo, M., Meijer, D., Avanzini, F., & Baumgartner, R. (2023).
*A Bayesian model for human directional localization of broadband static sound sources.*
Acta Acustica, 7, 12. [Paper](https://doi.org/10.1051/aacus/2023006)

Barumerli, R., Brinkmann, F., Zanoni, E., Hoyer, A., Picinali, L., & Geronazzo, M. (2026).
*Statistical validation and full-sphere extension of a Bayesian model for human static sound localisation.*
Acta Acustica (under review). [Pre-print](https://arxiv.org/abs/2606.24367)

## License

EUPL 1.2 (European Union Public Licence)

## Authors

- **Roberto Barumerli** — Main model implementation and validation
- **Fabian Brinkmann** — Spherical harmonics interpolation and interface design
- **Anton Hoyer** — Continuous integration, PyFAR and SOFAr integration
- **Emanuele Zanoni** — Implementation validation

## Acknowledgements

The original model was implemented in MATLAB within the **Auditory Modeling Toolbox (AMT)**. This Python version incorporates code components from AMT (e.g., gammatone filtering). We are grateful to the AMT authors for sharing their work.

The spherical t-design grids bundled with this package were originally published by Manuel Gräf and redistributed by [spaudiopy](https://github.com/chris-hold/spaudiopy).

The VBAP interpolation algorithm is adapted from [spaudiopy](https://github.com/chris-hold/spaudiopy), based on Pulkki, V. (1997).
