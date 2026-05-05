# -*- coding: utf-8 -*-

"""Top-level package for Bayesian Listener."""

__author__ = """Roberto Barumerli, Fabian Brinkmann, Emanuele Zanoni, Anton Hoyer"""
__email__ = 'r.barumerli@imperial.ac.uk'
__version__ = '0.1'
__all__ = [
    'BayesianListener',
    'Barumerli2023',
    'metrics',
    'resample',
    'utils',
    'fitting',
]

from .bayesian_listener import BayesianListener
from .auditory_representation import Barumerli2023

from . import metrics
from . import resample
from . import utils
from . import fitting


