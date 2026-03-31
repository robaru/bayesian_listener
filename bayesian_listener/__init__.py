# -*- coding: utf-8 -*-

"""Top-level package for Bayesian Listener."""

__author__ = """Roberto Barumerli, Fabian Brinkmann, Emanuele Zanoni, Anton Hoyer"""
__email__ = 'r.barumerli@imperial.ac.uk'
__version__ = '0.1'
__all__ = [
    'BayesianListener',
    'AuditoryRepresentation',
    'Barumerli2025',
    ]

from .bayesian_listener import BayesianListener
from .auditory_representation import AuditoryRepresentation, Barumerli2025

from . import metrics
from . import resample
from . import utils
from . import fitting


