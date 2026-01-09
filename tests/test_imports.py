import pytest


def test_import_bayesian_listener():
    """Test package import"""
    try:
        import bayesian_listener  # noqa
    except ImportError:
        pytest.fail('import bayesian_listener failed')


def test_import_bayesian_listener_class():
    """Test BayesianListener class import"""
    try:
        from bayesian_listener import BayesianListener  # noqa
    except ImportError:
        pytest.fail('from bayesian_listener import BayesianListener failed')


def test_import_metrics():
    """Test import of the metrics module"""
    try:
        from bayesian_listener import metrics  # noqa
    except ImportError:
        pytest.fail('from bayesian_listener import metrics failed')


def test_import_resample():
    """Test import of the resample module"""
    try:
        from bayesian_listener import resample  # noqa
    except ImportError:
        pytest.fail('from bayesian_listener import resample failed')


def test_import_utils():
    """Test import of the utils module"""
    try:
        from bayesian_listener import utils  # noqa
    except ImportError:
        pytest.fail('from bayesian_listener import utils failed')
