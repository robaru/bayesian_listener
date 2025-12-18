import pytest


def test_import_bayesian_listener():
    try:
        import bayesian_listener           # noqa
    except ImportError:
        pytest.fail('import bayesian_listener failed')
