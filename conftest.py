"""Shared pytest fixtures for the bayesian_listener test suite."""
import pytest
import urllib.request
from pathlib import Path


def _get_sofa_path(participant="P0001"):
    repo_root = Path(__file__).parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    sofa_file = f"{participant}_FreeFieldCompMinPhase_48kHz.sofa"
    sofa_path = data_dir / sofa_file

    if not sofa_path.exists():
        url = (
            f"https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/"
            f"{participant}/HRTF/HRTF/48kHz/" + sofa_file
        )
        try:
            urllib.request.urlretrieve(url, sofa_path)
        except Exception as e:
            pytest.skip(f"SOFA file not available and download failed: {e}")

    return str(sofa_path)


@pytest.fixture(scope="session")
def sofa_path():
    """Path to the SONICOM P0001 SOFA file; downloads once per session."""
    return _get_sofa_path("P0001")


@pytest.fixture(scope="session")
def sofa_path_non_individual():
    """Path to the SONICOM P0002 SOFA file; downloads once per session."""
    return _get_sofa_path("P0002")
