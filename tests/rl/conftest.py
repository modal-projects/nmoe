"""Pytest configuration for RL tests."""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """Get CUDA device if available."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
