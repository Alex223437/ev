import sys
from pathlib import Path

import numpy as np
import pytest

# Make the task packages (ga, problems, experiments) importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
