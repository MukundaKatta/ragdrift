"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture
def baseline_emb(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(200, 16)).astype(np.float32)


@pytest.fixture
def current_emb_drifted(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=2.0, size=(200, 16)).astype(np.float32)


@pytest.fixture
def baseline_features(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(500, 4)).astype(np.float64)


@pytest.fixture
def current_features_drifted(rng: np.random.Generator, baseline_features: np.ndarray) -> np.ndarray:
    arr = baseline_features.copy()
    arr[:, 2] += 3.0
    return arr.astype(np.float64)
