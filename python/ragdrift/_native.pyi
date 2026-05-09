"""Hand-written type stubs for the Rust-backed `ragdrift._native` module.

Generated stubs are unreliable; this file is the source of truth for type
information that ships with the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

class RagDriftError(Exception):
    """Raised by every detector for invalid input or numerical instability."""

class DriftScore:
    """A single per-dimension drift measurement."""

    @property
    def dimension(self) -> str: ...
    @property
    def score(self) -> float: ...
    @property
    def threshold(self) -> float: ...
    @property
    def exceeded(self) -> bool: ...
    @property
    def method(self) -> str: ...
    def to_dict(self) -> dict[str, float | bool | str]: ...
    def __repr__(self) -> str: ...

class DriftReport:
    """A unified report aggregating per-dimension scores."""

    def __init__(
        self,
        scores: Sequence[DriftScore],
        sample_size_baseline: int,
        sample_size_current: int,
    ) -> None: ...
    @property
    def scores(self) -> list[DriftScore]: ...
    @property
    def timestamp(self) -> int: ...
    @property
    def sample_size_baseline(self) -> int: ...
    @property
    def sample_size_current(self) -> int: ...
    def any_exceeded(self) -> bool: ...
    def max_score(self) -> float: ...
    def to_json(self) -> str: ...
    def __repr__(self) -> str: ...

class BaselineSnapshot:
    """A captured baseline used as the reference distribution for future detection."""

    def __init__(self) -> None: ...
    @property
    def captured_at(self) -> int: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(s: str) -> BaselineSnapshot: ...

class EmbeddingDrift:
    """MMD² + sliced Wasserstein-1 detector for embedding distributions."""

    def __init__(self, threshold: float = 0.05, n_projections: int = 64, seed: int = 0) -> None: ...
    def detect(
        self,
        baseline: NDArray[np.float32],
        current: NDArray[np.float32],
    ) -> DriftScore: ...

class DataDrift:
    """Per-feature KS + PSI detector for tabular data."""

    def __init__(self, threshold: float = 0.10, n_bins: int = 10) -> None: ...
    def detect(
        self,
        baseline: NDArray[np.float64],
        current: NDArray[np.float64],
    ) -> DriftScore: ...

class ResponseDrift:
    """KS test on response lengths, optionally combined with sliced Wasserstein on embeddings."""

    def __init__(self, threshold: float = 0.20, seed: int = 0) -> None: ...
    def detect(
        self,
        baseline_lengths: Sequence[float],
        current_lengths: Sequence[float],
    ) -> DriftScore: ...
    def detect_with_embeddings(
        self,
        baseline_lengths: Sequence[float],
        current_lengths: Sequence[float],
        baseline_embeddings: NDArray[np.float32],
        current_embeddings: NDArray[np.float32],
    ) -> DriftScore: ...

class ConfidenceDrift:
    """KS on confidence score distributions plus optional ECE delta."""

    def __init__(self, threshold: float = 0.20, n_calib_bins: int = 10) -> None: ...
    def detect(
        self,
        baseline_confidences: Sequence[float],
        current_confidences: Sequence[float],
    ) -> DriftScore: ...
    def detect_with_correctness(
        self,
        baseline_confidences: Sequence[float],
        baseline_correct: Sequence[bool],
        current_confidences: Sequence[float],
        current_correct: Sequence[bool],
    ) -> DriftScore: ...

class QueryDrift:
    """k-means + symmetric KL detector for query intent shift."""

    def __init__(self, threshold: float = 0.10, k: int = 8, seed: int = 0) -> None: ...
    def detect(
        self,
        baseline: NDArray[np.float32],
        current: NDArray[np.float32],
    ) -> DriftScore: ...
