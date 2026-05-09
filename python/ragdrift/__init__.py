"""ragdrift: five-dimensional drift detection for production RAG systems.

Public surface re-exports the Rust-backed detectors plus the high-level
``RagDriftMonitor`` that composes them into a single ``DriftReport``.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from numpy.typing import NDArray

from ragdrift._native import (
    BaselineSnapshot,
    ConfidenceDrift,
    DataDrift,
    DriftReport,
    DriftScore,
    EmbeddingDrift,
    QueryDrift,
    RagDriftError,
    ResponseDrift,
)

# PyPI distribution name is `ragdrift-py` (the bare `ragdrift` was taken),
# import name is `ragdrift`. Look up metadata under both — editable installs
# during local dev sometimes register under the import name instead.
try:
    __version__ = version("ragdrift-py")
except PackageNotFoundError:
    try:
        __version__ = version("ragdrift")
    except PackageNotFoundError:  # pragma: no cover - source checkout, never installed
        __version__ = "0.0.0+local"

__all__ = [
    "BaselineSnapshot",
    "ConfidenceDrift",
    "DataDrift",
    "DriftReport",
    "DriftScore",
    "EmbeddingDrift",
    "QueryDrift",
    "RagDriftError",
    "RagDriftMonitor",
    "ResponseDrift",
    "__version__",
]


class RagDriftMonitor:
    """Compose all five detectors into a single ``DriftReport``.

    Each detector is optional. Provide thresholds for the dimensions you care
    about; pass ``None`` (the default) to skip a dimension. Then call
    :meth:`evaluate` with whatever data you have for the current window — the
    monitor only evaluates the dimensions whose required inputs are present.

    Example:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> baseline = rng.normal(size=(200, 16)).astype(np.float32)
    >>> current = rng.normal(loc=2.0, size=(200, 16)).astype(np.float32)
    >>> mon = RagDriftMonitor(embedding_threshold=0.05)
    >>> report = mon.evaluate(
    ...     baseline_embeddings=baseline,
    ...     current_embeddings=current,
    ... )
    >>> report.any_exceeded()
    True
    """

    def __init__(
        self,
        *,
        embedding_threshold: float | None = 0.05,
        data_threshold: float | None = 0.10,
        response_threshold: float | None = 0.20,
        confidence_threshold: float | None = 0.20,
        query_threshold: float | None = 0.10,
        query_k: int = 8,
        seed: int = 0,
    ) -> None:
        self._embedding = (
            EmbeddingDrift(threshold=embedding_threshold, seed=seed)
            if embedding_threshold is not None
            else None
        )
        self._data = DataDrift(threshold=data_threshold) if data_threshold is not None else None
        self._response = (
            ResponseDrift(threshold=response_threshold, seed=seed)
            if response_threshold is not None
            else None
        )
        self._confidence = (
            ConfidenceDrift(threshold=confidence_threshold)
            if confidence_threshold is not None
            else None
        )
        self._query = (
            QueryDrift(threshold=query_threshold, k=query_k, seed=seed)
            if query_threshold is not None
            else None
        )

    def evaluate(
        self,
        *,
        baseline_embeddings: NDArray[np.float32] | None = None,
        current_embeddings: NDArray[np.float32] | None = None,
        baseline_features: NDArray[np.float64] | None = None,
        current_features: NDArray[np.float64] | None = None,
        baseline_response_lengths: Sequence[float] | None = None,
        current_response_lengths: Sequence[float] | None = None,
        baseline_confidences: Sequence[float] | None = None,
        current_confidences: Sequence[float] | None = None,
        baseline_correct: Sequence[bool] | None = None,
        current_correct: Sequence[bool] | None = None,
        baseline_query_embeddings: NDArray[np.float32] | None = None,
        current_query_embeddings: NDArray[np.float32] | None = None,
    ) -> DriftReport:
        """Run every configured detector that has matching inputs."""
        scores: list[DriftScore] = []
        max_baseline = 0
        max_current = 0

        if (
            self._embedding is not None
            and baseline_embeddings is not None
            and current_embeddings is not None
        ):
            scores.append(self._embedding.detect(baseline_embeddings, current_embeddings))
            max_baseline = max(max_baseline, baseline_embeddings.shape[0])
            max_current = max(max_current, current_embeddings.shape[0])

        if (
            self._data is not None
            and baseline_features is not None
            and current_features is not None
        ):
            scores.append(self._data.detect(baseline_features, current_features))
            max_baseline = max(max_baseline, baseline_features.shape[0])
            max_current = max(max_current, current_features.shape[0])

        if (
            self._response is not None
            and baseline_response_lengths is not None
            and current_response_lengths is not None
        ):
            scores.append(
                self._response.detect(
                    list(baseline_response_lengths), list(current_response_lengths)
                )
            )
            max_baseline = max(max_baseline, len(baseline_response_lengths))
            max_current = max(max_current, len(current_response_lengths))

        if (
            self._confidence is not None
            and baseline_confidences is not None
            and current_confidences is not None
        ):
            if baseline_correct is not None and current_correct is not None:
                scores.append(
                    self._confidence.detect_with_correctness(
                        list(baseline_confidences),
                        list(baseline_correct),
                        list(current_confidences),
                        list(current_correct),
                    )
                )
            else:
                scores.append(
                    self._confidence.detect(list(baseline_confidences), list(current_confidences))
                )
            max_baseline = max(max_baseline, len(baseline_confidences))
            max_current = max(max_current, len(current_confidences))

        if (
            self._query is not None
            and baseline_query_embeddings is not None
            and current_query_embeddings is not None
        ):
            scores.append(self._query.detect(baseline_query_embeddings, current_query_embeddings))
            max_baseline = max(max_baseline, baseline_query_embeddings.shape[0])
            max_current = max(max_current, current_query_embeddings.shape[0])

        return DriftReport(scores, max_baseline, max_current)
