"""Quickstart: synthetic data, all five detectors, print a report.

Run::

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
from ragdrift import RagDriftMonitor


def main() -> None:
    rng = np.random.default_rng(seed=0)

    # Baseline: 500 RAG calls from last week.
    baseline_emb = rng.normal(size=(500, 32)).astype(np.float32)
    baseline_features = rng.normal(size=(500, 4)).astype(np.float64)
    baseline_lengths = rng.integers(80, 200, size=500).astype(float).tolist()
    baseline_confs = rng.uniform(0.5, 0.95, size=500).tolist()
    baseline_queries = rng.normal(size=(500, 32)).astype(np.float32)

    # Current: 500 RAG calls from this week. Embeddings have shifted, lengths
    # are longer, and the confidence distribution has compressed toward 1.0.
    current_emb = rng.normal(loc=1.0, size=(500, 32)).astype(np.float32)
    current_features = baseline_features.copy()
    current_features[:, 2] += 2.0  # one feature drifted
    current_lengths = [x + 80 for x in baseline_lengths]
    current_confs = [min(0.99, x + 0.1) for x in baseline_confs]
    current_queries = rng.normal(loc=0.5, size=(500, 32)).astype(np.float32)

    monitor = RagDriftMonitor(
        embedding_threshold=0.5,
        data_threshold=0.2,
        response_threshold=0.2,
        confidence_threshold=0.2,
        query_threshold=0.5,
        query_k=8,
    )
    report = monitor.evaluate(
        baseline_embeddings=baseline_emb,
        current_embeddings=current_emb,
        baseline_features=baseline_features,
        current_features=current_features,
        baseline_response_lengths=baseline_lengths,
        current_response_lengths=current_lengths,
        baseline_confidences=baseline_confs,
        current_confidences=current_confs,
        baseline_query_embeddings=baseline_queries,
        current_query_embeddings=current_queries,
    )

    print(f"Drift report: max_score={report.max_score():.4f} any_exceeded={report.any_exceeded()}")
    for s in report.scores:
        flag = " ALERT" if s.exceeded else ""
        print(
            f"  {s.dimension:11s}  score={s.score:8.4f}  "
            f"threshold={s.threshold:.4f}  method={s.method}{flag}"
        )


if __name__ == "__main__":
    main()
