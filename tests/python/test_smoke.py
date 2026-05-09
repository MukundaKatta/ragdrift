"""Smoke test: import the package, instantiate every detector, run on tiny data."""

from __future__ import annotations

import numpy as np
import ragdrift


def test_version_string_is_set() -> None:
    assert isinstance(ragdrift.__version__, str)
    assert ragdrift.__version__.count(".") >= 1


def test_every_detector_runs_on_small_data() -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 4)).astype(np.float32)
    feats = rng.normal(size=(20, 3)).astype(np.float64)

    s = ragdrift.EmbeddingDrift(threshold=1.0).detect(emb, emb)
    assert s.dimension == "embedding"
    assert s.score >= 0.0

    s = ragdrift.DataDrift(threshold=1.0).detect(feats, feats)
    assert s.dimension == "data"

    s = ragdrift.ResponseDrift(threshold=1.0).detect([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert s.dimension == "response"

    s = ragdrift.ConfidenceDrift(threshold=1.0).detect([0.5, 0.6], [0.5, 0.6])
    assert s.dimension == "confidence"

    s = ragdrift.QueryDrift(threshold=1.0, k=2).detect(emb, emb)
    assert s.dimension == "query"


def test_drift_score_repr_and_dict() -> None:
    s = ragdrift.EmbeddingDrift(threshold=0.05).detect(
        np.zeros((5, 3), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
    )
    assert "DriftScore" in repr(s)
    d = s.to_dict()
    assert d["dimension"] == "embedding"
    assert "score" in d


def test_report_json_roundtrips() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(size=(50, 8)).astype(np.float32)
    c = rng.normal(loc=0.5, size=(50, 8)).astype(np.float32)
    mon = ragdrift.RagDriftMonitor(
        embedding_threshold=0.05,
        data_threshold=None,
        response_threshold=None,
        confidence_threshold=None,
        query_threshold=None,
    )
    rep = mon.evaluate(baseline_embeddings=b, current_embeddings=c)
    s = rep.to_json()
    assert "embedding" in s.lower()
