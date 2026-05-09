"""Per-detector behavioral tests over 3 drift scenarios: none, mild, severe."""

from __future__ import annotations

import numpy as np
import pytest
import ragdrift


def _emb(rng: np.random.Generator, n: int, d: int, shift: float) -> np.ndarray:
    return (rng.normal(size=(n, d)) + shift).astype(np.float32)


@pytest.mark.parametrize(
    "shift,expected_exceeded",
    [(0.0, False), (0.5, False), (3.0, True)],
)
def test_embedding_drift_scales_with_shift(
    rng: np.random.Generator, shift: float, expected_exceeded: bool
) -> None:
    base = _emb(rng, 200, 16, 0.0)
    curr = _emb(rng, 200, 16, shift)
    det = ragdrift.EmbeddingDrift(threshold=2.0, n_projections=64, seed=0)
    s = det.detect(base, curr)
    assert s.exceeded == expected_exceeded, f"score={s.score} shift={shift}"


@pytest.mark.parametrize(
    "shift,expected_exceeded",
    [(0.0, False), (0.5, False), (5.0, True)],
)
def test_data_drift_per_feature_shift(
    rng: np.random.Generator, shift: float, expected_exceeded: bool
) -> None:
    base = rng.normal(size=(500, 4)).astype(np.float64)
    curr = base.copy()
    curr[:, 2] += shift
    det = ragdrift.DataDrift(threshold=0.3)
    s = det.detect(base, curr)
    assert s.exceeded == expected_exceeded, f"score={s.score} shift={shift}"


@pytest.mark.parametrize(
    "scenario,expected_exceeded",
    [("none", False), ("mild", False), ("severe", True)],
)
def test_response_length_drift(
    rng: np.random.Generator, scenario: str, expected_exceeded: bool
) -> None:
    base = rng.integers(50, 200, size=500).astype(float).tolist()
    if scenario == "none":
        curr = list(base)
    elif scenario == "mild":
        curr = [x + 5 for x in base]
    else:
        curr = [x + 200 for x in base]
    det = ragdrift.ResponseDrift(threshold=0.3, seed=0)
    s = det.detect(base, curr)
    assert s.exceeded == expected_exceeded, f"score={s.score} {scenario}"


@pytest.mark.parametrize(
    "scenario,expected_exceeded",
    [("none", False), ("mild", False), ("severe", True)],
)
def test_confidence_drift(rng: np.random.Generator, scenario: str, expected_exceeded: bool) -> None:
    base = rng.uniform(0.4, 0.9, size=500).tolist()
    if scenario == "none":
        curr = list(base)
    elif scenario == "mild":
        curr = [min(0.99, x + 0.05) for x in base]
    else:
        curr = [0.99] * 500
    det = ragdrift.ConfidenceDrift(threshold=0.5)
    s = det.detect(base, curr)
    assert s.exceeded == expected_exceeded, f"score={s.score} {scenario}"


@pytest.mark.parametrize(
    "scenario,expected_exceeded",
    [("none", False), ("mild", False), ("severe", True)],
)
def test_query_drift(rng: np.random.Generator, scenario: str, expected_exceeded: bool) -> None:
    # Two well-separated baseline clusters in 8D.
    a = rng.normal(loc=0.0, scale=0.1, size=(150, 8))
    b = rng.normal(loc=10.0, scale=0.1, size=(150, 8))
    base = np.vstack([a, b]).astype(np.float32)
    if scenario == "none":
        curr = base.copy()
    elif scenario == "mild":
        # 60/40 split instead of 50/50
        curr_a = rng.normal(loc=0.0, scale=0.1, size=(180, 8))
        curr_b = rng.normal(loc=10.0, scale=0.1, size=(120, 8))
        curr = np.vstack([curr_a, curr_b]).astype(np.float32)
    else:
        curr = rng.normal(loc=0.0, scale=0.1, size=(300, 8)).astype(np.float32)
    det = ragdrift.QueryDrift(threshold=1.0, k=2, seed=7)
    s = det.detect(base, curr)
    assert s.exceeded == expected_exceeded, f"score={s.score} {scenario}"


def test_dimension_mismatch_raises() -> None:
    a = np.zeros((10, 4), dtype=np.float32)
    b = np.zeros((10, 5), dtype=np.float32)
    det = ragdrift.EmbeddingDrift(threshold=0.05)
    with pytest.raises(ragdrift.RagDriftError):
        det.detect(a, b)


def test_monitor_skips_dimensions_without_inputs() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 8)).astype(np.float32)
    c = rng.normal(loc=2.0, size=(50, 8)).astype(np.float32)
    mon = ragdrift.RagDriftMonitor()
    rep = mon.evaluate(baseline_embeddings=b, current_embeddings=c)
    # Only embedding given, only embedding scored.
    assert len(rep.scores) == 1
    assert rep.scores[0].dimension == "embedding"
