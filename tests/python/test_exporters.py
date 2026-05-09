"""Exporter tests with mocked sinks."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import ragdrift


def _make_report() -> ragdrift.DriftReport:
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 8)).astype(np.float32)
    c = rng.normal(loc=2.0, size=(50, 8)).astype(np.float32)
    mon = ragdrift.RagDriftMonitor(
        embedding_threshold=0.05,
        data_threshold=None,
        response_threshold=None,
        confidence_threshold=None,
        query_threshold=None,
    )
    return mon.evaluate(baseline_embeddings=b, current_embeddings=c)


def test_cloudwatch_exporter_emits_metric_data() -> None:
    pytest.importorskip("boto3")
    from ragdrift.exporters import CloudWatchExporter

    client = MagicMock()
    exp = CloudWatchExporter(
        client=client,
        namespace="rag/drift",
        dimensions=[{"Name": "service", "Value": "ask-anything"}],
    )
    exp.record(_make_report())
    client.put_metric_data.assert_called_once()
    kwargs = client.put_metric_data.call_args.kwargs
    assert kwargs["Namespace"] == "rag/drift"
    metrics = kwargs["MetricData"]
    # 1 dimension scored => 2 metrics (score + exceeded).
    assert len(metrics) == 2
    names = {m["MetricName"] for m in metrics}
    assert "RagDrift_embedding" in names
    assert "RagDrift_embedding_exceeded" in names


def test_prometheus_exporter_sets_gauges() -> None:
    pc = pytest.importorskip("prometheus_client")
    from ragdrift.exporters import PrometheusExporter

    registry = pc.CollectorRegistry()
    exp = PrometheusExporter(registry=registry, service="ask-anything")
    exp.record(_make_report())
    body = pc.generate_latest(registry).decode()
    assert "ragdrift_score" in body
    assert "ragdrift_exceeded" in body
    assert 'dimension="embedding"' in body


def test_datadog_exporter_submits_series() -> None:
    pytest.importorskip("datadog_api_client")
    from ragdrift.exporters import DatadogExporter

    api = MagicMock()
    exp = DatadogExporter(metrics_api=api, tags=["service:ask-anything"])
    exp.record(_make_report())
    api.submit_metrics.assert_called_once()
