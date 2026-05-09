"""CloudWatch metric exporter.

Install with::

    pip install 'ragdrift[aws]'
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ragdrift._native import DriftReport

_INSTALL_HINT = (
    "CloudWatch exporter requires the 'aws' extra. Install with: pip install 'ragdrift[aws]'"
)


class CloudWatchExporter:
    """Emit each drift dimension as a CloudWatch ``MetricData`` point.

    Dimensions are emitted as ``RagDrift_<dimension>`` (e.g. ``RagDrift_embedding``)
    so they show up as siblings in the CloudWatch console.

    Example::

        import boto3
        from ragdrift.exporters import CloudWatchExporter

        cw = boto3.client("cloudwatch", region_name="us-east-1")
        exporter = CloudWatchExporter(
            client=cw,
            namespace="rag/drift",
            dimensions=[{"Name": "service", "Value": "ask-anything"}],
        )
        exporter.record(report)
    """

    def __init__(
        self,
        client: Any,
        namespace: str = "rag/drift",
        dimensions: list[dict[str, str]] | None = None,
    ) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.client = client
        self.namespace = namespace
        self.dimensions = dimensions or []

    def record(self, report: DriftReport) -> None:
        """Emit one MetricData point per drift dimension in the report."""
        ts = datetime.fromtimestamp(report.timestamp, tz=timezone.utc)
        metric_data = []
        for s in report.scores:
            metric_data.append(
                {
                    "MetricName": f"RagDrift_{s.dimension}",
                    "Dimensions": self.dimensions,
                    "Timestamp": ts,
                    "Value": float(s.score),
                    "Unit": "None",
                }
            )
            metric_data.append(
                {
                    "MetricName": f"RagDrift_{s.dimension}_exceeded",
                    "Dimensions": self.dimensions,
                    "Timestamp": ts,
                    "Value": 1.0 if s.exceeded else 0.0,
                    "Unit": "Count",
                }
            )
        # CloudWatch caps PutMetricData at 1000 entries per call; we never exceed
        # 2 metrics * 5 dimensions = 10, so a single call is enough.
        if metric_data:
            self.client.put_metric_data(Namespace=self.namespace, MetricData=metric_data)
