"""Datadog metric exporter.

Install with::

    pip install 'ragdrift[datadog]'
"""

from __future__ import annotations

from typing import Any

from ragdrift._native import DriftReport

_INSTALL_HINT = (
    "Datadog exporter requires the 'datadog' extra. Install with: pip install 'ragdrift[datadog]'"
)


class DatadogExporter:
    """Emit drift dimensions as Datadog gauge metrics via the official v1 API.

    Example::

        from datadog_api_client import ApiClient, Configuration
        from datadog_api_client.v1.api.metrics_api import MetricsApi
        from ragdrift.exporters import DatadogExporter

        config = Configuration()
        with ApiClient(config) as client:
            api = MetricsApi(client)
            exporter = DatadogExporter(metrics_api=api, tags=["service:ask-anything"])
            exporter.record(report)
    """

    def __init__(
        self,
        metrics_api: Any,
        tags: list[str] | None = None,
    ) -> None:
        try:
            from datadog_api_client.v1.model.metrics_payload import MetricsPayload
            from datadog_api_client.v1.model.point import Point
            from datadog_api_client.v1.model.series import Series
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self._MetricsPayload = MetricsPayload
        self._Point = Point
        self._Series = Series
        self.metrics_api = metrics_api
        self.tags = list(tags) if tags else []

    def _point(self, ts: float, value: float) -> Any:
        # `Point` is a ModelSimple wrapping a [timestamp, value] pair.
        return self._Point([ts, value])  # type: ignore[no-untyped-call]

    def record(self, report: DriftReport) -> None:
        """Submit one Datadog series per drift dimension in the report."""
        ts = float(report.timestamp)
        series_list = []
        for s in report.scores:
            series_list.append(
                self._Series(
                    metric=f"ragdrift.{s.dimension}.score",
                    points=[self._point(ts, float(s.score))],
                    tags=self.tags,
                    type="gauge",
                )
            )
            series_list.append(
                self._Series(
                    metric=f"ragdrift.{s.dimension}.exceeded",
                    points=[self._point(ts, 1.0 if s.exceeded else 0.0)],
                    tags=self.tags,
                    type="gauge",
                )
            )
        payload = self._MetricsPayload(series=series_list)
        self.metrics_api.submit_metrics(body=payload)
