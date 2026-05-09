"""Prometheus metric exporter.

Install with::

    pip install 'ragdrift[prometheus]'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragdrift._native import DriftReport

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

_INSTALL_HINT = (
    "Prometheus exporter requires the 'prometheus' extra. "
    "Install with: pip install 'ragdrift[prometheus]'"
)


class PrometheusExporter:
    """Maintain per-dimension gauges that the Prometheus client can scrape.

    Pass an existing ``CollectorRegistry`` to integrate with your own scrape
    endpoint, or omit it to use the global default registry.

    Example::

        from prometheus_client import CollectorRegistry, generate_latest
        from ragdrift.exporters import PrometheusExporter

        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry=registry, service="ask-anything")
        exporter.record(report)
        body = generate_latest(registry)
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        service: str = "default",
    ) -> None:
        try:
            from prometheus_client import CollectorRegistry, Gauge
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self._Gauge = Gauge
        self.registry = registry if registry is not None else CollectorRegistry()
        self.service = service
        self._score: Any = self._Gauge(
            "ragdrift_score",
            "Drift score by dimension.",
            labelnames=("dimension", "service"),
            registry=self.registry,
        )
        self._exceeded: Any = self._Gauge(
            "ragdrift_exceeded",
            "1 if drift exceeded the threshold for this dimension, else 0.",
            labelnames=("dimension", "service"),
            registry=self.registry,
        )

    def record(self, report: DriftReport) -> None:
        """Update gauges for every dimension present in the report."""
        for s in report.scores:
            self._score.labels(dimension=s.dimension, service=self.service).set(float(s.score))
            self._exceeded.labels(dimension=s.dimension, service=self.service).set(
                1.0 if s.exceeded else 0.0
            )
