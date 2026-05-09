"""Exporters that ship a ``DriftReport`` to a metrics backend.

Each exporter is gated behind an optional install extra. Importing one before
installing its extra raises ``ImportError`` with the exact ``pip install`` command.
"""

from __future__ import annotations

__all__ = ["CloudWatchExporter", "DatadogExporter", "PrometheusExporter"]


def __getattr__(name: str) -> object:
    if name == "CloudWatchExporter":
        from ragdrift.exporters.cloudwatch import CloudWatchExporter

        return CloudWatchExporter
    if name == "PrometheusExporter":
        from ragdrift.exporters.prometheus import PrometheusExporter

        return PrometheusExporter
    if name == "DatadogExporter":
        from ragdrift.exporters.datadog import DatadogExporter

        return DatadogExporter
    raise AttributeError(f"module 'ragdrift.exporters' has no attribute {name!r}")
