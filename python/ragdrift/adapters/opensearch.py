"""Adapter for AWS OpenSearch (and OpenSearch Serverless) k-NN indices.

Install with::

    pip install 'ragdrift[opensearch]'
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from opensearchpy import OpenSearch

_INSTALL_HINT = (
    "OpenSearch adapter requires the 'opensearch' extra. "
    "Install with: pip install 'ragdrift[opensearch]'"
)


@dataclass
class OpenSearchWindow:
    """A time window expressed as inclusive ISO-8601 timestamps."""

    start: str
    end: str


class OpenSearchAdapter:
    """Pull embedding sample sets from an OpenSearch k-NN index.

    The adapter expects each document to carry an embedding field of fixed
    dimension and a timestamp field used to slice baseline vs. current windows.

    Example::

        from opensearchpy import OpenSearch
        from ragdrift.adapters import OpenSearchAdapter, OpenSearchWindow

        client = OpenSearch(hosts=[...], http_auth=(...))
        adapter = OpenSearchAdapter(
            client=client,
            index="rag-prod-embeddings",
            embedding_field="embedding",
            timestamp_field="@timestamp",
        )
        b, c = adapter.fetch_pair(
            baseline=OpenSearchWindow("2026-04-01T00:00:00Z", "2026-04-08T00:00:00Z"),
            current=OpenSearchWindow("2026-05-01T00:00:00Z", "2026-05-08T00:00:00Z"),
            sample_size=2000,
        )
    """

    def __init__(
        self,
        client: OpenSearch,
        index: str,
        embedding_field: str = "embedding",
        timestamp_field: str = "@timestamp",
        page_size: int = 1000,
    ) -> None:
        try:
            import opensearchpy  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(_INSTALL_HINT) from e
        self.client = client
        self.index = index
        self.embedding_field = embedding_field
        self.timestamp_field = timestamp_field
        self.page_size = page_size

    def fetch_pair(
        self,
        baseline: OpenSearchWindow,
        current: OpenSearchWindow,
        sample_size: int = 1000,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``(baseline_embeddings, current_embeddings)`` as float32 matrices."""
        b = self._fetch_window(baseline, sample_size)
        c = self._fetch_window(current, sample_size)
        return b, c

    def _fetch_window(self, window: OpenSearchWindow, max_docs: int) -> NDArray[np.float32]:
        body: dict[str, Any] = {
            "size": min(self.page_size, max_docs),
            "_source": [self.embedding_field],
            "query": {
                "range": {
                    self.timestamp_field: {
                        "gte": window.start,
                        "lte": window.end,
                    }
                }
            },
            "sort": [{self.timestamp_field: "asc"}],
        }
        rows: list[Sequence[float]] = []
        search_after: list[Any] | None = None
        while len(rows) < max_docs:
            if search_after is not None:
                body["search_after"] = search_after
            resp = self.client.search(index=self.index, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                emb = h["_source"].get(self.embedding_field)
                if emb is None:
                    continue
                rows.append(emb)
                if len(rows) >= max_docs:
                    break
            search_after = hits[-1].get("sort")
            if not search_after:
                break
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
