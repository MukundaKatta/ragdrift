"""Adapter for Pinecone serverless and pod-based indices.

Install with::

    pip install 'ragdrift[pinecone]'
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_INSTALL_HINT = (
    "Pinecone adapter requires the 'pinecone' extra. Install with: pip install 'ragdrift[pinecone]'"
)


class PineconeAdapter:
    """Pull embedding sample sets from a Pinecone index by id list or filter.

    Pinecone has no native time-window query; pre-segment your ids by capture
    window (e.g. by namespace) and pass the right id lists to ``fetch_pair``.

    Example::

        from pinecone import Pinecone
        from ragdrift.adapters import PineconeAdapter

        pc = Pinecone(api_key="...")
        index = pc.Index("rag-prod")
        adapter = PineconeAdapter(index=index)
        b, c = adapter.fetch_pair(
            baseline_ids=baseline_id_list,
            current_ids=current_id_list,
        )
    """

    def __init__(self, index: Any, batch_size: int = 100) -> None:
        try:
            import pinecone  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.index = index
        self.batch_size = batch_size

    def fetch_pair(
        self,
        baseline_ids: list[str],
        current_ids: list[str],
        namespace: str | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``(baseline_embeddings, current_embeddings)`` as float32 matrices."""
        b = self._fetch_ids(baseline_ids, namespace)
        c = self._fetch_ids(current_ids, namespace)
        return b, c

    def _fetch_ids(self, ids: list[str], namespace: str | None) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for i in range(0, len(ids), self.batch_size):
            batch = ids[i : i + self.batch_size]
            kwargs: dict[str, Any] = {"ids": batch}
            if namespace is not None:
                kwargs["namespace"] = namespace
            resp = self.index.fetch(**kwargs)
            vectors = getattr(resp, "vectors", None) or resp.get("vectors", {})
            for vid in batch:
                v = vectors.get(vid)
                if v is None:
                    continue
                values = getattr(v, "values", None) or v.get("values")
                if values is not None:
                    rows.append(list(values))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
