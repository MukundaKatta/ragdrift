"""Adapter tests with mocked vector-store clients."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_opensearch_adapter_reads_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opensearchpy")
    from ragdrift.adapters import OpenSearchAdapter
    from ragdrift.adapters.opensearch import OpenSearchWindow

    client = MagicMock()
    page1 = {
        "hits": {
            "hits": [
                {"_source": {"embedding": [0.1, 0.2]}, "sort": [1]},
                {"_source": {"embedding": [0.3, 0.4]}, "sort": [2]},
            ]
        }
    }
    page2: dict[str, Any] = {"hits": {"hits": []}}
    client.search.side_effect = [page1, page2]

    adapter = OpenSearchAdapter(
        client=client,
        index="rag",
        embedding_field="embedding",
        timestamp_field="@timestamp",
        page_size=2,
    )
    b, _c = adapter.fetch_pair(
        baseline=OpenSearchWindow("2026-04-01", "2026-04-02"),
        current=OpenSearchWindow("2026-05-01", "2026-05-02"),
        sample_size=2,
    )
    assert b.shape == (2, 2)
    # search called twice (one per window): each fetch starts fresh.
    # The first window consumed both pages; the second window also tries — total >= 3 calls.
    assert client.search.call_count >= 2


def test_pgvector_adapter_executes_query() -> None:
    pytest.importorskip("sqlalchemy")
    from ragdrift.adapters import PgVectorAdapter

    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn

    rows = [(np.array([0.1, 0.2, 0.3]),) for _ in range(3)]
    conn.execute.return_value.fetchall.return_value = rows

    adapter = PgVectorAdapter(
        engine=engine,
        table="rag_logs",
        embedding_column="embedding",
        timestamp_column="created_at",
    )
    b, _c = adapter.fetch_pair(
        baseline=("2026-04-01", "2026-04-02"),
        current=("2026-05-01", "2026-05-02"),
        sample_size=10,
    )
    assert b.shape == (3, 3)
    assert b.dtype == np.float32


def test_pinecone_adapter_chunks_fetches() -> None:
    pytest.importorskip("pinecone")
    from ragdrift.adapters import PineconeAdapter

    index = MagicMock()
    # Mimic the v3 client response shape: an object with a `.vectors` dict mapping id -> Vector
    fake_vectors = {f"id{i}": MagicMock(values=[0.1, 0.2, 0.3]) for i in range(5)}
    index.fetch.return_value = MagicMock(
        vectors=fake_vectors,
        get=lambda k, default=None: fake_vectors if k == "vectors" else default,
    )

    adapter = PineconeAdapter(index=index, batch_size=2)
    ids = list(fake_vectors.keys())
    b, c = adapter.fetch_pair(baseline_ids=ids, current_ids=ids[:2])
    assert b.shape == (5, 3)
    assert c.shape == (2, 3)
