"""Adapters that pull baseline + current sample sets from common vector stores.

Each adapter is gated behind an optional install extra. Importing one before
installing its extra raises ``ImportError`` with the exact `pip install` command
to fix it.
"""

from __future__ import annotations

__all__ = ["OpenSearchAdapter", "PgVectorAdapter", "PineconeAdapter"]


def __getattr__(name: str) -> object:
    """Lazy import so that an unused adapter never blows up on missing extras."""
    if name == "OpenSearchAdapter":
        from ragdrift.adapters.opensearch import OpenSearchAdapter

        return OpenSearchAdapter
    if name == "PgVectorAdapter":
        from ragdrift.adapters.pgvector import PgVectorAdapter

        return PgVectorAdapter
    if name == "PineconeAdapter":
        from ragdrift.adapters.pinecone import PineconeAdapter

        return PineconeAdapter
    raise AttributeError(f"module 'ragdrift.adapters' has no attribute {name!r}")
