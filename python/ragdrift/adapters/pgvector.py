"""Adapter for Postgres + pgvector.

Install with::

    pip install 'ragdrift[pgvector]'
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_INSTALL_HINT = (
    "pgvector adapter requires the 'pgvector' extra. Install with: pip install 'ragdrift[pgvector]'"
)


class PgVectorAdapter:
    """Pull embedding sample sets from a Postgres ``vector`` column.

    Uses raw SQLAlchemy text queries (no ORM dependency) so it composes with
    whatever schema you already have. Pass any object that satisfies
    ``sqlalchemy.engine.Connectable``.

    Example::

        from sqlalchemy import create_engine
        from ragdrift.adapters import PgVectorAdapter

        engine = create_engine("postgresql+psycopg://user:pw@host/db")
        adapter = PgVectorAdapter(
            engine=engine,
            table="rag_logs",
            embedding_column="embedding",
            timestamp_column="created_at",
        )
        b, c = adapter.fetch_pair(
            baseline=("2026-04-01", "2026-04-08"),
            current=("2026-05-01", "2026-05-08"),
            sample_size=2000,
        )
    """

    def __init__(
        self,
        engine: Any,
        table: str,
        embedding_column: str = "embedding",
        timestamp_column: str = "created_at",
    ) -> None:
        try:
            import sqlalchemy  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.engine = engine
        self.table = table
        self.embedding_column = embedding_column
        self.timestamp_column = timestamp_column

    def fetch_pair(
        self,
        baseline: tuple[str, str],
        current: tuple[str, str],
        sample_size: int = 1000,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return ``(baseline_embeddings, current_embeddings)`` as float32 matrices."""
        b = self._fetch_window(baseline[0], baseline[1], sample_size)
        c = self._fetch_window(current[0], current[1], sample_size)
        return b, c

    def _fetch_window(self, start: str, end: str, sample_size: int) -> NDArray[np.float32]:
        from sqlalchemy import text

        # Identifiers are interpolated; bind only the values. Callers control the
        # table and column names — they should be trusted (config), not user input.
        sql = text(
            f"SELECT {self.embedding_column} "
            f"FROM {self.table} "
            f"WHERE {self.timestamp_column} >= :start "
            f"AND {self.timestamp_column} < :end "
            f"ORDER BY {self.timestamp_column} "
            f"LIMIT :limit"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"start": start, "end": end, "limit": sample_size}).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        # pgvector returns lists of float; np.asarray handles the rest.
        data = [list(r[0]) for r in rows]
        return np.asarray(data, dtype=np.float32)
