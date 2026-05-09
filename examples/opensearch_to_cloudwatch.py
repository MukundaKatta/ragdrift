"""End-to-end production loop: pull embeddings from OpenSearch, score drift,
push the result to CloudWatch.

Configure via environment variables, then::

    python examples/opensearch_to_cloudwatch.py

Required env vars:

    OPENCLAW_OS_HOST           e.g. https://search-foo.us-east-1.es.amazonaws.com
    OPENCLAW_OS_INDEX          index name carrying embeddings
    OPENCLAW_OS_EMBEDDING_FIELD  default: embedding
    OPENCLAW_OS_TIMESTAMP_FIELD  default: @timestamp
    OPENCLAW_BASELINE_START      ISO8601
    OPENCLAW_BASELINE_END        ISO8601
    OPENCLAW_CURRENT_START       ISO8601
    OPENCLAW_CURRENT_END         ISO8601
    OPENCLAW_AWS_REGION          default: us-east-1
    OPENCLAW_CW_NAMESPACE        default: rag/drift
"""

from __future__ import annotations

import os
import sys

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from ragdrift import DriftReport, EmbeddingDrift
from ragdrift.adapters import OpenSearchAdapter
from ragdrift.adapters.opensearch import OpenSearchWindow
from ragdrift.exporters import CloudWatchExporter


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        sys.exit(f"missing required env var: {name}")
    return v


def main() -> None:
    region = env("OPENCLAW_AWS_REGION", "us-east-1")
    host = env("OPENCLAW_OS_HOST")
    index = env("OPENCLAW_OS_INDEX")
    emb_field = env("OPENCLAW_OS_EMBEDDING_FIELD", "embedding")
    ts_field = env("OPENCLAW_OS_TIMESTAMP_FIELD", "@timestamp")
    cw_namespace = env("OPENCLAW_CW_NAMESPACE", "rag/drift")

    baseline = OpenSearchWindow(env("OPENCLAW_BASELINE_START"), env("OPENCLAW_BASELINE_END"))
    current = OpenSearchWindow(env("OPENCLAW_CURRENT_START"), env("OPENCLAW_CURRENT_END"))

    os_client = OpenSearch(
        hosts=[host],
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    adapter = OpenSearchAdapter(
        client=os_client,
        index=index,
        embedding_field=emb_field,
        timestamp_field=ts_field,
        page_size=1000,
    )
    base_emb, curr_emb = adapter.fetch_pair(baseline, current, sample_size=2000)
    print(f"fetched baseline={base_emb.shape} current={curr_emb.shape}")

    detector = EmbeddingDrift(threshold=0.05)
    score = detector.detect(base_emb, curr_emb)
    report = DriftReport([score], base_emb.shape[0], curr_emb.shape[0])
    print(f"score: {score!r}")

    cw = boto3.client("cloudwatch", region_name=region)
    CloudWatchExporter(
        client=cw,
        namespace=cw_namespace,
        dimensions=[{"Name": "index", "Value": index}],
    ).record(report)
    print("emitted to CloudWatch")


if __name__ == "__main__":
    main()
