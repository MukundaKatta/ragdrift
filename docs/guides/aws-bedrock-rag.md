# AWS Bedrock RAG

End-to-end wiring for a Bedrock + OpenSearch Serverless + CloudWatch RAG stack.

## What you need

- An OpenSearch Serverless collection with a k-NN index that stores per-call
  embeddings and a timestamp field.
- IAM credentials with read access to that collection and `cloudwatch:PutMetricData`.
- `pip install 'ragdrift[opensearch,aws]'`

## The loop

```python
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection

from ragdrift import EmbeddingDrift, DriftReport
from ragdrift.adapters import OpenSearchAdapter
from ragdrift.adapters.opensearch import OpenSearchWindow
from ragdrift.exporters import CloudWatchExporter

os_client = OpenSearch(
    hosts=[os.environ["OPENCLAW_OS_HOST"]],
    connection_class=RequestsHttpConnection,
    timeout=30,
)
adapter = OpenSearchAdapter(
    client=os_client,
    index="rag-prod-embeddings",
    embedding_field="embedding",
    timestamp_field="@timestamp",
)
baseline_emb, current_emb = adapter.fetch_pair(
    baseline=OpenSearchWindow("2026-04-01T00:00:00Z", "2026-04-08T00:00:00Z"),
    current=OpenSearchWindow("2026-05-01T00:00:00Z", "2026-05-08T00:00:00Z"),
    sample_size=2000,
)

score = EmbeddingDrift(threshold=0.05).detect(baseline_emb, current_emb)
report = DriftReport([score], baseline_emb.shape[0], current_emb.shape[0])

cw = boto3.client("cloudwatch", region_name="us-east-1")
CloudWatchExporter(
    client=cw,
    namespace="rag/drift",
    dimensions=[{"Name": "service", "Value": "ask-anything"}],
).record(report)
```

## Schedule it

Run hourly via EventBridge -> Lambda, or as a sidecar on your ingestion service.
The Rust core takes single-digit milliseconds for 2000-vector samples, so this
is not a meaningful tax on your pipeline.

## Alarming

The exporter publishes `RagDrift_embedding` (raw score) and
`RagDrift_embedding_exceeded` (0/1) per dimension. Alarm on the raw score with
your own threshold, or on `_exceeded == 1` if you want to honor the threshold
configured in code.
