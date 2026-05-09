# ragdrift

Your RAG system is in production. Embeddings drift. Retrieval quality silently
degrades. By the time users complain, it has been broken for weeks.

`ragdrift` watches five dimensions at once — **data**, **embedding**, **response**,
**confidence**, and **query-pattern** — and gives you a single report you can
threshold-alert on. Rust core for the math, Python for the glue.

## Install

```bash
pip install ragdrift-py
```

Optional adapters and exporters:

```bash
pip install 'ragdrift-py[opensearch,aws]'    # OpenSearch + CloudWatch
pip install 'ragdrift-py[pgvector,prometheus]'
pip install 'ragdrift-py[pinecone,datadog]'
```

## 30-second example

```python
import numpy as np
from ragdrift import RagDriftMonitor

baseline = np.load("baseline_embeddings.npy")  # (n, 768) float32
current  = np.load("current_embeddings.npy")

monitor = RagDriftMonitor(embedding_threshold=0.05)
report = monitor.evaluate(
    baseline_embeddings=baseline,
    current_embeddings=current,
)

if report.any_exceeded():
    print(report.to_json())
```

## Where to next

- [What is drift?](concepts/what-is-drift.md) — the production rationale.
- [Five dimensions](concepts/five-dimensions.md) — what each detector measures and the math behind it.
- [Quickstart](guides/quickstart.md) — first end-to-end run on synthetic data.
- [AWS Bedrock RAG](guides/aws-bedrock-rag.md) — wiring it into a Bedrock + OpenSearch + CloudWatch stack.
