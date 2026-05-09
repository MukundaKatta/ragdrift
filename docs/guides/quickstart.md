# Quickstart

Five minutes from `pip install` to a working drift report on synthetic data.

## Install

```bash
pip install ragdrift
```

## Run

Save as `quickstart.py`:

```python
import numpy as np
from ragdrift import RagDriftMonitor

rng = np.random.default_rng(seed=0)
baseline = rng.normal(size=(500, 32)).astype(np.float32)
current  = rng.normal(loc=1.0, size=(500, 32)).astype(np.float32)

mon = RagDriftMonitor(embedding_threshold=0.05)
report = mon.evaluate(
    baseline_embeddings=baseline,
    current_embeddings=current,
)
for s in report.scores:
    print(f"{s.dimension}: score={s.score:.4f} exceeded={s.exceeded}")
```

```text
embedding: score=1.0247 exceeded=True
```

## What just happened

You created two embedding sets, ran `EmbeddingDrift` (which composes MMD² with
the RBF kernel and sliced Wasserstein), and got a single non-negative score
that crossed your threshold of 0.05.

## Adding more dimensions

Pass more inputs to `evaluate` to enable additional detectors:

```python
report = mon.evaluate(
    baseline_embeddings=base_emb,
    current_embeddings=curr_emb,
    baseline_response_lengths=base_lengths,
    current_response_lengths=curr_lengths,
    baseline_confidences=base_conf,
    current_confidences=curr_conf,
)
```

The monitor only evaluates dimensions where both baseline and current are
provided, so you can roll out one dimension at a time.
