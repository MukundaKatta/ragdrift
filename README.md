# ragdrift

Five-dimensional drift detection for production RAG systems. Rust core, Python bindings.

[![CI](https://github.com/MukundaKatta/ragdrift/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/ragdrift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragdrift.svg)](https://pypi.org/project/ragdrift/)
[![crates.io](https://img.shields.io/crates/v/ragdrift-core.svg)](https://crates.io/crates/ragdrift-core)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mukundakatta.github.io/ragdrift/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## The problem

Your RAG system is in production. The retriever silently starts returning slightly
worse results because your corpus drifted. The model gets more confident but no
better calibrated. The mix of intents your users send shifts week-over-week. None
of this surfaces as a 5xx, none of it fails your tests, none of it fires your
latency alerts.

By the time someone notices in a Slack thread, it has been broken for weeks.

`ragdrift` watches **five dimensions at once** and gives you a single threshold-able
report. The numerical core is Rust; the Python wheel is a thin PyO3 binding that
releases the GIL on the heavy paths.

## Install

```bash
pip install ragdrift-py
```

> The PyPI distribution is **`ragdrift-py`** (the bare `ragdrift` name on PyPI
> was taken by an unrelated project). The Python import is still
> `import ragdrift`, same convention as `pyyaml`/`yaml` or `opencv-python`/`cv2`.
> The Rust crates on crates.io are unaffected — `cargo add ragdrift` works.

Optional extras:

```bash
pip install 'ragdrift-py[opensearch,aws]'        # adapter + CloudWatch exporter
pip install 'ragdrift-py[pgvector,prometheus]'
pip install 'ragdrift-py[pinecone,datadog]'
```

## 30-second quickstart

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

## The five dimensions

| Dimension     | Method                          | What it catches |
|---------------|---------------------------------|-----------------|
| **Embedding** | MMD² (RBF) + sliced Wasserstein | corpus or model embedding distribution shift |
| **Data**      | per-feature KS + PSI            | tabular feature drift (latency, retrieval count, etc.) |
| **Response**  | KS on lengths, optional SW      | response length / semantic shift |
| **Confidence**| KS, optional ECE delta          | confidence score collapse, calibration breakage |
| **Query**     | k-means + symmetric KL          | intent-mix shift in incoming queries |

See [docs](https://mukundakatta.github.io/ragdrift/concepts/five-dimensions/) for
the math and citations.

## Why not X?

- **Arize Phoenix** is great for embedding visualization and notebook-style
  exploration. It does not give you a single Rust-fast scalar you can alert on
  from Lambda.
- **Evidently** is excellent for tabular drift and report generation. It does
  not have a sliced-Wasserstein-on-embeddings primitive in the hot path.
- **WhyLabs / NannyML** are mature monitoring platforms — useful, vendor-tied,
  not embeddable as a library inside your service.
- **Roll your own**: most teams write the first 80% in 200 lines of numpy and
  then hit a wall when MMD on 10k×768 takes 8 seconds in pure Python.

`ragdrift` is the library you reach for when you want the math right, the
runtime tight, and a single dependency that handles all five dimensions.

## Architecture

```
                       +----------------------+
                       |  RagDriftMonitor     |
                       |  (Python facade)     |
                       +----------+-----------+
                                  |
            +---------+-----------+-----------+--------+
            |         |           |           |        |
     EmbeddingDrift  DataDrift  ResponseDrift  ConfidenceDrift  QueryDrift
            |         |           |           |        |
            +---------+-----------+-----------+--------+
                                  |
                           ragdrift._native
                          (PyO3, GIL-released)
                                  |
                          ragdrift-core (Rust)
                          KS  PSI  MMD  SW  k-means
```

## Status

**0.1.0 — alpha.** API may change in 0.x. Semver respected within 0.x.y; minor
versions are allowed to break the surface. Core math is well-tested (45+ Rust
tests, 25+ Python tests); production deployments should pin an exact version.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All quality gates must pass on Linux,
macOS, and Windows for Python 3.10–3.13 before merge.

## License

Dual-licensed under MIT or Apache-2.0 (Rust convention). Pick whichever suits.

[LICENSE-MIT](LICENSE-MIT) · [LICENSE-APACHE](LICENSE-APACHE)
