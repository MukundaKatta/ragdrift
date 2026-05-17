# ragdrift

**Five-dimensional drift detection for production RAG systems. Rust core, Python bindings.**

[![CI](https://github.com/MukundaKatta/ragdrift/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/ragdrift/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragdrift-py.svg)](https://pypi.org/project/ragdrift-py/)
[![crates.io](https://img.shields.io/crates/v/ragdrift-core.svg)](https://crates.io/crates/ragdrift-core)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mukundakatta.github.io/ragdrift/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## The problem

Your RAG system is in production. The retriever silently starts returning
slightly worse results because your corpus drifted. The model gets more
confident but no better calibrated. The mix of intents your users send
shifts week-over-week. None of this surfaces as a 5xx, none of it fails
your tests, none of it fires your latency alerts.

By the time someone notices it in a Slack thread, it has been broken for
weeks.

`ragdrift` watches **five dimensions at once** and gives you a single,
threshold-able report. The numerical core is Rust; the Python wheel is a
thin PyO3 binding that releases the GIL on the heavy paths.

## Install

```bash
pip install ragdrift-py
```

> The PyPI distribution is **`ragdrift-py`** because the bare `ragdrift`
> name on PyPI was taken by an unrelated project. The Python import is
> still `import ragdrift`, same convention as `pyyaml`/`yaml` or
> `opencv-python`/`cv2`. On crates.io, `cargo add ragdrift-core` works.

Optional extras:

```bash
pip install 'ragdrift-py[opensearch,aws]'        # adapter + CloudWatch exporter
pip install 'ragdrift-py[pgvector,prometheus]'
pip install 'ragdrift-py[pinecone,datadog]'
```

## Worked example

This snippet plants a known shift (mean +2σ on the current window) and
shows the monitor catching it. Run it as written:

```python
import numpy as np
from ragdrift import RagDriftMonitor

rng = np.random.default_rng(0)
baseline = rng.normal(size=(1000, 64)).astype(np.float32)
current  = rng.normal(loc=2.0, size=(1000, 64)).astype(np.float32)  # shifted

monitor = RagDriftMonitor(embedding_threshold=0.05)
report  = monitor.evaluate(
    baseline_embeddings=baseline,
    current_embeddings=current,
)

print(report.any_exceeded())     # True
for s in report.scores:
    print(s.dimension.value, s.score, s.exceeded, s.method)
# embedding 1.0324 True mmd+sw
```

The report is a plain dataclass; use `report.to_json()` for a log line,
`report.summary()` for a human-readable rollup. Wire it into a detector
loop, a cron, or a CloudWatch alarm, your choice.

## The five dimensions

| Dimension     | Method                          | What it catches |
|---------------|---------------------------------|-----------------|
| **Embedding** | MMD² (RBF) + sliced Wasserstein | corpus or model embedding distribution shift |
| **Data**      | per-feature KS + PSI            | tabular feature drift (latency, retrieval count, etc.) |
| **Response**  | KS on lengths, optional SW      | response length / semantic shift |
| **Confidence**| KS, optional ECE delta          | confidence score collapse, calibration breakage |
| **Query**     | k-means + symmetric KL          | intent-mix shift in incoming queries |

Skip any dimension by passing `<dim>_threshold=None`. See
[docs](https://mukundakatta.github.io/ragdrift/concepts/five-dimensions/)
for the math, the cites, and the constant choices.

## Why not X

- **Arize Phoenix** is great for embedding visualization and
  notebook-style exploration. It does not give you a single Rust-fast
  scalar you can alert on from Lambda.
- **Evidently** is excellent for tabular drift and report generation. It
  does not have a sliced-Wasserstein-on-embeddings primitive in the hot
  path.
- **WhyLabs / NannyML** are mature monitoring platforms — useful but
  vendor-tied, not embeddable as a library inside your service.
- **Roll your own** in numpy: most teams write the first 80% in 200
  lines and then hit a wall when MMD on 10k×768 takes 8 seconds in pure
  Python.

`ragdrift` is the library you reach for when you want the math right,
the runtime tight, and a single dependency that handles all five
dimensions.

## Performance

The Rust core uses `ndarray` + `rayon` on the hot paths. Indicative
numbers on a 2024 M-series mac:

| Operation                              | Pure numpy | ragdrift-core |
|----------------------------------------|------------|---------------|
| MMD (RBF), 10k × 768 vs 10k × 768      | ~8.0 s     | ~120 ms       |
| Sliced Wasserstein, 50 projections     | ~2.4 s     | ~35 ms        |
| Full report (all 5 dims, 10k samples)  | ~12 s      | ~180 ms       |

Numbers from `cargo bench` and `tests/perf_smoke.py`, not from a paper.

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

**0.1.x, alpha.** API may change in 0.x. Semver respected within 0.x.y;
minor versions are allowed to break the surface. Core math is
well-tested (45+ Rust tests, 25+ Python tests); production deployments
should pin an exact version.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quality gates must pass on
Linux, macOS, and Windows for Python 3.10–3.13 before merge.

## License

Dual-licensed under MIT or Apache-2.0 (Rust convention). Pick whichever
suits.

[LICENSE-MIT](LICENSE-MIT) · [LICENSE-APACHE](LICENSE-APACHE)
