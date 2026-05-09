# API reference

This page is hand-curated; the canonical type signatures live in
[`python/ragdrift/_native.pyi`](https://github.com/MukundaKatta/ragdrift/blob/main/python/ragdrift/_native.pyi).

## Detectors

### `EmbeddingDrift(threshold=0.05, n_projections=64, seed=0)`

MMD² with the RBF kernel plus sliced Wasserstein-1. Method label: `"mmd+sw"`.

- `detect(baseline, current) -> DriftScore`
  - `baseline`, `current` — `numpy.ndarray[float32]` of shape `(n_samples, dim)`.

### `DataDrift(threshold=0.10, n_bins=10)`

Per-feature `max(KS, PSI)`. Method label: `"ks+psi"`.

- `detect(baseline, current) -> DriftScore`
  - `baseline`, `current` — `numpy.ndarray[float64]` of shape `(n_samples, n_features)`.

### `ResponseDrift(threshold=0.20, seed=0)`

- `detect(baseline_lengths, current_lengths) -> DriftScore` — KS on lengths.
- `detect_with_embeddings(baseline_lengths, current_lengths, baseline_embeddings, current_embeddings) -> DriftScore`
  — adds sliced Wasserstein-1 over response embeddings.

### `ConfidenceDrift(threshold=0.20, n_calib_bins=10)`

- `detect(baseline_confidences, current_confidences) -> DriftScore` — KS only.
- `detect_with_correctness(baseline_confidences, baseline_correct, current_confidences, current_correct) -> DriftScore`
  — KS plus `|ECE_current − ECE_baseline|`.

### `QueryDrift(threshold=0.10, k=8, seed=0)`

k-means baseline -> assign current -> symmetric KL. Method label: `"kmeans+sym_kl"`.

- `detect(baseline, current) -> DriftScore`
  - `baseline`, `current` — `numpy.ndarray[float32]` of shape `(n_queries, dim)`.

## High-level

### `RagDriftMonitor(...)` 

Composes any subset of the detectors. Call `evaluate(**kwargs)` with whatever
inputs you have; only the dimensions whose inputs are present run. Returns a
`DriftReport`.

## Types

### `DriftScore`

Frozen dataclass-like with `dimension: str`, `score: float`, `threshold: float`,
`exceeded: bool`, `method: str`, plus `to_dict()`.

### `DriftReport`

`scores: list[DriftScore]`, `timestamp: int`, `sample_size_baseline: int`,
`sample_size_current: int`. Methods: `any_exceeded()`, `max_score()`,
`to_json()`.

### `BaselineSnapshot`

JSON-serializable container for cached baseline statistics.
`to_json()` / `from_json(s)`.

### `RagDriftError`

Single exception class raised for invalid input or numerical instability.
Subclass of `Exception`.

## Adapters

- `ragdrift.adapters.OpenSearchAdapter`
- `ragdrift.adapters.PgVectorAdapter`
- `ragdrift.adapters.PineconeAdapter`

Each requires the matching extra; see the [Quickstart](../guides/quickstart.md).

## Exporters

- `ragdrift.exporters.CloudWatchExporter`
- `ragdrift.exporters.PrometheusExporter`
- `ragdrift.exporters.DatadogExporter`
