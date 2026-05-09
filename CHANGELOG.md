# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-09

### Changed
- **PyPI distribution name is `ragdrift-py`**, not `ragdrift`. The bare name
  was already taken by an unrelated project. The Python *import* stays
  `import ragdrift`, same convention as `pyyaml`/`yaml`. The Rust crates on
  crates.io (`ragdrift` and `ragdrift-core`) are unaffected.

### Fixed
- Crate metadata `rust-version` corrected from `1.75` to `1.80`. The `parallel`
  default feature pulls `rayon-core 1.13.0` which itself requires rustc 1.80,
  so 1.75 was never actually buildable. No public-API or behavior changes.

## [0.1.0] - 2026-05-09

### Added
- Five-dimensional drift detection: data, embedding, response, confidence, query.
- Pure-Rust core crate `ragdrift-core` with KS, PSI, MMD (RBF kernel), 1D and sliced Wasserstein.
- PyO3 bindings (`ragdrift._native`) built with `abi3-py310`, one wheel covers Python 3.10–3.13.
- Adapters: OpenSearch, pgvector, Pinecone (each gated behind an optional extra).
- Exporters: CloudWatch, Prometheus, Datadog (each gated behind an optional extra).
- High-level `RagDriftMonitor` that composes all five detectors into a unified `DriftReport`.
- Quickstart and end-to-end OpenSearch->CloudWatch examples.
- mkdocs site with concepts, quickstart, AWS Bedrock guide, and API reference.

### Notes
- Rust MSRV is **1.80** (not 1.75 as originally claimed), bumped to match what
  `rayon-core 1.13.0` actually requires.
- `ragdrift-core` published to crates.io. The Python wheel is not yet on PyPI
  pending Trusted Publisher setup; install from source via `maturin develop`
  in the meantime.

[Unreleased]: https://github.com/MukundaKatta/ragdrift/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MukundaKatta/ragdrift/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MukundaKatta/ragdrift/releases/tag/v0.1.0
