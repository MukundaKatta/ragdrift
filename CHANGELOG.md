# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-08

### Added
- Five-dimensional drift detection: data, embedding, response, confidence, query.
- Pure-Rust core crate `ragdrift-core` with KS, PSI, MMD (RBF kernel), 1D and sliced Wasserstein.
- PyO3 bindings (`ragdrift._native`) built with `abi3-py310`, one wheel covers Python 3.10–3.13.
- Adapters: OpenSearch, pgvector, Pinecone (each gated behind an optional extra).
- Exporters: CloudWatch, Prometheus, Datadog (each gated behind an optional extra).
- High-level `RagDriftMonitor` that composes all five detectors into a unified `DriftReport`.
- Quickstart and end-to-end OpenSearch->CloudWatch examples.
- mkdocs site with concepts, quickstart, AWS Bedrock guide, and API reference.

[Unreleased]: https://github.com/MukundaKatta/ragdrift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MukundaKatta/ragdrift/releases/tag/v0.1.0
