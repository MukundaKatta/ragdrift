# ragdrift-core

Pure Rust core of [`ragdrift`](https://github.com/MukundaKatta/ragdrift). Five-dimensional
drift detection for production RAG systems: data, embedding, response, confidence, query.

This crate is usable standalone from Rust. Python users want the
[`ragdrift`](https://pypi.org/project/ragdrift/) wheel, which wraps this crate via PyO3.

## Quick example

```rust
use ndarray::Array2;
use ragdrift_core::detectors::EmbeddingDriftDetector;

let baseline = Array2::<f32>::zeros((100, 64));
let mut current = Array2::<f32>::zeros((100, 64));
current += 0.5;

let detector = EmbeddingDriftDetector::new(0.1);
let score = detector.detect(baseline.view(), current.view()).unwrap();
assert!(score.exceeded);
```

License: MIT OR Apache-2.0.
