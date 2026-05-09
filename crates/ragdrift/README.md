# ragdrift

Five-dimensional drift detection for production RAG systems. Detects data,
embedding, response, confidence, and query-pattern drift.

This crate is a single-name re-export of
[`ragdrift-core`](https://crates.io/crates/ragdrift-core). They share a version.
Use whichever you prefer.

For the Python package, see <https://pypi.org/project/ragdrift/>.

```toml
[dependencies]
ragdrift = "0.1"
ndarray = "0.16"
```

```rust
use ndarray::Array2;
use ragdrift::detectors::EmbeddingDriftDetector;

let baseline = Array2::<f32>::zeros((100, 64));
let mut current = Array2::<f32>::zeros((100, 64));
current += 0.5;

let detector = EmbeddingDriftDetector::new(0.1);
let score = detector.detect(baseline.view(), current.view()).unwrap();
assert!(score.exceeded);
```

License: MIT OR Apache-2.0.
