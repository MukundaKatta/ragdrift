//! ragdrift — five-dimensional drift detection for production RAG systems.
//!
//! This crate is a thin re-export of [`ragdrift_core`]. It exists so that
//! Rust users can write `cargo add ragdrift` and get the canonical entry
//! point without remembering the `-core` suffix. There is no behavior or
//! API here that isn't already in `ragdrift-core`; pin whichever crate you
//! prefer and the version will track.
//!
//! For the Python wheel, see <https://pypi.org/project/ragdrift/>.
//!
//! # Quick example
//!
//! ```
//! use ndarray::Array2;
//! use ragdrift::detectors::EmbeddingDriftDetector;
//!
//! let baseline = Array2::<f32>::zeros((50, 16));
//! let mut current = Array2::<f32>::zeros((50, 16));
//! current += 1.0;
//!
//! let detector = EmbeddingDriftDetector::new(0.1);
//! let score = detector.detect(baseline.view(), current.view()).unwrap();
//! assert!(score.score > 0.0);
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub use ragdrift_core::*;
