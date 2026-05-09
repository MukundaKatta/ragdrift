//! ragdrift-core: five-dimensional drift detection for production RAG systems.
//!
//! This crate exposes detectors for **data**, **embedding**, **response**,
//! **confidence**, and **query-pattern** drift, plus the underlying statistical
//! primitives (KS, PSI, MMD, sliced Wasserstein). All numerics live here; the
//! Python `ragdrift` package is a thin PyO3 wrapper.
//!
//! # Example
//!
//! ```
//! use ndarray::Array2;
//! use ragdrift_core::detectors::EmbeddingDriftDetector;
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

pub mod detectors;
pub mod error;
pub mod stats;
pub mod types;

pub use error::RagDriftError;
pub use types::{BaselineSnapshot, DriftDimension, DriftReport, DriftScore};

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, RagDriftError>;
