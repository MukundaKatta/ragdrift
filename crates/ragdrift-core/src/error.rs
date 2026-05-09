//! Error type returned by all fallible ragdrift operations.

use thiserror::Error;

/// All errors raised by the core crate.
#[derive(Debug, Error)]
pub enum RagDriftError {
    /// Two arrays were expected to share a dimension and did not.
    #[error("dimension mismatch: expected {expected}, got {actual} ({context})")]
    DimensionMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
        /// Where the mismatch was detected.
        context: &'static str,
    },

    /// Not enough samples to run the detector.
    #[error("insufficient samples: need at least {required}, got {got} ({context})")]
    InsufficientSamples {
        /// Minimum required.
        required: usize,
        /// Actual count.
        got: usize,
        /// Where the check happened.
        context: &'static str,
    },

    /// A computation produced a non-finite value where it should have been finite.
    #[error("numerical instability in {0} (NaN or infinity)")]
    NumericalInstability(&'static str),

    /// Invalid configuration value (e.g., negative threshold).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// IO error while reading or writing a snapshot.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Serde error while serializing or deserializing a snapshot.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
