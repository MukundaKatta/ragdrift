//! Statistical primitives used by the detectors. Each lives in its own
//! submodule and is unit-tested independently.

pub mod kmeans;
pub mod ks;
pub mod mmd;
pub mod psi;
pub mod wasserstein;

pub use kmeans::{kmeans, KMeansResult};
pub use ks::ks_two_sample;
pub use mmd::{mmd_rbf, MmdEstimator};
pub use psi::psi;
pub use wasserstein::{sliced_wasserstein, wasserstein_1d};
