//! Embedding distribution drift via MMD² + sliced Wasserstein.

use crate::stats::{mmd_rbf, sliced_wasserstein, MmdEstimator};
use crate::types::{DriftDimension, DriftScore};
use crate::Result;
use ndarray::ArrayView2;

/// Detects drift in an embedding distribution.
///
/// Combines a sample-efficient kernel test (MMD² with the RBF kernel and the
/// median-bandwidth heuristic) with a bandwidth-free geometric measure (sliced
/// Wasserstein-1). The reported score is `mmd² + sw`, which is a non-negative
/// summary; consult the underlying methods if you need to disentangle them.
pub struct EmbeddingDriftDetector {
    threshold: f64,
    n_projections: usize,
    seed: u64,
}

impl EmbeddingDriftDetector {
    /// Create a detector with the given alert threshold. Defaults to 64 random
    /// projections for sliced Wasserstein.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_projections: 64,
            seed: 0,
        }
    }

    /// Override the number of random projections used by sliced Wasserstein.
    pub fn with_projections(mut self, n: usize) -> Self {
        self.n_projections = n;
        self
    }

    /// Override the RNG seed used for sliced-Wasserstein projections.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run the detector against a baseline and a current embedding matrix.
    pub fn detect(
        &self,
        baseline: ArrayView2<f32>,
        current: ArrayView2<f32>,
    ) -> Result<DriftScore> {
        let mmd2 = mmd_rbf(baseline, current, None, MmdEstimator::Unbiased)?;
        let sw = sliced_wasserstein(baseline, current, self.n_projections, self.seed)?;
        // MMD can be slightly negative under H0; clamp before combining.
        let score = mmd2.max(0.0) + sw;
        Ok(DriftScore::new(
            DriftDimension::Embedding,
            score,
            self.threshold,
            "mmd+sw",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn identical_embeddings_score_low() {
        let x: Array2<f32> = Array2::ones((50, 16));
        let det = EmbeddingDriftDetector::new(0.05);
        let s = det.detect(x.view(), x.view()).unwrap();
        assert!(!s.exceeded);
        assert!(s.score < 1e-3, "score was {}", s.score);
    }

    #[test]
    fn shifted_embeddings_score_high() {
        let mut x: Array2<f32> = Array2::zeros((100, 8));
        let mut y: Array2<f32> = Array2::zeros((100, 8));
        // give x and y disjoint means
        for i in 0..100 {
            for j in 0..8 {
                x[[i, j]] = (i + j) as f32 * 0.01;
                y[[i, j]] = (i + j) as f32 * 0.01 + 5.0;
            }
        }
        let det = EmbeddingDriftDetector::new(0.1);
        let s = det.detect(x.view(), y.view()).unwrap();
        assert!(s.exceeded, "score was {}", s.score);
    }
}
