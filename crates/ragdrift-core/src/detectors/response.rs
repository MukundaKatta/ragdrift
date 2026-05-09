//! Response distribution drift: length shift, optional semantic shift.

use crate::stats::{ks_two_sample, sliced_wasserstein};
use crate::types::{DriftDimension, DriftScore};
use crate::Result;
use ndarray::ArrayView2;

/// Detects drift in the distribution of model responses.
///
/// The default mode runs a KS test on response lengths (in characters or
/// tokens — your choice, just be consistent). With response embeddings the
/// detector also adds a sliced-Wasserstein semantic shift component.
pub struct ResponseDriftDetector {
    threshold: f64,
    n_projections: usize,
    seed: u64,
}

impl ResponseDriftDetector {
    /// Create a detector. Default 32 projections for the optional embedding pass.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_projections: 32,
            seed: 0,
        }
    }

    /// Override the seed used for sliced-Wasserstein projections.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// KS-only mode: compare two length distributions.
    pub fn detect(&self, baseline_lengths: &[f64], current_lengths: &[f64]) -> Result<DriftScore> {
        let d = ks_two_sample(baseline_lengths, current_lengths)?.d;
        Ok(DriftScore::new(
            DriftDimension::Response,
            d,
            self.threshold,
            "ks(length)",
        ))
    }

    /// Full mode: combine length-KS with sliced Wasserstein on response embeddings.
    pub fn detect_with_embeddings(
        &self,
        baseline_lengths: &[f64],
        current_lengths: &[f64],
        baseline_embeddings: ArrayView2<f32>,
        current_embeddings: ArrayView2<f32>,
    ) -> Result<DriftScore> {
        let d = ks_two_sample(baseline_lengths, current_lengths)?.d;
        let sw = sliced_wasserstein(
            baseline_embeddings,
            current_embeddings,
            self.n_projections,
            self.seed,
        )?;
        Ok(DriftScore::new(
            DriftDimension::Response,
            d + sw,
            self.threshold,
            "ks(length)+sw(emb)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn identical_lengths_score_zero() {
        let lens: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let det = ResponseDriftDetector::new(0.1);
        let s = det.detect(&lens, &lens).unwrap();
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn longer_responses_flagged() {
        let base: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let curr: Vec<f64> = (0..100).map(|i| (i + 200) as f64).collect();
        let det = ResponseDriftDetector::new(0.5);
        let s = det.detect(&base, &curr).unwrap();
        assert!(s.exceeded);
    }

    #[test]
    fn embedding_mode_combines_signals() {
        let base_len: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let curr_len: Vec<f64> = base_len.clone();
        let base_emb: Array2<f32> = Array2::zeros((100, 8));
        let mut curr_emb: Array2<f32> = Array2::zeros((100, 8));
        curr_emb += 3.0;
        let det = ResponseDriftDetector::new(0.5);
        let s = det
            .detect_with_embeddings(&base_len, &curr_len, base_emb.view(), curr_emb.view())
            .unwrap();
        // length signal is zero; semantic SW should still pop.
        assert!(s.score > 0.5, "score was {}", s.score);
        assert!(s.method.contains("sw(emb)"));
    }
}
