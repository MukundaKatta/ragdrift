//! Tabular data drift via per-feature KS + PSI, reduced to a worst-feature score.

use crate::stats::{ks_two_sample, psi};
use crate::types::{DriftDimension, DriftScore};
use crate::Result;
use ndarray::{ArrayView2, Axis};

/// Detects feature-wise distribution drift on tabular data.
///
/// For each column, computes the KS statistic D and PSI, then takes the
/// maximum across features. KS dominates when the shape of the distribution
/// changes; PSI dominates when bin masses redistribute.
pub struct DataDriftDetector {
    threshold: f64,
    n_bins: usize,
    eps: f64,
}

impl DataDriftDetector {
    /// Create a detector with the given threshold, 10 PSI bins, and epsilon = 1e-4.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_bins: 10,
            eps: 1e-4,
        }
    }

    /// Override the number of PSI bins.
    pub fn with_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Run the detector. Inputs are `(n_samples, n_features)` and must share
    /// `n_features`.
    pub fn detect(
        &self,
        baseline: ArrayView2<f64>,
        current: ArrayView2<f64>,
    ) -> Result<DriftScore> {
        if baseline.ncols() != current.ncols() {
            return Err(crate::error::RagDriftError::DimensionMismatch {
                expected: baseline.ncols(),
                actual: current.ncols(),
                context: "DataDriftDetector::detect",
            });
        }
        let mut worst = 0.0_f64;
        for ((base_col, curr_col), _idx) in baseline
            .axis_iter(Axis(1))
            .zip(current.axis_iter(Axis(1)))
            .zip(0..)
        {
            let b: Vec<f64> = base_col.iter().copied().collect();
            let c: Vec<f64> = curr_col.iter().copied().collect();
            let ks = ks_two_sample(&b, &c)?.d;
            let p = if b.len() >= self.n_bins {
                psi(&b, &c, self.n_bins, self.eps).unwrap_or(0.0)
            } else {
                0.0
            };
            // Map PSI (unbounded but typically <=1) and D ([0,1]) onto a comparable scale.
            // We use the raw maximum: PSI > D in the regime that matters for alerts.
            let combined = ks.max(p);
            if combined > worst {
                worst = combined;
            }
        }
        Ok(DriftScore::new(
            DriftDimension::Data,
            worst,
            self.threshold,
            "ks+psi",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn identical_features_score_low() {
        let mut x: Array2<f64> = Array2::zeros((200, 3));
        for i in 0..200 {
            for j in 0..3 {
                x[[i, j]] = (i as f64) + (j as f64);
            }
        }
        let det = DataDriftDetector::new(0.1);
        let s = det.detect(x.view(), x.view()).unwrap();
        assert!(s.score < 1e-3, "score was {}", s.score);
        assert!(!s.exceeded);
    }

    #[test]
    fn shifted_feature_flagged() {
        let mut x: Array2<f64> = Array2::zeros((200, 3));
        let mut y: Array2<f64> = Array2::zeros((200, 3));
        for i in 0..200 {
            x[[i, 0]] = i as f64;
            x[[i, 1]] = i as f64;
            x[[i, 2]] = i as f64;
            y[[i, 0]] = i as f64;
            y[[i, 1]] = i as f64;
            // shift only column 2
            y[[i, 2]] = i as f64 + 100.0;
        }
        let det = DataDriftDetector::new(0.1);
        let s = det.detect(x.view(), y.view()).unwrap();
        assert!(s.exceeded, "score was {}", s.score);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let x: Array2<f64> = Array2::zeros((10, 3));
        let y: Array2<f64> = Array2::zeros((10, 4));
        let det = DataDriftDetector::new(0.1);
        assert!(det.detect(x.view(), y.view()).is_err());
    }
}
