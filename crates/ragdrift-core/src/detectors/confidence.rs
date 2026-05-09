//! Confidence score drift: KS on score distributions plus optional ECE delta.

use crate::error::RagDriftError;
use crate::stats::ks_two_sample;
use crate::types::{DriftDimension, DriftScore};
use crate::Result;

/// Detects drift in the distribution of model confidence scores. When ground-truth
/// correctness is available, also tracks the change in Expected Calibration Error
/// (ECE; Naeini et al., 2015).
pub struct ConfidenceDriftDetector {
    threshold: f64,
    n_calib_bins: usize,
}

impl ConfidenceDriftDetector {
    /// Create a detector. Default 10 ECE bins.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            n_calib_bins: 10,
        }
    }

    /// Override the number of bins used for ECE.
    pub fn with_calib_bins(mut self, n: usize) -> Self {
        self.n_calib_bins = n;
        self
    }

    /// KS-only mode: compare confidence score distributions.
    pub fn detect(
        &self,
        baseline_confidences: &[f64],
        current_confidences: &[f64],
    ) -> Result<DriftScore> {
        let d = ks_two_sample(baseline_confidences, current_confidences)?.d;
        Ok(DriftScore::new(
            DriftDimension::Confidence,
            d,
            self.threshold,
            "ks(conf)",
        ))
    }

    /// Full mode: also compute |ECE_current - ECE_baseline|.
    ///
    /// `correct_*` arrays must align with `confidence_*` and contain `true` when
    /// the response was correct.
    pub fn detect_with_correctness(
        &self,
        baseline_confidences: &[f64],
        baseline_correct: &[bool],
        current_confidences: &[f64],
        current_correct: &[bool],
    ) -> Result<DriftScore> {
        if baseline_confidences.len() != baseline_correct.len() {
            return Err(RagDriftError::DimensionMismatch {
                expected: baseline_confidences.len(),
                actual: baseline_correct.len(),
                context: "ConfidenceDriftDetector::baseline correctness",
            });
        }
        if current_confidences.len() != current_correct.len() {
            return Err(RagDriftError::DimensionMismatch {
                expected: current_confidences.len(),
                actual: current_correct.len(),
                context: "ConfidenceDriftDetector::current correctness",
            });
        }
        let d = ks_two_sample(baseline_confidences, current_confidences)?.d;
        let ece_base =
            expected_calibration_error(baseline_confidences, baseline_correct, self.n_calib_bins)?;
        let ece_curr =
            expected_calibration_error(current_confidences, current_correct, self.n_calib_bins)?;
        let ece_delta = (ece_curr - ece_base).abs();
        Ok(DriftScore::new(
            DriftDimension::Confidence,
            d + ece_delta,
            self.threshold,
            "ks(conf)+ece_delta",
        ))
    }
}

/// Expected Calibration Error: weighted average gap between bin confidence
/// and bin accuracy (Naeini et al., 2015). Confidence values must be in `[0, 1]`.
fn expected_calibration_error(conf: &[f64], correct: &[bool], n_bins: usize) -> Result<f64> {
    if conf.is_empty() {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: 0,
            context: "expected_calibration_error",
        });
    }
    if n_bins == 0 {
        return Err(RagDriftError::InvalidConfig(
            "ece: n_bins must be > 0".into(),
        ));
    }
    let n = conf.len() as f64;
    let mut sums = vec![0.0_f64; n_bins];
    let mut hits = vec![0_u32; n_bins];
    let mut counts = vec![0_u32; n_bins];
    for (&c, &ok) in conf.iter().zip(correct.iter()) {
        let c = c.clamp(0.0, 1.0);
        let bin = ((c * n_bins as f64) as usize).min(n_bins - 1);
        sums[bin] += c;
        if ok {
            hits[bin] += 1;
        }
        counts[bin] += 1;
    }
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        if counts[b] == 0 {
            continue;
        }
        let avg_conf = sums[b] / counts[b] as f64;
        let acc = hits[b] as f64 / counts[b] as f64;
        let weight = counts[b] as f64 / n;
        ece += weight * (avg_conf - acc).abs();
    }
    Ok(ece)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_confidences_score_zero() {
        let xs: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let det = ConfidenceDriftDetector::new(0.1);
        let s = det.detect(&xs, &xs).unwrap();
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn confidence_collapse_flagged() {
        let base: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.005).collect();
        let curr: Vec<f64> = (0..100).map(|_| 0.95).collect();
        let det = ConfidenceDriftDetector::new(0.3);
        let s = det.detect(&base, &curr).unwrap();
        assert!(s.exceeded, "score was {}", s.score);
    }

    #[test]
    fn ece_delta_picked_up() {
        // baseline well-calibrated: high conf -> usually correct
        let conf_base: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let correct_base: Vec<bool> = conf_base
            .iter()
            .enumerate()
            .map(|(i, _)| i >= 50) // bottom half wrong, top half right (matches mid-range conf)
            .collect();
        // current: same conf distribution but fully wrong (calibration broken)
        let conf_curr = conf_base.clone();
        let correct_curr: Vec<bool> = vec![false; 100];

        let det = ConfidenceDriftDetector::new(0.1);
        let s = det
            .detect_with_correctness(&conf_base, &correct_base, &conf_curr, &correct_curr)
            .unwrap();
        // KS = 0 (same conf dist) but ECE delta should be substantial.
        assert!(s.exceeded, "score was {}", s.score);
        assert!(s.method.contains("ece_delta"));
    }
}
