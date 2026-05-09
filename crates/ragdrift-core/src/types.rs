//! Public data types: dimensions, scores, reports, baseline snapshots.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// The five drift dimensions ragdrift monitors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriftDimension {
    /// Tabular feature distribution shift.
    Data,
    /// Embedding distribution shift (MMD + sliced Wasserstein).
    Embedding,
    /// Response distribution shift (length, entropy, optional semantic).
    Response,
    /// Confidence score distribution + calibration shift.
    Confidence,
    /// Query intent cluster reassignment shift.
    Query,
}

impl DriftDimension {
    /// Stable string label, useful for logs and metric names.
    pub fn as_str(self) -> &'static str {
        match self {
            DriftDimension::Data => "data",
            DriftDimension::Embedding => "embedding",
            DriftDimension::Response => "response",
            DriftDimension::Confidence => "confidence",
            DriftDimension::Query => "query",
        }
    }
}

/// A single per-dimension drift measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftScore {
    /// Which dimension was measured.
    pub dimension: DriftDimension,
    /// The measured drift score. Always non-negative.
    pub score: f64,
    /// The threshold above which the detector flags drift.
    pub threshold: f64,
    /// Whether `score >= threshold`.
    pub exceeded: bool,
    /// Human-readable name of the underlying method (e.g. `"mmd+sw"`).
    pub method: String,
}

impl DriftScore {
    /// Build a `DriftScore`, computing `exceeded` from `score >= threshold`.
    pub fn new(
        dimension: DriftDimension,
        score: f64,
        threshold: f64,
        method: impl Into<String>,
    ) -> Self {
        Self {
            dimension,
            score,
            threshold,
            exceeded: score >= threshold,
            method: method.into(),
        }
    }
}

/// A unified report aggregating per-dimension scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// One score per dimension that was monitored. Order is detector-defined.
    pub scores: Vec<DriftScore>,
    /// Unix timestamp (seconds) at which the report was created.
    pub timestamp: i64,
    /// Number of baseline samples that fed the detectors.
    pub sample_size_baseline: usize,
    /// Number of current samples that fed the detectors.
    pub sample_size_current: usize,
}

impl DriftReport {
    /// Construct a report stamped with the current wall-clock time.
    pub fn new(
        scores: Vec<DriftScore>,
        sample_size_baseline: usize,
        sample_size_current: usize,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            scores,
            timestamp,
            sample_size_baseline,
            sample_size_current,
        }
    }

    /// True if any dimension exceeded its threshold.
    pub fn any_exceeded(&self) -> bool {
        self.scores.iter().any(|s| s.exceeded)
    }

    /// Highest score across dimensions, or `0.0` if there are no scores.
    pub fn max_score(&self) -> f64 {
        self.scores.iter().map(|s| s.score).fold(0.0_f64, f64::max)
    }
}

/// A captured baseline used as the reference distribution for future detection.
///
/// Snapshots are designed to round-trip through `serde_json` so callers can
/// store them alongside model artifacts (S3, local disk, OCI registry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    /// Embedding matrix `(n_samples, dim)`. `None` means no embedding baseline.
    pub embeddings: Option<Array2<f32>>,
    /// Per-feature means (length = n_features) for tabular data drift.
    pub feature_means: Option<Vec<f64>>,
    /// Per-feature standard deviations (length = n_features).
    pub feature_stds: Option<Vec<f64>>,
    /// Confidence scores (length = n_responses) used as the calibration baseline.
    pub confidence_scores: Option<Vec<f64>>,
    /// Response lengths (length = n_responses) in characters or tokens.
    pub response_lengths: Option<Vec<usize>>,
    /// k-means centroids (k, dim) over baseline query embeddings.
    pub query_centroids: Option<Array2<f32>>,
    /// Cluster assignment frequency from baseline queries (length = k).
    pub query_cluster_freqs: Option<Vec<f64>>,
    /// Unix timestamp (seconds) at which the snapshot was captured.
    pub captured_at: i64,
}

impl BaselineSnapshot {
    /// An empty snapshot stamped with the current time.
    pub fn empty() -> Self {
        let captured_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            embeddings: None,
            feature_means: None,
            feature_stds: None,
            confidence_scores: None,
            response_lengths: None,
            query_centroids: None,
            query_cluster_freqs: None,
            captured_at,
        }
    }
}

impl Default for BaselineSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_score_exceeded_flag() {
        let s = DriftScore::new(DriftDimension::Data, 0.3, 0.2, "psi");
        assert!(s.exceeded);
        let s = DriftScore::new(DriftDimension::Data, 0.1, 0.2, "psi");
        assert!(!s.exceeded);
    }

    #[test]
    fn report_aggregations() {
        let scores = vec![
            DriftScore::new(DriftDimension::Data, 0.05, 0.1, "psi"),
            DriftScore::new(DriftDimension::Embedding, 0.3, 0.1, "mmd"),
        ];
        let r = DriftReport::new(scores, 100, 100);
        assert!(r.any_exceeded());
        assert_eq!(r.max_score(), 0.3);
    }

    #[test]
    fn snapshot_roundtrips_via_json() {
        let snap = BaselineSnapshot {
            feature_means: Some(vec![1.0, 2.0, 3.0]),
            feature_stds: Some(vec![0.1, 0.2, 0.3]),
            ..BaselineSnapshot::empty()
        };
        let s = serde_json::to_string(&snap).unwrap();
        let back: BaselineSnapshot = serde_json::from_str(&s).unwrap();
        assert_eq!(back.feature_means, snap.feature_means);
    }
}
