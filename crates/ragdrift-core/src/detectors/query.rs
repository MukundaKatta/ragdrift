//! Query-pattern drift: cluster baseline queries, measure assignment shift via KL.

use crate::error::RagDriftError;
use crate::stats::kmeans::{assign, kmeans};
use crate::types::{DriftDimension, DriftScore};
use crate::Result;
use ndarray::ArrayView2;

/// Detects drift in the *intent* mix of incoming queries.
///
/// Clusters the baseline embeddings with k-means, then re-assigns the current
/// embeddings to those baseline centroids. The score is the symmetric
/// Kullback–Leibler divergence between the two assignment distributions.
pub struct QueryDriftDetector {
    threshold: f64,
    k: usize,
    max_iters: usize,
    tol: f32,
    seed: u64,
    smoothing: f64,
}

impl QueryDriftDetector {
    /// Create a detector with `k` clusters. Defaults: 50 Lloyd iters, tol 1e-4,
    /// 1e-6 add-epsilon for KL smoothing.
    pub fn new(threshold: f64, k: usize) -> Self {
        Self {
            threshold,
            k,
            max_iters: 50,
            tol: 1e-4,
            seed: 0,
            smoothing: 1e-6,
        }
    }

    /// Override the RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run the detector against baseline and current query embeddings.
    pub fn detect(
        &self,
        baseline: ArrayView2<f32>,
        current: ArrayView2<f32>,
    ) -> Result<DriftScore> {
        if baseline.ncols() != current.ncols() {
            return Err(RagDriftError::DimensionMismatch {
                expected: baseline.ncols(),
                actual: current.ncols(),
                context: "QueryDriftDetector::detect",
            });
        }
        let res = kmeans(baseline, self.k, self.max_iters, self.tol, self.seed)?;
        let baseline_freqs = freqs(&res.labels, self.k);

        let curr_labels = assign(current, res.centroids.view());
        let current_freqs = freqs(&curr_labels, self.k);

        let kl = symmetric_kl(&baseline_freqs, &current_freqs, self.smoothing);
        Ok(DriftScore::new(
            DriftDimension::Query,
            kl,
            self.threshold,
            "kmeans+sym_kl",
        ))
    }
}

fn freqs(labels: &[usize], k: usize) -> Vec<f64> {
    let mut c = vec![0_u64; k];
    for &l in labels {
        if l < k {
            c[l] += 1;
        }
    }
    let n = labels.len() as f64;
    if n == 0.0 {
        return vec![0.0; k];
    }
    c.into_iter().map(|x| x as f64 / n).collect()
}

fn symmetric_kl(p: &[f64], q: &[f64], eps: f64) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let mut kl_pq = 0.0_f64;
    let mut kl_qp = 0.0_f64;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let ps = pi + eps;
        let qs = qi + eps;
        kl_pq += ps * (ps / qs).ln();
        kl_qp += qs * (qs / ps).ln();
    }
    0.5 * (kl_pq + kl_qp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn same_query_mix_scores_low() {
        // two distinct clusters at (0,0) and (10,10), 20 points each
        let mut points = Array2::<f32>::zeros((40, 2));
        for i in 0..20 {
            points[[i, 0]] = (i as f32) * 0.01;
            points[[i, 1]] = (i as f32) * 0.01;
            points[[i + 20, 0]] = 10.0 + (i as f32) * 0.01;
            points[[i + 20, 1]] = 10.0 + (i as f32) * 0.01;
        }
        let det = QueryDriftDetector::new(0.1, 2).with_seed(7);
        let s = det.detect(points.view(), points.view()).unwrap();
        assert!(s.score < 1e-3, "score was {}", s.score);
    }

    #[test]
    fn shifted_query_mix_flagged() {
        // baseline is 50/50 between two clusters; current is 100% in one cluster
        let mut baseline = Array2::<f32>::zeros((40, 2));
        for i in 0..20 {
            baseline[[i, 0]] = (i as f32) * 0.01;
            baseline[[i + 20, 0]] = 10.0 + (i as f32) * 0.01;
            baseline[[i + 20, 1]] = 10.0;
        }
        let mut current = Array2::<f32>::zeros((40, 2));
        for i in 0..40 {
            current[[i, 0]] = (i as f32) * 0.01; // all in cluster A
        }
        let det = QueryDriftDetector::new(0.1, 2).with_seed(7);
        let s = det.detect(baseline.view(), current.view()).unwrap();
        assert!(s.exceeded, "score was {}", s.score);
    }
}
