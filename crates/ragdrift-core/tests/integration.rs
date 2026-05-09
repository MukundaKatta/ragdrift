//! End-to-end synthetic-data tests: build a baseline, build a drifted current,
//! verify each detector flags drift above its threshold.

use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use ragdrift_core::detectors::{
    ConfidenceDriftDetector, DataDriftDetector, EmbeddingDriftDetector, QueryDriftDetector,
    ResponseDriftDetector,
};

fn rng(seed: u64) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[test]
fn embedding_drift_flagged_on_distribution_shift() {
    let mut r = rng(0);
    let baseline: Array2<f32> =
        Array2::random_using((200, 32), Normal::new(0.0, 1.0).unwrap(), &mut r);
    let mut r = rng(1);
    let mut current: Array2<f32> =
        Array2::random_using((200, 32), Normal::new(0.0, 1.0).unwrap(), &mut r);
    current += 1.5;

    let det = EmbeddingDriftDetector::new(0.05).with_seed(7);
    let s = det.detect(baseline.view(), current.view()).unwrap();
    assert!(s.exceeded, "expected drift; score = {}", s.score);
}

#[test]
fn data_drift_flagged_on_per_feature_shift() {
    let n = 500;
    let mut baseline = Array2::<f64>::zeros((n, 4));
    let mut current = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        for j in 0..4 {
            baseline[[i, j]] = (i as f64) * 0.01 + (j as f64) * 0.1;
            current[[i, j]] = (i as f64) * 0.01 + (j as f64) * 0.1;
        }
        // shift only column 3 by a clear margin
        current[[i, 3]] += 5.0;
    }
    let det = DataDriftDetector::new(0.1);
    let s = det.detect(baseline.view(), current.view()).unwrap();
    assert!(s.exceeded, "expected drift; score = {}", s.score);
}

#[test]
fn response_drift_flagged_on_length_shift() {
    let baseline: Vec<f64> = (0..500).map(|i| (i % 100) as f64).collect();
    let current: Vec<f64> = (0..500).map(|i| (i % 100) as f64 + 50.0).collect();
    let det = ResponseDriftDetector::new(0.3);
    let s = det.detect(&baseline, &current).unwrap();
    assert!(s.exceeded, "expected drift; score = {}", s.score);
}

#[test]
fn confidence_drift_flagged_on_distribution_collapse() {
    let baseline: Vec<f64> = (0..500).map(|i| (i as f64) / 500.0).collect();
    let current: Vec<f64> = (0..500).map(|_| 0.95).collect();
    let det = ConfidenceDriftDetector::new(0.5);
    let s = det.detect(&baseline, &current).unwrap();
    assert!(s.exceeded, "expected drift; score = {}", s.score);
}

#[test]
fn query_drift_flagged_when_intent_mix_collapses() {
    // Two clusters in 4D, baseline is 50/50, current is 100% in cluster A.
    let mut baseline = Array2::<f32>::zeros((200, 4));
    for i in 0..100 {
        baseline[[i, 0]] = 0.0 + (i as f32) * 0.001;
        baseline[[i + 100, 0]] = 10.0 + (i as f32) * 0.001;
    }
    let mut current = Array2::<f32>::zeros((200, 4));
    for i in 0..200 {
        current[[i, 0]] = (i as f32) * 0.001;
    }
    let det = QueryDriftDetector::new(0.1, 2).with_seed(7);
    let s = det.detect(baseline.view(), current.view()).unwrap();
    assert!(s.exceeded, "expected drift; score = {}", s.score);
}

#[test]
fn no_drift_under_h0() {
    // Same distribution sampled twice should not trip thresholds with reasonable margins.
    let mut r = rng(42);
    let a: Array2<f32> = Array2::random_using((300, 16), Normal::new(0.0, 1.0).unwrap(), &mut r);
    let mut r = rng(43);
    let b: Array2<f32> = Array2::random_using((300, 16), Normal::new(0.0, 1.0).unwrap(), &mut r);
    let det = EmbeddingDriftDetector::new(0.5).with_seed(7);
    let s = det.detect(a.view(), b.view()).unwrap();
    assert!(!s.exceeded, "false positive; score = {}", s.score);
}
