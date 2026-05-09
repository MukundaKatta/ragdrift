//! Tiny k-means implementation used by the query-pattern detector.
//!
//! Lloyd's algorithm with k-means++ initialization. Deterministic given a seed.
//! Hand-rolled to avoid pulling `linfa-clustering`'s BLAS-bound transitive deps.

use crate::error::RagDriftError;
use crate::Result;
use ndarray::{Array2, ArrayView2, Axis};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Result of a k-means run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Final centroids `(k, dim)`.
    pub centroids: Array2<f32>,
    /// Cluster assignment per input row.
    pub labels: Vec<usize>,
    /// Sum of squared distances of points to their assigned centroid.
    pub inertia: f64,
    /// Number of Lloyd iterations actually run.
    pub iters: usize,
}

/// Run k-means on `points` `(n, dim)` to produce `k` clusters.
///
/// `max_iters` caps Lloyd iterations (50 is plenty for d <= 1024). `tol` is the
/// minimum centroid movement (L2) below which the algorithm halts early.
///
/// # Errors
///
/// Returns `InvalidConfig` if `k == 0` or `k > n`.
pub fn kmeans(
    points: ArrayView2<f32>,
    k: usize,
    max_iters: usize,
    tol: f32,
    seed: u64,
) -> Result<KMeansResult> {
    let n = points.nrows();
    let dim = points.ncols();
    if k == 0 {
        return Err(RagDriftError::InvalidConfig("kmeans: k must be > 0".into()));
    }
    if k > n {
        return Err(RagDriftError::InvalidConfig(format!(
            "kmeans: k={} exceeds sample count n={}",
            k, n
        )));
    }
    if n == 0 {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: 0,
            context: "kmeans",
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeans_pp_init(points, k, &mut rng);
    let mut labels = vec![0_usize; n];
    let mut iters = 0usize;
    let mut inertia = 0.0_f64;

    for it in 0..max_iters {
        iters = it + 1;
        // Assign step.
        inertia = 0.0;
        for (i, row) in points.axis_iter(Axis(0)).enumerate() {
            let (best, dist) = nearest_centroid(row.as_slice().unwrap(), &centroids);
            labels[i] = best;
            inertia += dist as f64;
        }

        // Update step.
        let mut new_centroids = Array2::<f32>::zeros((k, dim));
        let mut counts = vec![0_u32; k];
        for (i, row) in points.axis_iter(Axis(0)).enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for (out, v) in new_centroids.row_mut(c).iter_mut().zip(row.iter()) {
                *out += *v;
            }
        }
        for (c, &count) in counts.iter().enumerate() {
            if count == 0 {
                // Empty cluster: re-seed to a random point to avoid collapse.
                let idx = rng.gen_range(0..n);
                new_centroids.row_mut(c).assign(&points.row(idx));
            } else {
                let inv = 1.0_f32 / count as f32;
                for v in new_centroids.row_mut(c).iter_mut() {
                    *v *= inv;
                }
            }
        }

        // Check convergence.
        let mut max_shift = 0.0_f32;
        for c in 0..k {
            let mut s = 0.0_f32;
            for (a, b) in centroids.row(c).iter().zip(new_centroids.row(c).iter()) {
                let d = *a - *b;
                s += d * d;
            }
            if s > max_shift {
                max_shift = s;
            }
        }
        centroids = new_centroids;
        if max_shift.sqrt() <= tol {
            break;
        }
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        iters,
    })
}

/// Assign each row of `points` to its nearest centroid. Returns a vector of
/// cluster indices.
pub fn assign(points: ArrayView2<f32>, centroids: ArrayView2<f32>) -> Vec<usize> {
    let centroids_owned: Array2<f32> = centroids.to_owned();
    points
        .axis_iter(Axis(0))
        .map(|row| nearest_centroid(row.as_slice().unwrap(), &centroids_owned).0)
        .collect()
}

fn nearest_centroid(point: &[f32], centroids: &Array2<f32>) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.axis_iter(Axis(0)).enumerate() {
        let mut d = 0.0_f32;
        for (a, b) in point.iter().zip(c.iter()) {
            let diff = *a - *b;
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    (best, best_d)
}

fn kmeans_pp_init(points: ArrayView2<f32>, k: usize, rng: &mut StdRng) -> Array2<f32> {
    let n = points.nrows();
    let dim = points.ncols();
    let mut centroids = Array2::<f32>::zeros((k, dim));
    let first = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&points.row(first));

    let mut min_dists = vec![f32::INFINITY; n];
    for ci in 1..k {
        // Update min distance to nearest existing centroid.
        for (i, row) in points.axis_iter(Axis(0)).enumerate() {
            let mut d = 0.0_f32;
            for (a, b) in row.iter().zip(centroids.row(ci - 1).iter()) {
                let diff = *a - *b;
                d += diff * diff;
            }
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        // Sample next centroid proportional to squared distance.
        let total: f32 = min_dists.iter().sum();
        let weights: Vec<f32> = if total > 0.0 {
            min_dists.iter().map(|&d| d.max(0.0)).collect()
        } else {
            // Pathological: all points coincident with chosen centroids.
            vec![1.0; n]
        };
        let dist = WeightedIndex::new(&weights).unwrap();
        let pick = dist.sample(rng);
        centroids.row_mut(ci).assign(&points.row(pick));
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn kmeans_separates_two_clusters() {
        // Two well-separated clouds in 2D.
        let mut points = Array2::<f32>::zeros((20, 2));
        for i in 0..10 {
            points[[i, 0]] = 0.0 + (i as f32) * 0.01;
            points[[i, 1]] = 0.0;
            points[[i + 10, 0]] = 10.0 + (i as f32) * 0.01;
            points[[i + 10, 1]] = 10.0;
        }
        let r = kmeans(points.view(), 2, 50, 1e-6, 42).unwrap();
        assert_eq!(r.labels.len(), 20);
        // First 10 should share a label; last 10 should share the other.
        let l0 = r.labels[0];
        let l1 = r.labels[10];
        assert_ne!(l0, l1);
        for i in 0..10 {
            assert_eq!(r.labels[i], l0);
            assert_eq!(r.labels[i + 10], l1);
        }
    }

    #[test]
    fn kmeans_rejects_k_larger_than_n() {
        let p = Array2::<f32>::zeros((3, 2));
        assert!(kmeans(p.view(), 5, 10, 1e-6, 0).is_err());
    }

    #[test]
    fn assign_matches_kmeans_labels() {
        let mut points = Array2::<f32>::zeros((20, 2));
        for i in 0..10 {
            points[[i, 0]] = 0.0;
            points[[i + 10, 0]] = 10.0;
        }
        let r = kmeans(points.view(), 2, 50, 1e-6, 0).unwrap();
        let labels2 = assign(points.view(), r.centroids.view());
        assert_eq!(labels2, r.labels);
    }
}
