//! 1D Wasserstein-1 (exact) and sliced Wasserstein-1 for high-dimensional embeddings.

use crate::error::RagDriftError;
use crate::Result;
use ndarray::{Array2, ArrayView2, Axis};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;

/// Exact 1D Wasserstein-1 distance between two empirical samples.
///
/// Equivalent to the L1 distance between the empirical CDFs (Vallender, 1973).
/// When `a.len() == b.len()` this reduces to the mean of pairwise sorted absolute
/// differences. The general case integrates `|F_a^{-1}(t) - F_b^{-1}(t)| dt`.
///
/// # Errors
///
/// Returns `InsufficientSamples` if either input is empty.
///
/// # Example
///
/// ```
/// use ragdrift_core::stats::wasserstein_1d;
/// // U[0,1] vs U[1,2]: W1 = 1.0 exactly.
/// let a: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
/// let b: Vec<f64> = a.iter().map(|x| x + 1.0).collect();
/// let w = wasserstein_1d(&a, &b).unwrap();
/// assert!((w - 1.0).abs() < 1e-3);
/// ```
pub fn wasserstein_1d(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.is_empty() || b.is_empty() {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: a.len().min(b.len()),
            context: "wasserstein_1d",
        });
    }
    let mut a_sorted: Vec<f64> = a.to_vec();
    let mut b_sorted: Vec<f64> = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    if a_sorted.len() == b_sorted.len() {
        let n = a_sorted.len() as f64;
        let s: f64 = a_sorted
            .iter()
            .zip(b_sorted.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        return Ok(s / n);
    }

    // General case: walk the union of step locations of the two empirical CDFs
    // and integrate |F_a - F_b| (which equals integral of |F^-1| differences).
    let n = a_sorted.len();
    let m = b_sorted.len();
    let n_f = n as f64;
    let m_f = m as f64;
    let mut all: Vec<f64> = Vec::with_capacity(n + m);
    all.extend_from_slice(&a_sorted);
    all.extend_from_slice(&b_sorted);
    all.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let mut i = 0usize;
    let mut j = 0usize;
    let mut prev = all[0];
    let mut total = 0.0_f64;
    for &x in all.iter().skip(1) {
        // CDF values at the *left* of the interval (prev, x].
        let fa = i as f64 / n_f;
        let fb = j as f64 / m_f;
        total += (fa - fb).abs() * (x - prev);
        prev = x;
        // Advance whichever array(s) had a step at x.
        while i < n && a_sorted[i] <= x {
            i += 1;
        }
        while j < m && b_sorted[j] <= x {
            j += 1;
        }
    }
    Ok(total)
}

/// Sliced Wasserstein-1 distance for d-dimensional embeddings (Bonneel et al., 2015).
///
/// Projects both sample sets onto `n_projections` random unit directions and
/// averages the 1D Wasserstein distance along each direction. Bandwidth-free
/// and embarrassingly parallel.
///
/// # Errors
///
/// Returns `DimensionMismatch` if column counts differ; `InsufficientSamples` for
/// empty inputs; `InvalidConfig` if `n_projections == 0`.
pub fn sliced_wasserstein(
    x: ArrayView2<f32>,
    y: ArrayView2<f32>,
    n_projections: usize,
    seed: u64,
) -> Result<f64> {
    if x.ncols() != y.ncols() {
        return Err(RagDriftError::DimensionMismatch {
            expected: x.ncols(),
            actual: y.ncols(),
            context: "sliced_wasserstein",
        });
    }
    if x.nrows() == 0 || y.nrows() == 0 {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: x.nrows().min(y.nrows()),
            context: "sliced_wasserstein",
        });
    }
    if n_projections == 0 {
        return Err(RagDriftError::InvalidConfig(
            "sliced_wasserstein: n_projections must be > 0".into(),
        ));
    }

    let dim = x.ncols();
    let projections = sample_unit_directions(dim, n_projections, seed);

    let mut total = 0.0_f64;
    for col in projections.axis_iter(Axis(0)) {
        let xp: Vec<f64> = project(&x, col.as_slice().unwrap());
        let yp: Vec<f64> = project(&y, col.as_slice().unwrap());
        total += wasserstein_1d(&xp, &yp)?;
    }
    Ok(total / n_projections as f64)
}

fn project(x: &ArrayView2<f32>, dir: &[f32]) -> Vec<f64> {
    x.axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .zip(dir.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum::<f64>()
        })
        .collect()
}

fn sample_unit_directions(dim: usize, n: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    let mut a = Array2::<f32>::zeros((n, dim));
    for mut row in a.axis_iter_mut(Axis(0)) {
        let mut sumsq = 0.0_f32;
        for v in row.iter_mut() {
            let s = normal.sample(&mut rng);
            *v = s;
            sumsq += s * s;
        }
        let norm = sumsq.sqrt().max(f32::EPSILON);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn identical_samples_have_w_zero() {
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let w = wasserstein_1d(&a, &a).unwrap();
        assert_eq!(w, 0.0);
    }

    #[test]
    fn unit_uniform_shift_has_w_one() {
        // U[0,1] approximated by 1000 evenly spaced points; shifted by 1.0.
        let a: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let b: Vec<f64> = a.iter().map(|x| x + 1.0).collect();
        let w = wasserstein_1d(&a, &b).unwrap();
        assert!((w - 1.0).abs() < 1e-3, "w was {}", w);
    }

    #[test]
    fn unequal_lengths_handled() {
        let a: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let b: Vec<f64> = (0..200).map(|i| i as f64 / 200.0 + 1.0).collect();
        let w = wasserstein_1d(&a, &b).unwrap();
        // Both ~ U[0,1] / U[1,2], W1 ~ 1.0.
        assert!((w - 1.0).abs() < 0.05, "w was {}", w);
    }

    #[test]
    fn sliced_w_identical_near_zero() {
        let x: Array2<f32> = Array2::ones((50, 8));
        let s = sliced_wasserstein(x.view(), x.view(), 32, 0).unwrap();
        assert!(s.abs() < 1e-6, "s was {}", s);
    }

    #[test]
    fn sliced_w_shift_positive() {
        let x: Array2<f32> = Array2::zeros((100, 8));
        let y: Array2<f32> = Array2::ones((100, 8)) * 2.0;
        let s = sliced_wasserstein(x.view(), y.view(), 64, 7).unwrap();
        assert!(s > 0.5, "s was {}", s);
    }
}
