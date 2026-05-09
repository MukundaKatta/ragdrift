//! Maximum Mean Discrepancy with the RBF (Gaussian) kernel.
//!
//! MMD² is computed in the unbiased form by default (Gretton et al., 2012,
//! "A Kernel Two-Sample Test", JMLR). The RBF bandwidth is selected by the
//! median heuristic on the pooled sample.

use crate::error::RagDriftError;
use crate::Result;
use ndarray::{ArrayView2, Axis};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Which MMD² estimator to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmdEstimator {
    /// Biased (V-statistic) — always non-negative, slightly inflated for small samples.
    Biased,
    /// Unbiased (U-statistic) — can be slightly negative under H0; preferred for testing.
    Unbiased,
}

/// Compute MMD² between two sample sets `x` and `y` using an RBF kernel.
///
/// `x` and `y` are `(n_samples, dim)` views; both must share the same `dim`.
/// If `bandwidth` is `None`, the median heuristic on the pooled distances is used.
///
/// # Errors
///
/// Returns `DimensionMismatch` if the second axes differ, `InsufficientSamples`
/// if either input has fewer than 2 rows.
pub fn mmd_rbf(
    x: ArrayView2<f32>,
    y: ArrayView2<f32>,
    bandwidth: Option<f64>,
    estimator: MmdEstimator,
) -> Result<f64> {
    let (m, dx) = (x.nrows(), x.ncols());
    let (n, dy) = (y.nrows(), y.ncols());
    if dx != dy {
        return Err(RagDriftError::DimensionMismatch {
            expected: dx,
            actual: dy,
            context: "mmd_rbf",
        });
    }
    if m < 2 || n < 2 {
        return Err(RagDriftError::InsufficientSamples {
            required: 2,
            got: m.min(n),
            context: "mmd_rbf",
        });
    }

    let sigma2 = match bandwidth {
        Some(b) => {
            if b <= 0.0 {
                return Err(RagDriftError::InvalidConfig(
                    "mmd_rbf: bandwidth must be > 0".into(),
                ));
            }
            b * b
        }
        None => median_heuristic_sigma2(x, y)?,
    };
    if !sigma2.is_finite() || sigma2 == 0.0 {
        // All points identical: MMD is 0 by definition.
        return Ok(0.0);
    }

    let kxx = sum_kernel_offdiag(x, x, sigma2);
    let kyy = sum_kernel_offdiag(y, y, sigma2);
    let kxy = sum_kernel_full(x, y, sigma2);

    let m_f = m as f64;
    let n_f = n as f64;
    let mmd2 = match estimator {
        MmdEstimator::Unbiased => {
            kxx / (m_f * (m_f - 1.0)) + kyy / (n_f * (n_f - 1.0)) - 2.0 * kxy / (m_f * n_f)
        }
        MmdEstimator::Biased => {
            let kxx_full = kxx + m_f; // diagonal of self-kernel is 1 each
            let kyy_full = kyy + n_f;
            kxx_full / (m_f * m_f) + kyy_full / (n_f * n_f) - 2.0 * kxy / (m_f * n_f)
        }
    };
    if !mmd2.is_finite() {
        return Err(RagDriftError::NumericalInstability("mmd_rbf"));
    }
    Ok(mmd2)
}

#[inline]
fn sq_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum()
}

fn sum_kernel_full(x: ArrayView2<f32>, y: ArrayView2<f32>, sigma2: f64) -> f64 {
    let inv = -1.0 / (2.0 * sigma2);
    let x_rows: Vec<&[f32]> = x
        .axis_iter(Axis(0))
        .map(|r| r.to_slice().unwrap())
        .collect();
    let y_rows: Vec<&[f32]> = y
        .axis_iter(Axis(0))
        .map(|r| r.to_slice().unwrap())
        .collect();

    #[cfg(feature = "parallel")]
    {
        x_rows
            .par_iter()
            .map(|xi| {
                y_rows
                    .iter()
                    .map(|yj| (sq_distance(xi, yj) * inv).exp())
                    .sum::<f64>()
            })
            .sum::<f64>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        x_rows
            .iter()
            .map(|xi| {
                y_rows
                    .iter()
                    .map(|yj| (sq_distance(xi, yj) * inv).exp())
                    .sum::<f64>()
            })
            .sum::<f64>()
    }
}

fn sum_kernel_offdiag(a: ArrayView2<f32>, b: ArrayView2<f32>, sigma2: f64) -> f64 {
    debug_assert_eq!(a.nrows(), b.nrows());
    let n = a.nrows();
    let inv = -1.0 / (2.0 * sigma2);
    let rows: Vec<&[f32]> = a
        .axis_iter(Axis(0))
        .map(|r| r.to_slice().unwrap())
        .collect();

    #[cfg(feature = "parallel")]
    {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = 0.0_f64;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    acc += (sq_distance(rows[i], rows[j]) * inv).exp();
                }
                acc
            })
            .sum::<f64>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                acc += (sq_distance(rows[i], rows[j]) * inv).exp();
            }
        }
        acc
    }
}

fn median_heuristic_sigma2(x: ArrayView2<f32>, y: ArrayView2<f32>) -> Result<f64> {
    // Sample up to 256 pairs from the pooled set to keep this O(p*p) bounded.
    let cap = 256;
    let m = x.nrows().min(cap);
    let n = y.nrows().min(cap);
    let mut dists = Vec::with_capacity(m * (m - 1) / 2 + n * (n - 1) / 2 + m * n);
    let xs: Vec<&[f32]> = x
        .axis_iter(Axis(0))
        .take(m)
        .map(|r| r.to_slice().unwrap())
        .collect();
    let ys: Vec<&[f32]> = y
        .axis_iter(Axis(0))
        .take(n)
        .map(|r| r.to_slice().unwrap())
        .collect();
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            dists.push(sq_distance(xs[i], xs[j]).sqrt());
        }
        for y_row in &ys {
            dists.push(sq_distance(xs[i], y_row).sqrt());
        }
    }
    for i in 0..ys.len() {
        for j in (i + 1)..ys.len() {
            dists.push(sq_distance(ys[i], ys[j]).sqrt());
        }
    }
    if dists.is_empty() {
        return Err(RagDriftError::InsufficientSamples {
            required: 2,
            got: 0,
            context: "median_heuristic",
        });
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = dists[dists.len() / 2];
    // sigma chosen so 2*sigma^2 ~ med^2; standard convention.
    Ok((med * med).max(f64::EPSILON))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    #[test]
    fn identical_samples_unbiased_near_zero() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let x: Array2<f32> =
            Array2::random_using((50, 8), Normal::new(0.0, 1.0).unwrap(), &mut rng);
        let mmd2 = mmd_rbf(x.view(), x.view(), None, MmdEstimator::Unbiased).unwrap();
        // Same set: kxx == kyy == kxy and m==n, so unbiased reduces to:
        //   2 * kxx/(m(m-1)) - 2 * kxx/m^2 = 2*kxx*(m - (m-1)) / (m^2*(m-1)) > 0 but tiny.
        // We just want it bounded.
        assert!(mmd2.abs() < 0.05, "mmd2 was {}", mmd2);
    }

    #[test]
    fn shifted_samples_have_positive_mmd() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let x: Array2<f32> =
            Array2::random_using((100, 4), Normal::new(0.0, 1.0).unwrap(), &mut rng);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(2);
        let mut y: Array2<f32> =
            Array2::random_using((100, 4), Normal::new(0.0, 1.0).unwrap(), &mut rng2);
        y += 2.0; // shift mean by 2
        let mmd2 = mmd_rbf(x.view(), y.view(), None, MmdEstimator::Unbiased).unwrap();
        assert!(mmd2 > 0.05, "mmd2 was {}", mmd2);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let x: Array2<f32> = Array2::zeros((10, 4));
        let y: Array2<f32> = Array2::zeros((10, 5));
        assert!(mmd_rbf(x.view(), y.view(), None, MmdEstimator::Unbiased).is_err());
    }
}
