//! Two-sample Kolmogorov–Smirnov test.

use crate::error::RagDriftError;
use crate::Result;

/// Result of a two-sample Kolmogorov–Smirnov test.
#[derive(Debug, Clone, Copy)]
pub struct KsResult {
    /// The KS statistic D — the supremum of |F1(x) - F2(x)|. Always in [0, 1].
    pub d: f64,
    /// Asymptotic two-sided p-value. Approximate; reliable for `n*m/(n+m) >= 4`.
    pub p_value: f64,
}

/// Two-sample two-sided Kolmogorov–Smirnov test.
///
/// Returns the KS statistic and an asymptotic p-value computed from the
/// Kolmogorov distribution survival function (Smirnov, 1948).
///
/// # Errors
///
/// Returns `InsufficientSamples` if either input is empty.
///
/// # Example
///
/// ```
/// use ragdrift_core::stats::ks_two_sample;
/// let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = a.clone();
/// let r = ks_two_sample(&a, &b).unwrap();
/// assert_eq!(r.d, 0.0);
/// assert!(r.p_value > 0.99);
/// ```
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> Result<KsResult> {
    if a.is_empty() || b.is_empty() {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: a.len().min(b.len()),
            context: "ks_two_sample",
        });
    }

    let mut a_sorted: Vec<f64> = a.to_vec();
    let mut b_sorted: Vec<f64> = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let n = a_sorted.len();
    let m = b_sorted.len();
    let n_inv = 1.0 / n as f64;
    let m_inv = 1.0 / m as f64;

    // Walk both sorted arrays in lockstep, tracking the empirical CDFs.
    let mut i = 0usize;
    let mut j = 0usize;
    let mut f1 = 0.0_f64;
    let mut f2 = 0.0_f64;
    let mut d = 0.0_f64;
    while i < n && j < m {
        let xa = a_sorted[i];
        let xb = b_sorted[j];
        if xa <= xb {
            i += 1;
            f1 = i as f64 * n_inv;
        }
        if xb <= xa {
            j += 1;
            f2 = j as f64 * m_inv;
        }
        let diff = (f1 - f2).abs();
        if diff > d {
            d = diff;
        }
    }

    let en = ((n * m) as f64 / (n + m) as f64).sqrt();
    let p_value = ks_asymptotic_pvalue((en + 0.12 + 0.11 / en) * d);
    Ok(KsResult { d, p_value })
}

/// Survival function of the Kolmogorov distribution evaluated at `lambda`.
///
/// Implementation of `Q(lambda) = 2 * sum_{j=1}^infty (-1)^(j-1) * exp(-2 j^2 lambda^2)`,
/// with the asymptotic formula from Press et al., *Numerical Recipes*, ch. 14.
fn ks_asymptotic_pvalue(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let eps1 = 1e-6;
    let eps2 = 1e-16;
    let a2 = -2.0 * lambda * lambda;
    let mut sum = 0.0_f64;
    let mut prev_term = 0.0_f64;
    let mut sign = 1.0_f64;
    for j in 1..101 {
        let term = sign * (a2 * (j as f64) * (j as f64)).exp();
        sum += term;
        if term.abs() <= eps1 * prev_term.abs() || term.abs() <= eps2 * sum.abs() {
            return (2.0 * sum).clamp(0.0, 1.0);
        }
        prev_term = term;
        sign = -sign;
    }
    1.0 // converged poorly; conservative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_d_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = ks_two_sample(&a, &a).unwrap();
        assert_eq!(r.d, 0.0);
        assert!(r.p_value > 0.99);
    }

    #[test]
    fn fully_disjoint_samples_have_d_one() {
        let a = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let b = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let r = ks_two_sample(&a, &b).unwrap();
        assert_eq!(r.d, 1.0);
        assert!(r.p_value < 0.01);
    }

    #[test]
    fn shifted_samples_have_intermediate_d() {
        // Half-overlap: D should be 0.5 exactly with these tied step boundaries.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let r = ks_two_sample(&a, &b).unwrap();
        assert!((r.d - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_input_errors() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0];
        assert!(ks_two_sample(&a, &b).is_err());
    }
}
