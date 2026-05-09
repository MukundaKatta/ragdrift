//! Population Stability Index — the credit-risk industry's standard distribution-shift metric.
//!
//! Common interpretation thresholds (Karakoulas, 2004; widely adopted in credit risk):
//!
//! | PSI         | Interpretation                  |
//! |-------------|---------------------------------|
//! | `< 0.10`    | No significant shift            |
//! | `0.10–0.25` | Moderate shift, investigate     |
//! | `> 0.25`    | Significant shift, action needed |

use crate::error::RagDriftError;
use crate::Result;

/// Compute the Population Stability Index between `baseline` and `current`.
///
/// Bins are constructed from quantiles of `baseline` (so PSI is invariant to
/// baseline scale). Bin counts are smoothed by adding `eps` to every bin before
/// normalizing, which keeps PSI finite when a bin is empty in either sample.
///
/// # Errors
///
/// Returns `InsufficientSamples` if either input has fewer than `n_bins` elements,
/// or `InvalidConfig` if `n_bins < 2` or `eps <= 0`.
///
/// # Example
///
/// ```
/// use ragdrift_core::stats::psi;
/// let baseline: Vec<f64> = (0..1000).map(|i| i as f64).collect();
/// let current = baseline.clone();
/// let p = psi(&baseline, &current, 10, 1e-4).unwrap();
/// assert!(p < 1e-6);
/// ```
pub fn psi(baseline: &[f64], current: &[f64], n_bins: usize, eps: f64) -> Result<f64> {
    if n_bins < 2 {
        return Err(RagDriftError::InvalidConfig(
            "psi: n_bins must be >= 2".into(),
        ));
    }
    if eps <= 0.0 {
        return Err(RagDriftError::InvalidConfig("psi: eps must be > 0".into()));
    }
    if baseline.len() < n_bins {
        return Err(RagDriftError::InsufficientSamples {
            required: n_bins,
            got: baseline.len(),
            context: "psi(baseline)",
        });
    }
    if current.is_empty() {
        return Err(RagDriftError::InsufficientSamples {
            required: 1,
            got: 0,
            context: "psi(current)",
        });
    }

    // Quantile-based bin edges from the baseline.
    let mut sorted: Vec<f64> = baseline.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(n_bins - 1);
    for k in 1..n_bins {
        let q = k as f64 / n_bins as f64;
        let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
        edges.push(sorted[idx]);
    }

    // Count samples per bin.
    let mut base_counts = vec![0_usize; n_bins];
    let mut curr_counts = vec![0_usize; n_bins];
    for &x in baseline {
        base_counts[bin_index(x, &edges)] += 1;
    }
    for &x in current {
        curr_counts[bin_index(x, &edges)] += 1;
    }

    let base_total = baseline.len() as f64 + eps * n_bins as f64;
    let curr_total = current.len() as f64 + eps * n_bins as f64;
    let mut total = 0.0_f64;
    for k in 0..n_bins {
        let p = (base_counts[k] as f64 + eps) / base_total;
        let q = (curr_counts[k] as f64 + eps) / curr_total;
        total += (q - p) * (q / p).ln();
    }
    if !total.is_finite() {
        return Err(RagDriftError::NumericalInstability("psi"));
    }
    // PSI is non-negative by construction; floating-point can produce tiny negatives.
    Ok(total.max(0.0))
}

fn bin_index(x: f64, edges: &[f64]) -> usize {
    // edges has n_bins - 1 entries, sorted ascending. Returns 0..=n_bins-1.
    match edges.binary_search_by(|e| e.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i + 1, // x equals edge: place in upper bin
        Err(i) => i,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_psi_near_zero() {
        let xs: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let p = psi(&xs, &xs, 10, 1e-4).unwrap();
        assert!(p < 1e-3, "psi was {}", p);
    }

    #[test]
    fn shifted_samples_have_high_psi() {
        let baseline: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let current: Vec<f64> = (0..1000).map(|i| (i + 500) as f64).collect();
        let p = psi(&baseline, &current, 10, 1e-4).unwrap();
        // A 50% population shift in a uniform should land well above the 0.25 alert threshold.
        assert!(p > 0.25, "psi was {}", p);
    }

    #[test]
    fn small_n_bins_rejected() {
        let xs = vec![1.0, 2.0, 3.0];
        assert!(psi(&xs, &xs, 1, 1e-4).is_err());
    }

    #[test]
    fn insufficient_baseline_rejected() {
        let xs = vec![1.0, 2.0, 3.0];
        assert!(psi(&xs, &xs, 10, 1e-4).is_err());
    }
}
