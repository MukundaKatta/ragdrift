# Five dimensions

Each detector returns a `DriftScore` with a `score`, a `threshold`, and an
`exceeded` flag. The score is non-negative and comparable within a dimension
across runs. Across dimensions, scores are *not* comparable on absolute scale —
set thresholds per-dimension.

## 1. Data drift

Per-feature distribution shift on tabular features (latency, retrieval count,
prompt token count, etc.). For each feature column the detector computes:

- **KS statistic** D — the maximum gap between the two empirical CDFs.
- **PSI** — Population Stability Index, weighted log-ratio of bin masses.

The reported score is `max(D, PSI)` across all features. KS dominates when the
*shape* of the distribution changes; PSI dominates when bin masses redistribute.

PSI thresholds (from credit-risk practice): `<0.10` stable, `0.10–0.25` moderate,
`>0.25` significant.

## 2. Embedding drift

Distribution shift on the embedding vectors themselves. Combines:

- **MMD²** with an RBF kernel, unbiased estimator (Gretton et al., *A Kernel
  Two-Sample Test*, JMLR 2012). Bandwidth is the median heuristic on the
  pooled pairwise distances.
- **Sliced Wasserstein-1** (Bonneel et al., *Sliced and Radon Wasserstein
  Barycenters of Measures*, J. Math. Imaging Vis. 2015). Projects onto N
  random unit directions and averages 1D Wasserstein on each projection.

Combined score is `max(0, MMD²) + SW`. MMD is sample-efficient and sensitive
to subtle shape changes; SW is bandwidth-free and geometric. Together they
catch more failure modes than either alone.

## 3. Response drift

Length distribution shift via KS on response lengths (characters or tokens
— pick one and stick with it). Optionally adds a sliced-Wasserstein semantic
shift on response embeddings.

## 4. Confidence drift

KS on the distribution of confidence scores. With ground-truth correctness
labels, also tracks **|ECE_current − ECE_baseline|** where ECE is the Expected
Calibration Error (Naeini et al., *Obtaining Well-Calibrated Probabilities
Using Bayesian Binning*, AAAI 2015). Detects "the model got more confident but
no better calibrated" silently.

## 5. Query-pattern drift

Captures workload composition shift. The detector clusters baseline query
embeddings into k centroids, then assigns the current queries to those same
centroids and measures the symmetric KL divergence between the two assignment
distributions. Sensitive to "the mix of intents changed" without being sensitive
to within-intent variation.

## Composing them

`RagDriftMonitor` runs every dimension you provide inputs for and returns a
single `DriftReport`. Set per-dimension thresholds based on your false-positive
budget; defaults are deliberately conservative.
