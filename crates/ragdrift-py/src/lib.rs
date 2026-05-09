//! PyO3 bindings for ragdrift-core. Exposes all five detectors plus the
//! supporting types as the `ragdrift._native` module.

// PyO3 0.22's `create_exception!` macro emits a `cfg(feature = "gil-refs")` check
// that triggers `unexpected_cfgs` since this crate doesn't declare that feature.
// Harmless and removed in 0.23+.
#![allow(unexpected_cfgs)]

use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ragdrift_core::detectors::{
    ConfidenceDriftDetector as RConfidence, DataDriftDetector as RData,
    EmbeddingDriftDetector as REmb, QueryDriftDetector as RQuery, ResponseDriftDetector as RResp,
};
use ragdrift_core::types::{
    BaselineSnapshot as RSnap, DriftDimension as RDim, DriftReport as RReport, DriftScore as RScore,
};
use ragdrift_core::RagDriftError as CoreError;

create_exception!(ragdrift._native, RagDriftError, PyException);

fn map_err(e: CoreError) -> PyErr {
    RagDriftError::new_err(e.to_string())
}

fn dim_to_str(d: RDim) -> &'static str {
    d.as_str()
}

// --------- DriftScore ---------

#[pyclass(name = "DriftScore", frozen, module = "ragdrift._native")]
#[derive(Clone)]
struct PyDriftScore {
    inner: RScore,
}

#[pymethods]
impl PyDriftScore {
    #[getter]
    fn dimension(&self) -> &'static str {
        dim_to_str(self.inner.dimension)
    }
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }
    #[getter]
    fn exceeded(&self) -> bool {
        self.inner.exceeded
    }
    #[getter]
    fn method(&self) -> &str {
        &self.inner.method
    }
    fn __repr__(&self) -> String {
        format!(
            "DriftScore(dimension='{}', score={:.6}, threshold={:.6}, exceeded={}, method='{}')",
            dim_to_str(self.inner.dimension),
            self.inner.score,
            self.inner.threshold,
            self.inner.exceeded,
            self.inner.method
        )
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("dimension", dim_to_str(self.inner.dimension))?;
        d.set_item("score", self.inner.score)?;
        d.set_item("threshold", self.inner.threshold)?;
        d.set_item("exceeded", self.inner.exceeded)?;
        d.set_item("method", &self.inner.method)?;
        Ok(d)
    }
}

// --------- DriftReport ---------

#[pyclass(name = "DriftReport", frozen, module = "ragdrift._native")]
#[derive(Clone)]
struct PyDriftReport {
    inner: RReport,
}

#[pymethods]
impl PyDriftReport {
    #[new]
    fn new(
        scores: Vec<PyDriftScore>,
        sample_size_baseline: usize,
        sample_size_current: usize,
    ) -> Self {
        let inner = RReport::new(
            scores.into_iter().map(|s| s.inner).collect(),
            sample_size_baseline,
            sample_size_current,
        );
        Self { inner }
    }
    #[getter]
    fn scores(&self) -> Vec<PyDriftScore> {
        self.inner
            .scores
            .iter()
            .cloned()
            .map(|s| PyDriftScore { inner: s })
            .collect()
    }
    #[getter]
    fn timestamp(&self) -> i64 {
        self.inner.timestamp
    }
    #[getter]
    fn sample_size_baseline(&self) -> usize {
        self.inner.sample_size_baseline
    }
    #[getter]
    fn sample_size_current(&self) -> usize {
        self.inner.sample_size_current
    }
    fn any_exceeded(&self) -> bool {
        self.inner.any_exceeded()
    }
    fn max_score(&self) -> f64 {
        self.inner.max_score()
    }
    fn __repr__(&self) -> String {
        format!(
            "DriftReport(scores={}, max_score={:.6}, any_exceeded={})",
            self.inner.scores.len(),
            self.inner.max_score(),
            self.inner.any_exceeded()
        )
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| RagDriftError::new_err(format!("serialize: {}", e)))
    }
}

// --------- BaselineSnapshot ---------

#[pyclass(name = "BaselineSnapshot", module = "ragdrift._native")]
#[derive(Clone)]
struct PyBaselineSnapshot {
    inner: RSnap,
}

#[pymethods]
impl PyBaselineSnapshot {
    #[new]
    fn new() -> Self {
        Self {
            inner: RSnap::empty(),
        }
    }
    #[getter]
    fn captured_at(&self) -> i64 {
        self.inner.captured_at
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| RagDriftError::new_err(format!("serialize: {}", e)))
    }
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: RSnap = serde_json::from_str(s)
            .map_err(|e| RagDriftError::new_err(format!("deserialize: {}", e)))?;
        Ok(Self { inner })
    }
}

// --------- EmbeddingDrift ---------

#[pyclass(name = "EmbeddingDrift", module = "ragdrift._native")]
struct PyEmbeddingDrift {
    inner: REmb,
}

#[pymethods]
impl PyEmbeddingDrift {
    #[new]
    #[pyo3(signature = (threshold=0.05, n_projections=64, seed=0))]
    fn new(threshold: f64, n_projections: usize, seed: u64) -> Self {
        Self {
            inner: REmb::new(threshold)
                .with_projections(n_projections)
                .with_seed(seed),
        }
    }

    fn detect(
        &self,
        py: Python<'_>,
        baseline: PyReadonlyArray2<f32>,
        current: PyReadonlyArray2<f32>,
    ) -> PyResult<PyDriftScore> {
        let b = baseline.as_array().to_owned();
        let c = current.as_array().to_owned();
        let result = py.allow_threads(|| self.inner.detect(b.view(), c.view()));
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }
}

// --------- DataDrift ---------

#[pyclass(name = "DataDrift", module = "ragdrift._native")]
struct PyDataDrift {
    inner: RData,
}

#[pymethods]
impl PyDataDrift {
    #[new]
    #[pyo3(signature = (threshold=0.1, n_bins=10))]
    fn new(threshold: f64, n_bins: usize) -> Self {
        Self {
            inner: RData::new(threshold).with_bins(n_bins),
        }
    }

    fn detect(
        &self,
        py: Python<'_>,
        baseline: PyReadonlyArray2<f64>,
        current: PyReadonlyArray2<f64>,
    ) -> PyResult<PyDriftScore> {
        let b = baseline.as_array().to_owned();
        let c = current.as_array().to_owned();
        let result = py.allow_threads(|| self.inner.detect(b.view(), c.view()));
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }
}

// --------- ResponseDrift ---------

#[pyclass(name = "ResponseDrift", module = "ragdrift._native")]
struct PyResponseDrift {
    inner: RResp,
}

#[pymethods]
impl PyResponseDrift {
    #[new]
    #[pyo3(signature = (threshold=0.2, seed=0))]
    fn new(threshold: f64, seed: u64) -> Self {
        Self {
            inner: RResp::new(threshold).with_seed(seed),
        }
    }

    fn detect(
        &self,
        py: Python<'_>,
        baseline_lengths: Vec<f64>,
        current_lengths: Vec<f64>,
    ) -> PyResult<PyDriftScore> {
        let result = py.allow_threads(|| self.inner.detect(&baseline_lengths, &current_lengths));
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }

    fn detect_with_embeddings(
        &self,
        py: Python<'_>,
        baseline_lengths: Vec<f64>,
        current_lengths: Vec<f64>,
        baseline_embeddings: PyReadonlyArray2<f32>,
        current_embeddings: PyReadonlyArray2<f32>,
    ) -> PyResult<PyDriftScore> {
        let be: Array2<f32> = baseline_embeddings.as_array().to_owned();
        let ce: Array2<f32> = current_embeddings.as_array().to_owned();
        let result = py.allow_threads(|| {
            self.inner.detect_with_embeddings(
                &baseline_lengths,
                &current_lengths,
                be.view(),
                ce.view(),
            )
        });
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }
}

// --------- ConfidenceDrift ---------

#[pyclass(name = "ConfidenceDrift", module = "ragdrift._native")]
struct PyConfidenceDrift {
    inner: RConfidence,
}

#[pymethods]
impl PyConfidenceDrift {
    #[new]
    #[pyo3(signature = (threshold=0.2, n_calib_bins=10))]
    fn new(threshold: f64, n_calib_bins: usize) -> Self {
        Self {
            inner: RConfidence::new(threshold).with_calib_bins(n_calib_bins),
        }
    }

    fn detect(
        &self,
        py: Python<'_>,
        baseline_confidences: Vec<f64>,
        current_confidences: Vec<f64>,
    ) -> PyResult<PyDriftScore> {
        let result = py.allow_threads(|| {
            self.inner
                .detect(&baseline_confidences, &current_confidences)
        });
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }

    fn detect_with_correctness(
        &self,
        py: Python<'_>,
        baseline_confidences: Vec<f64>,
        baseline_correct: Vec<bool>,
        current_confidences: Vec<f64>,
        current_correct: Vec<bool>,
    ) -> PyResult<PyDriftScore> {
        let result = py.allow_threads(|| {
            self.inner.detect_with_correctness(
                &baseline_confidences,
                &baseline_correct,
                &current_confidences,
                &current_correct,
            )
        });
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }
}

// --------- QueryDrift ---------

#[pyclass(name = "QueryDrift", module = "ragdrift._native")]
struct PyQueryDrift {
    inner: RQuery,
}

#[pymethods]
impl PyQueryDrift {
    #[new]
    #[pyo3(signature = (threshold=0.1, k=8, seed=0))]
    fn new(threshold: f64, k: usize, seed: u64) -> Self {
        Self {
            inner: RQuery::new(threshold, k).with_seed(seed),
        }
    }

    fn detect(
        &self,
        py: Python<'_>,
        baseline: PyReadonlyArray2<f32>,
        current: PyReadonlyArray2<f32>,
    ) -> PyResult<PyDriftScore> {
        let b = baseline.as_array().to_owned();
        let c = current.as_array().to_owned();
        let result = py.allow_threads(|| self.inner.detect(b.view(), c.view()));
        result.map(|s| PyDriftScore { inner: s }).map_err(map_err)
    }
}

// --------- module init ---------

#[pymodule]
#[pyo3(name = "_native")]
fn ragdrift_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("RagDriftError", py.get_type_bound::<RagDriftError>())?;
    m.add_class::<PyDriftScore>()?;
    m.add_class::<PyDriftReport>()?;
    m.add_class::<PyBaselineSnapshot>()?;
    m.add_class::<PyEmbeddingDrift>()?;
    m.add_class::<PyDataDrift>()?;
    m.add_class::<PyResponseDrift>()?;
    m.add_class::<PyConfidenceDrift>()?;
    m.add_class::<PyQueryDrift>()?;
    Ok(())
}
