//! The five drift detectors. Each is a lightweight struct configured once and
//! invoked many times — the cost is in the sample data, not the detector.

pub mod confidence;
pub mod data;
pub mod embedding;
pub mod query;
pub mod response;

pub use confidence::ConfidenceDriftDetector;
pub use data::DataDriftDetector;
pub use embedding::EmbeddingDriftDetector;
pub use query::QueryDriftDetector;
pub use response::ResponseDriftDetector;
