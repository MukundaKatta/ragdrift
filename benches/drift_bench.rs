//! Criterion benchmarks for the embedding-heavy hot paths.
//!
//! Run with: `cargo bench -p ragdrift-core`.
//! Numbers are machine-dependent — re-run on your hardware before quoting.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use ragdrift_core::detectors::{
    DataDriftDetector, EmbeddingDriftDetector, QueryDriftDetector, ResponseDriftDetector,
};
use ragdrift_core::stats::{mmd_rbf, sliced_wasserstein, MmdEstimator};

fn make_emb(n: usize, d: usize, seed: u64, shift: f32) -> Array2<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut a: Array2<f32> = Array2::random_using((n, d), Normal::new(0.0, 1.0).unwrap(), &mut rng);
    a += shift;
    a
}

fn bench_mmd(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmd_rbf");
    for &n in &[1000_usize, 5000, 10_000] {
        let dim = 768;
        let x = make_emb(n, dim, 0, 0.0);
        let y = make_emb(n, dim, 1, 0.5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| mmd_rbf(x.view(), y.view(), None, MmdEstimator::Unbiased).unwrap());
        });
    }
    group.finish();
}

fn bench_sliced_w(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliced_wasserstein");
    for &n in &[1000_usize, 5000, 10_000] {
        let dim = 768;
        let x = make_emb(n, dim, 0, 0.0);
        let y = make_emb(n, dim, 1, 0.5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sliced_wasserstein(x.view(), y.view(), 64, 7).unwrap());
        });
    }
    group.finish();
}

fn bench_full_report(c: &mut Criterion) {
    let n = 2000;
    let dim = 384;
    let x = make_emb(n, dim, 0, 0.0);
    let y = make_emb(n, dim, 1, 0.5);
    let feat_x = Array2::<f64>::from_shape_fn((n, 8), |(i, j)| (i + j) as f64);
    let feat_y = Array2::<f64>::from_shape_fn((n, 8), |(i, j)| (i + j) as f64 + 0.3);
    let lens_x: Vec<f64> = (0..n).map(|i| (i % 200) as f64).collect();
    let lens_y: Vec<f64> = (0..n).map(|i| (i % 200) as f64 + 5.0).collect();

    c.bench_function("five_dimension_report", |b| {
        b.iter(|| {
            let _ = EmbeddingDriftDetector::new(0.05)
                .detect(x.view(), y.view())
                .unwrap();
            let _ = DataDriftDetector::new(0.1)
                .detect(feat_x.view(), feat_y.view())
                .unwrap();
            let _ = ResponseDriftDetector::new(0.2)
                .detect(&lens_x, &lens_y)
                .unwrap();
            let _ = QueryDriftDetector::new(0.1, 8)
                .with_seed(7)
                .detect(x.view(), y.view())
                .unwrap();
        })
    });
}

criterion_group!(benches, bench_mmd, bench_sliced_w, bench_full_report);
criterion_main!(benches);
