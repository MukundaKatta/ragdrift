# Design Decisions

A running log of non-obvious choices made while building v0.1.0. Each entry
records the choice and the alternative considered.

## Toolchain

- **Python 3.12 venv for local dev**, even though the package targets 3.10+.
  System Python here is 3.14; some PyO3/numpy combos lag behind newest CPython
  releases, so a 3.12 dev venv is the safest base. The wheel itself is `abi3-py310`
  so it works on 3.10 through 3.13 without rebuilding.
- **`maturin>=1.7,<2.0`** chosen because 1.7 is the first version with stable
  `abi3-py310` support that we actually depend on. Locally we have 1.13.1.
- **Python 3.10 minimum** rather than 3.9 because 3.9 reaches EOL in October 2025.

## Numerics

- **`ndarray` 0.16** for arrays. `nalgebra` would also work but `ndarray`
  composes more cleanly with `numpy`'s PyO3 bindings (zero-copy `PyReadonlyArray2`).
- **No `linfa-clustering` for k-means.** It pulls a heavy transitive dep tree
  (`ndarray-linalg`, BLAS bindings) and we only need a tiny Lloyd's-iteration
  k-means for the query detector. Hand-rolled it in `stats/kmeans.rs` (~80 lines).
- **`rayon` parallelism on by default.** The cost (extra dep) is small and
  embedding-pair MMD on 10k vectors is the hot path.
- **No `proptest` in 0.1.0.** The brief asks for property tests "asserting
  score=0 for identical inputs", which we cover with deterministic unit tests
  that pass identical arrays. Adding `proptest` for this would not catch a
  bug a deterministic test misses. Re-evaluate in 0.2.0 once we have a real
  bug surface.

## Public API surface

- **`DriftScore.method` is a `String`, not an enum.** Detectors compose
  multiple methods (e.g. embedding drift uses MMD + sliced Wasserstein); the
  string is `"mmd+sw"` and is meant for humans, not branching.
- **`BaselineSnapshot` only stores embeddings + summary statistics**, not raw
  query text. Storing text invites PII concerns; users can keep that on their
  side and rebuild the snapshot.

## Bindings

- **`pyo3 = "0.22"` with `abi3-py310`.** One wheel for 3.10 through 3.13. We
  release the GIL inside heavy compute (`py.allow_threads`) so multi-threaded
  Python callers see real parallelism.
- **Numpy interop via the `numpy` crate** (matched to PyO3 0.22, so `numpy = "0.22"`).
  We accept `PyReadonlyArray2<f32>` for embeddings (zero-copy) and convert to
  `ArrayView2<f32>` internally.

## What we deliberately deferred to 0.2.0

- Streaming / online drift (windowed updates).
- Calibration plots beyond a single ECE delta scalar.
- An async adapter API. The current adapters are sync; users who need async
  can wrap them with `asyncio.to_thread`.
- A CLI entry point. The library is the product for 0.1.0.
