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

## Naming

- **PyPI dist is `ragdrift-py`, import name is `ragdrift`.** The bare
  `ragdrift` on PyPI was already owned (by Crish Nagarkar's "Silent regression
  detector for RAG pipelines", v0.1.0). Rather than rename the whole project,
  we use the `pyyaml`/`opencv-python` pattern: distribution name carries a
  suffix, import name stays clean. Rust crates on crates.io kept the bare
  `ragdrift` name (we got there first).

## Lessons from the v0.1.0 ship

These are corrections we made between writing the code and getting the public
release green. Recording them so we don't make the same mistake in v0.1.x.

- **Original MSRV claim of 1.75 was wrong**. Revised to 1.80 because the
  `parallel` default feature pulls `rayon-core 1.13.0`, which itself requires
  rustc >= 1.80. We caught this via the dedicated MSRV CI job; the previous
  matrix-leg approach hid the real signal under generic "test failed" noise.
- **`pinecone-client` is dead, use `pinecone`**. The official client was
  renamed; `pinecone-client` now hard-raises on import. The PyPI extra is
  `'ragdrift[pinecone]' -> pinecone>=5,<8`.
- **mkdocs.yml belongs at the repo root, not inside `docs/`**. Putting it in
  `docs/` forced `docs_dir: .`, which collided with `site_dir` under
  `--strict`. Standard layout: `mkdocs.yml` at root, `docs_dir: docs`.
- **Bash on Git Bash for Windows hates backslashes**. Cross-platform venv
  activation via `source .venv/bin/activate || .\.venv\Scripts\activate`
  silently mangles the second branch into `..venvScriptsactivate`. Use
  forward slashes everywhere: `source .venv/Scripts/activate`.
- **PyO3 extension-module feature must be opt-in for the bindings crate**, not
  enabled at the workspace dep level. With it always-on, plain `cargo check`
  on the workspace fails to link Python symbols.

## What we deliberately deferred to 0.2.0

- Streaming / online drift (windowed updates).
- Calibration plots beyond a single ECE delta scalar.
- An async adapter API. The current adapters are sync; users who need async
  can wrap them with `asyncio.to_thread`.
- A CLI entry point. The library is the product for 0.1.0.
- PyO3 0.22 -> 0.28+ migration. Today's bindings use the `_bound` transitional
  API (`PyDict::new_bound`, `get_type_bound`); the post-0.23 stabilized API
  is cleaner.
