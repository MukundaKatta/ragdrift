# Contributing

Thanks for considering a contribution.

## Quick start

```bash
git clone https://github.com/MukundaKatta/ragdrift
cd ragdrift
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,opensearch,pgvector,pinecone,prometheus,datadog,aws]"
maturin develop
pytest
cargo test --all-features
```

## Quality gates

Before opening a PR, all of the following must pass locally:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
maturin develop
pytest
mypy --strict python/ragdrift
ruff check .
ruff format --check .
```

CI runs the same matrix on Linux, macOS, and Windows for Python 3.10 through 3.13.

## Where things live

- `crates/ragdrift-core/` — pure Rust crate, no Python dependencies.
- `crates/ragdrift-py/` — PyO3 bindings only. No business logic.
- `python/ragdrift/` — type stubs, adapters, exporters, high-level facade.
- `tests/python/` — Python tests. All under 5 seconds, no network.

## Style

- Rust: rustfmt defaults, clippy-clean. New stats live in `crates/ragdrift-core/src/stats/`.
- Python: ruff + mypy --strict. Public functions need a one-line docstring; complex ones need an example.

## Adding a detector or stat

1. Implement and unit-test it in `ragdrift-core`.
2. If it is part of the public API, expose it in `crates/ragdrift-py/src/lib.rs`.
3. Update `python/ragdrift/_native.pyi`.
4. Re-export it from `python/ragdrift/__init__.py`.
5. Add a Python smoke test under `tests/python/`.

## Commit style

Conventional Commits are encouraged but not required. Keep PRs small and focused.
