# Releasing ragdrift

How to cut a release. Captures everything we wish we'd known the first time.

## Versions to bump

A release bumps the version in **three** files plus the changelog:

```
Cargo.toml                       # workspace.package.version
crates/ragdrift-py/Cargo.toml    # path-dep version pin (= "x.y.z")
pyproject.toml                   # [project] version
CHANGELOG.md                     # add new section, update compare links
```

If you forget one, `cargo publish` succeeds but the wheel and the bare
`ragdrift` crate ship the wrong version.

## Quality gates (must all pass)

```bash
source .venv/bin/activate
cargo fmt --all -- --check
cargo clippy -p ragdrift-core --all-targets --all-features -- -D warnings
cargo test -p ragdrift-core --all-features
cargo publish --dry-run -p ragdrift-core
cargo publish --dry-run -p ragdrift
maturin build --release
maturin develop
pytest -q
mypy --strict python/ragdrift
ruff check .
ruff format --check .
```

CI runs the same on every push (Linux, macOS, Windows × Python 3.10–3.13).

## Crates.io publish

Two crates ship in tandem. They share a version. **Always publish core first**,
because `ragdrift` depends on `ragdrift-core` at the exact version.

```bash
# Get a fresh single-shot token: https://crates.io/settings/tokens
# Pick the smallest scope that works:
#   - 'publish-update' for a version bump on an existing crate
#   - 'publish-new'    only when adding a brand-new crate name
# Limit the scope to ragdrift + ragdrift-core (don't grant '*').

export CARGO_REGISTRY_TOKEN="ciok..."

cargo publish -p ragdrift-core
cargo publish -p ragdrift

unset CARGO_REGISTRY_TOKEN
# Revoke the token at https://crates.io/settings/tokens — single-use is the
# safest pattern.
```

## Tag and let release.yml do the rest

```bash
git tag -a vX.Y.Z -m "ragdrift vX.Y.Z: <one-line summary>"
git push origin vX.Y.Z
```

`release.yml` then:

1. Builds wheels for Linux/macOS/Windows × x86_64/aarch64 via `cibuildwheel`.
2. Builds the sdist.
3. Skips `cargo publish` cleanly if the version is already on crates.io
   (idempotent — safe to re-tag).
4. Publishes the wheel to PyPI via Trusted Publishing (see below).
5. Cuts a GitHub Release and attaches every wheel + the sdist.

The GitHub Release happens even if PyPI publishing fails (`if: always()`),
so users always get downloadable wheels off the release page.

## PyPI Trusted Publisher (one-time setup)

PyPI publishing uses OIDC, no API tokens. To set it up:

1. Make sure the project name is registered on PyPI. Verify with
   `curl -sI https://pypi.org/project/ragdrift/ | head -1` — should be `200`.
2. Visit https://pypi.org/manage/project/ragdrift/settings/publishing/.
3. Click **"Add a new publisher"**, then fill in **exactly**:
   ```
   Owner:               MukundaKatta
   Repository name:     ragdrift
   Workflow filename:   release.yml
   Environment name:    pypi
   ```
4. Save. Re-trigger the failed PyPI job:
   ```bash
   RUN=$(gh run list --repo MukundaKatta/ragdrift \
                     --workflow=release.yml --limit 1 \
                     --json databaseId --jq '.[0].databaseId')
   gh run rerun $RUN --failed --repo MukundaKatta/ragdrift
   ```

If you misspell *any* of the four fields, the OIDC handshake fails with a
generic "Trusted publishing exchange failure" — re-check spelling before
debugging anything else.

## Manual PyPI publish (fallback if Trusted Publisher is wedged)

```bash
# Token at https://pypi.org/manage/account/token/, scope to project: ragdrift
maturin upload \
  -u __token__ \
  -p "pypi-...REDACTED..." \
  target/wheels/*.whl
# Then revoke the token.
```

## docs.rs and GitHub Pages

Both auto-deploy. No action needed:
- docs.rs picks up the new crate version within ~5–15 min of publish.
- GitHub Pages re-deploys on any `main` push (`docs.yml` workflow).

## After the release

1. Verify on crates.io: `curl -s https://crates.io/api/v1/crates/ragdrift | jq '.crate.max_version'`
2. Verify on PyPI: `pip install ragdrift==X.Y.Z` in a fresh venv.
3. Verify the GitHub Release has every wheel attached.
4. Update the `[Unreleased]` link at the bottom of `CHANGELOG.md`.
5. Bump to `X.Y.(Z+1)-dev` on `main` *only if* you want to make the work-in-progress
   visible. We don't currently — versions stay at the last released number.

## When things go wrong

- **"Trusted publishing exchange failure"** → field misspelling, see PyPI section.
- **`cargo publish` says "version already exists"** → benign on a re-tag, the
  workflow's idempotency check turns this into a no-op exit 0.
- **MSRV CI job fails** → a transitive dep raised its MSRV. Check
  `cargo build -p ragdrift-core --all-features` against the pinned MSRV
  toolchain locally and bump `rust-version` in `Cargo.toml` if needed.
- **Windows CI fails** → check that any new bash glue uses forward slashes
  (Git Bash on Windows mangles backslashes silently).
