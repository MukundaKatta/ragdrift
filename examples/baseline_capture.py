"""Capture a baseline snapshot, save to disk, load it back later.

Run::

    python examples/baseline_capture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ragdrift import BaselineSnapshot


def main() -> None:
    rng = np.random.default_rng(seed=0)

    # Pretend this is your last good production window.
    baseline_emb = rng.normal(size=(500, 32)).astype(np.float32)
    baseline_lengths = rng.integers(80, 200, size=500).astype(int).tolist()
    baseline_confs = rng.uniform(0.5, 0.95, size=500).tolist()

    # BaselineSnapshot is the round-trip JSON shape that detectors persist.
    # We write a richer JSON envelope here that wraps both the snapshot and
    # the raw arrays you'll need at detection time.
    out = Path("/tmp/ragdrift_baseline.json")
    payload = {
        "snapshot": json.loads(BaselineSnapshot().to_json()),
        "embeddings": baseline_emb.tolist(),
        "lengths": baseline_lengths,
        "confidences": baseline_confs,
    }
    out.write_text(json.dumps(payload))
    print(f"baseline written to {out} ({out.stat().st_size:,} bytes)")

    loaded = json.loads(out.read_text())
    arr = np.asarray(loaded["embeddings"], dtype=np.float32)
    print(f"loaded embeddings: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
