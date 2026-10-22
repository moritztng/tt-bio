"""The shipped benchmark input library, as a manifest the test suite can check against.

Upstream ships its benchmark inputs as annotated CIFs under `data/benchmarks/` and every
how-to starts from one of them, so they are the port's natural parity fixtures. They are not
vendored (8.8 MB, upstream's data), so this builds a manifest from a clone instead: per input
its sha256, token count and the conditioning keys that are actually set.

    RFD4_BENCHMARKS=<clone>/data/benchmarks python perf/rfd4/benchmarks.py build

Writes `perf/rfd4/benchmark_manifest.json`. `tests/test_rfd4_benchmark_inputs.py` checks the
manifest's shape everywhere and re-featurizes every input where the library is on the box.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "benchmark_manifest.json"
DEFAULT = Path(os.environ.get(
    "RFD4_BENCHMARKS",
    Path.home() / "rfd4-ref-env/src/RFD4-Proteina/data/benchmarks"))


def library(root: Path = DEFAULT) -> list[Path]:
    """Every annotated CIF the library ships, in a stable order."""
    return sorted(root.rglob("*.cif.zst"))


def describe(path: Path, root: Path) -> dict:
    """One input's identity and what featurizing it produces."""
    from tt_bio.rfd4 import featurize

    features, _ = featurize(path, seed=0)
    set_conditions = sorted(
        key for key, value in features.items()
        if key.startswith("condition_") and hasattr(value, "any") and bool(value.any()))
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokens": int(features["padding_mask"].shape[-1]),
        "conditions": set_conditions,
    }


def build(root: Path) -> dict:
    inputs = library(root)
    if not inputs:
        raise SystemExit(f"no .cif.zst inputs under {root}")
    rows, failed = [], []
    for i, path in enumerate(inputs, 1):
        try:
            rows.append(describe(path, root))
        except Exception as e:  # an input the port cannot build is the point of the manifest
            failed.append({"path": str(path.relative_to(root)), "error": f"{type(e).__name__}: {e}"})
        if i % 25 == 0:
            print(f"{i}/{len(inputs)}", flush=True)
    return {
        "_note": "built by perf/rfd4/benchmarks.py from upstream's data/benchmarks; "
                 "sha256 is the shipped file, tokens and conditions are what tt_bio.rfd4.featurize "
                 "produces at seed 0",
        "families": sorted({row["path"].split("/")[0] for row in rows}),
        "inputs": rows,
        "failed": failed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    manifest = build(args.root)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest['inputs'])} inputs, {len(manifest['failed'])} failed -> {MANIFEST}",
          file=sys.stderr)
