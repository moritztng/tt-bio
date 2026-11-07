#!/usr/bin/env python3
"""Build the two-shape input tree this row times, from pvo-gpu's already-frozen trees.

    python3 make_kit_inputs.py ~/pvo-gpu-prep/inputs-timing ~/pvo-gpu-prep/inputs-9dbp ~/pvo-kit-prep/inputs-kit

9DBP (791 tokens, PopVax's shape) and 9TH6 (679) only. Nothing is re-frozen and no MSA is rebuilt: the files are
copied byte for byte and their recorded sha256 is carried over, then re-checked here, so the kit arms read exactly
the bytes pvo-gpu's non-kit arms read. A file whose bytes no longer match the hash it was frozen with is an error,
not a refreeze -- that is the whole point of the one-protocol rule.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

KEEP = ("9DBP", "9TH6")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    srcs = [Path(p).expanduser() for p in sys.argv[1:3]]
    out = Path(sys.argv[3]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    merged = {"sha256": {}, "depth": {}, "tokens": {}}
    for src in srcs:
        idx = json.loads((src / "INDEX.json").read_text())
        for rel, want in idx["sha256"].items():
            if not any(rel.startswith(k) or f"/{k}/" in rel for k in KEEP):
                continue
            got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
            if got != want:
                sys.exit(f"{src/rel}: sha256 {got} != frozen {want}")
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dst)
            merged["sha256"][rel] = want
        for field in ("depth", "tokens"):
            for k, v in (idx.get(field) or {}).items():
                if any(k.startswith(x) or f"/{x}/" in k or x in k for x in KEEP):
                    merged[field][k] = v
    (out / "INDEX.json").write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    print(f"{out}: {len(merged['sha256'])} files, tokens {merged['tokens']}")


if __name__ == "__main__":
    main()
