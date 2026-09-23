"""One length rung through the shipped CLI: does `tt-bio embed|saprot` return the right-shaped vector.

    python3 perf/mgx_embed/ceiling.py --model esmc-6b --L 1968 --pdb mtor.pdb --out rows.jsonl \
        [--fast] [--guard-off]

The sequence is the --pdb chain tiled to L (capacity only; accuracy is accuracy.py's job on an
untiled chain). --guard-off sets TT_BIO_SIZE_LIMIT=0 so a rung above the published cap reaches the
device and fails on the allocator rather than on the refusal. The row keeps the allocator line
of a failure, because "allocated / free / largest free block" is what separates weights,
activations and fragmentation.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "mgx_embed"))
from perf.clocksample import during  # noqa: E402
from accuracy import pdb_sequence  # noqa: E402

OOM = re.compile(r"(Out of Memory.*|.*largest free block.*|SizeTooLargeError.*)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--guard-off", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    full = pdb_sequence(a.pdb)
    seq = (full * (a.L // len(full) + 1))[:a.L]
    work = Path(tempfile.mkdtemp(prefix=f"mgxe_{a.model}_{a.L}_", dir=os.environ.get("S")))
    (work / "in.fasta").write_text(f">L{a.L}\n{seq}\n")
    verb = "saprot" if a.model.startswith("saprot") else "embed"
    cmd = [sys.executable, "-m", "tt_bio.main", verb, str(work / "in.fasta"), "--model", a.model,
           "--out_dir", str(work / "out")] + (["--fast"] if a.fast else [])
    env = dict(os.environ, **({"TT_BIO_SIZE_LIMIT": "0"} if a.guard_off else {}))
    with during() as clk:
        t = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
        wall = time.time() - t
    (work / "log.txt").write_text(r.stdout + r.stderr)
    rows = finite = None
    npz = work / "out" / f"L{a.L}.npz"
    if npz.is_file():
        z = np.load(npz)
        rows, finite = int(z["per_residue"].shape[0]), bool(np.isfinite(z["per_residue"]).all())
    ok = r.returncode == 0 and rows == a.L and finite
    oom = OOM.findall(r.stdout + r.stderr)
    row = {"model": a.model, "fast": a.fast, "L": a.L, "guard_off": a.guard_off, "ok": ok,
           "rc": r.returncode, "rows": rows, "finite": finite, "wall_s": round(wall, 1),
           "aiclk": clk.summary(), "fail_line": oom[-1][:400] if oom else None,
           "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"), "log": str(work / "log.txt")}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    if ok:
        for f in (work / "out").glob("*.npz"):
            f.unlink()


if __name__ == "__main__":
    main()
