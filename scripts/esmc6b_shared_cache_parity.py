"""Bit-exact parity gate for the ESMC-6B shared /dev/shm tiled-weight cache.

The fanout perf fix (`esmc.load_esmc6b_shared` + `tenstorrent.weight_cache`) makes
the N data-parallel workers share one host-tiled copy of the 24 GB checkpoint: the
first worker (`dump` mode) tiles every weight once and publishes it to a cache dir;
peers (`load` mode) `ttnn.load_tensor` the pre-tiled weight straight to their card.
This must be numerically identical to the untouched single-card path — a dumped tile
is exactly what `from_torch` would have produced, so the device tensors are identical.

Three short-lived pinned subprocesses on ONE leased card (so the parent never holds
the card): the single-card baseline (`load_esmc`, cache disabled), the cache *builder*
(`dump`), and a cache *peer* (`load`). We then assert Δ == 0 per-residue and pooled
across all three. End-to-end concurrent multi-card wall-clock scaling needs 2+ free
cards and is out of scope here.

Usage:
    TT_VISIBLE_DEVICES=0 python3 scripts/esmc6b_shared_cache_parity.py --n 8 --len 192
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _make_sequences(n: int, length: int) -> dict[str, str]:
    import random

    rng = random.Random(42)
    aa = "LAGVSERTIDPKQNFYMHWC"
    return {f"seq{i}": "".join(rng.choice(aa) for _ in range(length)) for i in range(n)}


def _run_pass(mode: str, seqs: dict[str, str], cache_dir: str | None, out_path: str) -> None:
    """Subprocess entry point: embed `seqs` with esmc-6b via the chosen load path."""
    from tt_bio import esmc

    if mode == "baseline":
        model = esmc.load_esmc("esmc-6b", fast=False)
    else:  # "build" (dump) or "load" — both go through the shared cache
        model = esmc.load_esmc6b_shared(cache_dir, name="esmc-6b", fast=False)
    res = esmc.embed_sequences(model, seqs, pool="mean")
    arrays = {}
    for e in res:
        arrays[f"{e.id}.res"] = e.per_residue
        arrays[f"{e.id}.pool"] = e.pooled
    np.savez(out_path, **arrays)


def _spawn(mode: str, seqs: dict[str, str], cache_dir: str | None, work: str) -> str:
    """Run one pass in a fresh pinned interpreter; return its npz path."""
    req = os.path.join(work, f"{mode}.req.pkl")
    out = os.path.join(work, f"{mode}.npz")
    with open(req, "wb") as f:
        pickle.dump(dict(mode=mode, seqs=seqs, cache_dir=cache_dir, out=out), f)
    code = (
        "import pickle, sys;"
        "from scripts.esmc6b_shared_cache_parity import _run_pass;"
        "r=pickle.load(open(sys.argv[1],'rb'));"
        "_run_pass(r['mode'], r['seqs'], r['cache_dir'], r['out'])"
    )
    subprocess.run([sys.executable, "-c", code, req], check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="number of sequences")
    ap.add_argument("--len", type=int, default=192, dest="length", help="sequence length")
    args = ap.parse_args()

    seqs = _make_sequences(args.n, args.length)
    work = tempfile.mkdtemp(prefix="esmc6b-parity-")
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    cache_dir = tempfile.mkdtemp(prefix="esmc6b-tiles-", dir=shm)
    try:
        base = np.load(_spawn("baseline", seqs, None, work))
        build = np.load(_spawn("build", seqs, cache_dir, work))  # populates the cache
        load = np.load(_spawn("load", seqs, cache_dir, work))    # reads the cache
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    keys = sorted(base.files)
    ok = True
    for label, ref, other in (("build vs baseline", base, build),
                              ("load  vs baseline", base, load),
                              ("load  vs build   ", build, load)):
        maxd = max(float(np.abs(ref[k].astype(np.float64) - other[k].astype(np.float64)).max())
                   for k in keys)
        exact = all(np.array_equal(ref[k], other[k]) for k in keys)
        print(f"{label}: max|Δ|={maxd:.3e}  bit_exact={exact}  ({len(keys)} arrays)")
        ok = ok and exact and maxd == 0.0

    print("PARITY: PASS" if ok else "PARITY: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
