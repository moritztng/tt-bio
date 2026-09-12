#!/usr/bin/env python3
"""Every head width `TT_BIO_HEAD_PAD_TAIL` fires at, censused and then priced, in one fold.

Pass 1 measured the lever on the diffusion step and then found the property gate accepts 96
`AttentionPairBias` objects, only 24 of which are in that step. The other 72 are at
`(n_heads 16, head_dim 24, padded 32)` in the trunk, and the trunk is 12.4 s of the fold against
the sampler's 5.3 s -- so an unmeasured sign there can outweigh everything the step gained. That is
what this settles.

One process, one model load, one real 512 aa fold:

  1. a counting hook on `AttentionPairBias.__call__` records, per (n_heads, head_dim, padded),
     how many calls the fold actually makes and what they cost. Counting is a dict update, and the
     wall it records is the caller's, so it is an attribution, not a benchmark;
  2. the first call at each width has its operands cloned and kept;
  3. after the fold, those saved calls are replayed with the arms ALTERNATING in one process, which
     is the only way the campaign accepts a ratio, plus the arm-against-arm parity read.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x-flag-levers"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="512")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import ab_flag_levers as AB
    import shutil
    import tempfile

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "size": a.size,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="b2z2-site-ab-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / f"cdk2x2_{a.size}.yaml", (AB.FIX / f"cdk2x2_{a.size}.a3m").read_text(),
                 msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-site-ab", cfg)

    # --- 1 + 2: census the fold, and keep one call at each width -------------------------------
    T._HEAD_PAD_TAIL = False
    census: dict = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0})
    saved: dict = {}
    orig = T.AttentionPairBias.__call__

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def hook(self_obj, *args, **kw):
        key = (self_obj.n_heads, self_obj.head_dim,
               getattr(self_obj, "padded_head_dim", self_obj.head_dim),
               bool(getattr(self_obj, "pad_tail", False)))
        if key not in saved and key[3]:
            saved[key] = {"obj": self_obj, "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
        t0 = time.perf_counter()
        out = orig(self_obj, *args, **kw)
        c = census[key]
        c["calls"] += 1
        c["wall_ms"] += 1e3 * (time.perf_counter() - t0)
        return out

    T.AttentionPairBias.__call__ = hook
    t0 = time.perf_counter()
    try:
        state.predict_one(AB.FIX / f"cdk2x2_{a.size}.yaml", cfg)
    finally:
        T.AttentionPairBias.__call__ = orig
    ttnn.synchronize_device(dev)
    OUT["fold_s"] = round(time.perf_counter() - t0, 3)
    OUT["census"] = {f"n_heads={k[0]} head_dim={k[1]} padded={k[2]} eligible={k[3]}":
                     {"calls": v["calls"], "wall_ms": round(v["wall_ms"], 2),
                      "us_per_call": round(1e3 * v["wall_ms"] / v["calls"], 2)}
                     for k, v in sorted(census.items())}
    dump()
    for k, v in OUT["census"].items():
        print(f"  {k:55s} {v['calls']:5d} calls  {v['wall_ms']:9.2f} ms  "
              f"{v['us_per_call']:8.2f} us/call", flush=True)

    # --- 3: the paired A/B at each eligible width ----------------------------------------------
    OUT["ab"] = {}
    for key, g in saved.items():
        tag = f"n_heads={key[0]} head_dim={key[1]} padded={key[2]}"
        call = lambda: g["obj"](*g["args"], **g["kwargs"])                    # noqa: E731
        for on in (False, True):
            T._HEAD_PAD_TAIL = on
            for _ in range(3):
                call()
        ttnn.synchronize_device(dev)

        walls = {"off": [], "on": []}
        order = []
        for b in range(a.blocks):
            for on in (False, True) if b % 2 == 0 else (True, False):
                T._HEAD_PAD_TAIL = on
                t = time.perf_counter()
                for _ in range(a.reps):
                    call()
                ttnn.synchronize_device(dev)
                w = 1e6 * (time.perf_counter() - t) / a.reps
                walls["on" if on else "off"].append(w)
                order.append(("on" if on else "off", round(w, 2)))
        T._HEAD_PAD_TAIL = False

        med = {k: st.median(v) for k, v in walls.items()}
        off = walls["off"]
        h = len(off) // 2
        # parity at this site, arm against arm
        T._HEAD_PAD_TAIL = False
        r_off = ttnn.to_torch(call()).clone()
        T._HEAD_PAD_TAIL = True
        r_on = ttnn.to_torch(call()).clone()
        T._HEAD_PAD_TAIL = False
        d = (r_off.float() - r_on.float()).abs()
        ulp = float(r_off.float().abs().max()) * 2 ** -8
        OUT["ab"][tag] = {
            "us_off": round(med["off"], 2), "us_on": round(med["on"], 2),
            "ratio": round(med["off"] / med["on"], 5),
            "delta_us_per_call": round(med["off"] - med["on"], 2),
            "aa_floor": round(st.median(off[:h]) / st.median(off[h:]), 5) if h else None,
            "calls_per_fold": census[key]["calls"],
            "fold_ms_if_it_holds": round(
                census[key]["calls"] * (med["off"] - med["on"]) / 1e3, 3),
            "parity": {"bit_exact": bool(d.max() == 0), "max_abs": float(d.max()),
                       "bf16_ulp_at_max": ulp,
                       "ulp_fraction": (float(d.max()) / ulp) if ulp else None,
                       "shape": list(r_off.shape)},
            "order": order,
        }
        dump()
        r = OUT["ab"][tag]
        print(f"  {tag:40s} off {r['us_off']:8.2f} us  on {r['us_on']:8.2f}  "
              f"ratio {r['ratio']:.5f}x  A/A {r['aa_floor']}  "
              f"fold {r['fold_ms_if_it_holds']:+.3f} ms", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
