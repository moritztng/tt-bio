#!/usr/bin/env python3
"""Does `TT_BIO_MM_SHORT_M_BW` FIRE on a model other than Boltz-2, and on which shapes?

`MM_SHORT_M_STATS` alone cannot answer it. `_short_m_proj_config` returns before it counts
whenever an operand is not bf16 or the weight is not 2-D, so 0/0 is ambiguous between "never
called" and "called and rejected early". This wraps the function instead, so every call is
recorded with its operand shapes and dtypes and whether a program config was served.

    fire_probe.py --model rf3 --data examples/prot.yaml --out <json> [-- <extra predict args>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio.tenstorrent as T   # noqa: E402
from tt_bio.main import predict   # noqa: E402

SEEN: dict[tuple, list[int]] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bw", type=int, default=1)
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()

    T._MM_SHORT_M_BW = bool(a.bw)
    orig = T._short_m_proj_config

    def wrapped(x, w):
        pc = orig(x, w)
        key = (tuple(int(v) for v in x.shape), tuple(int(v) for v in w.shape),
               str(x.dtype), str(w.dtype))
        e = SEEN.setdefault(key, [0, 0])
        e[0] += 1
        e[1] += int(pc is not None)
        return pc

    T._short_m_proj_config = wrapped

    argv = [a.data, "--model", a.model, "--out_dir", a.out_dir] + list(a.rest)
    t0 = time.perf_counter()
    rc = 0
    try:
        predict.main(argv, standalone_mode=False)
    except SystemExit as e:                                             # noqa: PERF203
        rc = int(e.code or 0)
    dt = time.perf_counter() - t0

    rows = [{"x": list(k[0]), "w": list(k[1]), "x_dtype": k[2], "w_dtype": k[3],
             "calls": v[0], "served": v[1]} for k, v in SEEN.items()]
    rows.sort(key=lambda r: -r["calls"])
    out = {"model": a.model, "data": a.data, "bw_flag": bool(a.bw), "pid": os.getpid(),
           "wall_s": round(dt, 2), "rc": rc,
           "counter_taken": int(T.MM_SHORT_M_STATS[0]),
           "counter_declined": int(T.MM_SHORT_M_STATS[1]),
           "wrapped_calls": sum(r["calls"] for r in rows),
           "wrapped_served": sum(r["served"] for r in rows),
           "groups": rows}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1))
    for r in rows[:20]:
        print("  x=%s w=%s %s/%s  calls %d  served %d"
              % (r["x"], r["w"], r["x_dtype"], r["w_dtype"], r["calls"], r["served"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
