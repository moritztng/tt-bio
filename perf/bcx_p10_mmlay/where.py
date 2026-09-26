#!/usr/bin/env python3
"""bcx-p10-mmlay leg 3 prep: which SOURCE LINE issues each plan-less matmul shape.

Leg 2 proved a core grid is worth 3-14x on the six matmul shape classes the round issues with
neither a program config nor a core grid. A shape key does not say where that call is written,
and a lever has to be written at a call site. So: patch `ttnn.matmul` for one warm block step,
and for every shape key record the innermost `tt_bio/` frame that issued it plus whether the
call carried a plan.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_stack import stack as S                # noqa: E402
from perf.bcx_p10_mmlay import mmkey as MK           # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_mmlay" / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="where_E.json")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)
    args.stacks, args.reps_cells, args.reps = "evo,extra", 1, 1

    import ttnn
    lv, dev, ref = S.open_all(args)

    seen = collections.defaultdict(collections.Counter)
    orig = ttnn.matmul

    def patched(*a, **k):
        try:
            pos = [x for x in a if isinstance(x, ttnn.Tensor)]
            dims = MK.mnk(pos[0], pos[1], bool(k.get("transpose_a")), bool(k.get("transpose_b")))
            flags = ("ta" if k.get("transpose_a") else "") + ("tb" if k.get("transpose_b") else "") or "-"
            plan = ("pc" if k.get("program_config") is not None else
                    "grid" if k.get("core_grid") is not None else "NONE")
            key = "%dx%dx%dx%d#%s#%s" % (*dims, flags, plan)
            site = "?"
            for fr in reversed(traceback.extract_stack()[:-1]):
                if "/tt_bio/" in fr.filename:
                    site = "%s:%d" % (fr.filename.split("/tt_bio/")[-1], fr.lineno)
                    break
            seen[key][site] += 1
        except Exception:                                                     # noqa: BLE001
            pass
        return orig(*a, **k)

    ttnn.matmul = patched
    spec = SH.CELLS["E"]
    raw_m, raw_z, _, _ = S.inputs(ref, SH.N_HOST, args.seed, ragged=True)
    m0, z0 = SH.pad_like_fold(raw_m, raw_z, 288)
    m0 = m0.repeat(spec["depth"], 1, 1)
    torch.manual_seed(args.seed + 1)
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    masks = SH.cell_masks(dev, 288, SH.N_REAL, spec["depth"], spec["masks"])
    SH.set_hifi(True)
    for stack in ("evo", "extra"):
        S.block_step(dev, lv, m0, z0, wm, wz, stack, k=2, ckpt=spec["ckpt"], masks=masks)
    ttnn.matmul = orig

    from tt_bio import mm_layout
    print("\nTT_BIO_MM_LAYOUT reach: on=%s side=%d %s"
          % (mm_layout.MM_LAYOUT, mm_layout.MM_LAYOUT_SIDE,
             json.dumps(mm_layout.reach(), sort_keys=True)))
    rows = sorted(seen.items(), key=lambda kv: -sum(kv[1].values()))
    print("%-34s %-6s %6s  %s" % ("batch x M x K x N # flags", "plan", "calls", "issued at"))
    for key, sites in rows:
        shape, flags, plan = key.split("#")
        print("%-28s %-5s %-6s %6d  %s"
              % (shape, flags, plan, sum(sites.values()),
                 ", ".join("%s x%d" % (s, n) for s, n in sites.most_common())))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps({k: dict(v) for k, v in seen.items()}, indent=1))
    print("\nwrote", OUT / args.out)


if __name__ == "__main__":
    main()
