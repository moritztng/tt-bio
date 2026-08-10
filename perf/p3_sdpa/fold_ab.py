#!/usr/bin/env python3
"""p3-sdpa deliverable 1+2: one full 298 aa protenix-v2 fold under one SDPA chunk setting.

Runs the production path (`scripts/gpu_vs_tt/tt_baseline.build_fold` -> `predict_one`), so the
number is a real fold wall and the CIF the fold writes is a real output structure. One arm per
process, because `SDPA_TRI_MID_CHUNK` is read at model construction and the program-config cache
is per-process.

Stage boundaries (`pf_stack`, `trunk_msa`, `trunk_template`, trunk, diffusion) come from
`perf/stage_split_298/stage_split.py`, which syncs the device on BOTH sides of every timed region.

Also saves the end-of-trunk `(s_trunk, z_trunk)` of the LAST cycle of the LAST fold, so the two
arms can be compared as tensors as well as as structures.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-sdpa \
      python3 perf/p3_sdpa/fold_ab.py --mid-chunk 64 --repeat 2 --out perf/p3_sdpa/fold_c64.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid-chunk", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--target", type=Path, default=REPO_ROOT / "examples/prot300.yaml")
    ap.add_argument("--msa-a3m", type=Path,
                    default=REPO_ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=None)
    args = ap.parse_args()
    art = args.artifacts or args.out.with_suffix("")
    art.mkdir(parents=True, exist_ok=True)

    import torch
    SS = _load("stage_split", "perf/stage_split_298/stage_split.py")

    # Set the chunk BEFORE the model is built: TriangleAttention reads it in __init__.
    import tt_bio.tenstorrent as T
    T.SDPA_TRI_MID_CHUNK = args.mid_chunk

    import tt_bio.protenix as P
    SS.install_patches()
    T.Pairformer.__call__ = SS.timed("pf_stack", T.Pairformer.__call__)
    P.Trunk._msa = SS.timed("trunk_msa", P.Trunk._msa)
    P.Trunk._template = SS.timed("trunk_template", P.Trunk._template)

    # Count the SDPA calls this fold really issues, per (shape, chunk), rather than inheriting.
    import ttnn
    calls: dict[str, int] = {}
    _orig_sdpa = ttnn.transformer.scaled_dot_product_attention

    def _counting_sdpa(q, k, v, **kw):
        m = kw.get("attn_mask")
        key = (f"q{tuple(q.shape)}"
               + (f"+mask{tuple(m.shape)}" if m is not None else "+nomask"))
        calls[key] = calls.get(key, 0) + 1
        return _orig_sdpa(q, k, v, **kw)

    ttnn.transformer.scaled_dot_product_attention = _counting_sdpa

    # End-of-trunk tap: keep the last cycle's (s_trunk, z_trunk) of the last fold.
    trunk_out = {}
    _orig_trunk = P.Trunk.__call__

    def _tapped_trunk(self, *a, **k):
        s, z = _orig_trunk(self, *a, **k)
        trunk_out["s"] = ttnn.to_torch(s).float().clone()
        trunk_out["z"] = ttnn.to_torch(z).float().clone()
        return s, z

    P.Trunk.__call__ = _tapped_trunk

    tb = _load("tt_baseline", "scripts/gpu_vs_tt/tt_baseline.py")
    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    one_fold, meta, _state = tb.build_fold("protenix-v2", msa_dir, args.target, args.msa_a3m)

    folds = []
    for i in range(args.repeat + 1):          # fold 0 is cold (kernel compile), never reported
        SS.TOP.clear(); SS.GROSS.clear(); SS.DEPTH[0] = 0
        calls.clear()
        total, metrics = one_fold()
        folds.append(dict(idx=i, cold=(i == 0), total_s=round(total, 4),
                          plddt=float(metrics.get("plddt", float("nan"))),
                          stages={n: [c, round(t, 4)] for n, (c, t) in SS.TOP.items()},
                          gross={n: [c, round(t, 4)] for n, (c, t) in SS.GROSS.items()},
                          sdpa_calls=dict(calls)))
        print(f"fold {i}{' (cold)' if i == 0 else ''}: {total:.3f}s  "
              f"plddt={metrics.get('plddt')}", flush=True)

    # The structure the last warm fold wrote.
    struct_dir = Path(meta["struct_dir"])
    for f in sorted(struct_dir.glob("*")):
        shutil.copy2(f, art / f.name)
    torch.save({k: v for k, v in trunk_out.items()}, art / "trunk_out.pt")

    warm = folds[1:]
    out = dict(mid_chunk=args.mid_chunk, host=os.uname().nodename,
               visible=os.environ.get("TT_VISIBLE_DEVICES"),
               loadavg=os.getloadavg(), meta={k: v for k, v in meta.items() if k != "job_cfg"},
               folds=folds,
               warm_total_s=[f["total_s"] for f in warm],
               warm_median_s=sorted(f["total_s"] for f in warm)[len(warm) // 2],
               artifacts=str(art))
    args.out.write_text(json.dumps(out, indent=2, default=str))
    w = warm[-1]
    print(f"\n=== mid_chunk={args.mid_chunk} WARM last: {w['total_s']}s ===")
    for n, (c, t) in sorted(w["gross"].items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:18s} n={c:<5d} {t:8.3f}s")
    print("  sdpa calls:", w["sdpa_calls"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
