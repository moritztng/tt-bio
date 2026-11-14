#!/usr/bin/env python3
"""Per-call precision of `pair_mm` (minimal_matmul) against ttnn.linear, both against float64.

The WH float64 stack bisect (spd-bc2, .108) put PAIR_MM among the two levers that move BindCraft 2's
dL/dlogits away from float64 on Wormhole. This scores one call at a time, at every swept (K, N) block
key, forward and transpose_b, on the AF2 trunk's own kernel config, so the mechanism can be read off
the op rather than off a 52-block stack:

    TT_VISIBLE_DEVICES=5 python perf/spd/bc2_pairmm_precision.py --tokens 224 --out pm.json
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import af2, ops, pair_mm  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+", default=[128, 224])
ap.add_argument("--bias", action="store_true", help="also score a bias riding in the matmul")
ap.add_argument("--repeat", type=int, default=3, help="runs per arm; a spike that moves between runs is not arithmetic")
ap.add_argument("--no-l1acc", action="store_true", help="packer_l1_acc off in the trunk config")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = get_device()
ckc = af2.compute_kernel_config()
if a.no_l1acc:
    ckc.packer_l1_acc = False
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)
rows = []
torch.manual_seed(0)
for n in a.tokens:
    for kt, nt in sorted(pair_mm._BLOCK):
        K, N = 32 * kt, 32 * nt
        for tb in (False, True):
            x = torch.randn(1, n, n, K).bfloat16().double()
            w = (torch.randn(N, K) if tb else torch.randn(K, N)).div(K ** 0.5).bfloat16().double()
            b = torch.randn(N).bfloat16().double() if a.bias and not tb else None
            ref = x @ (w.T if tb else w) + (0 if b is None else b)
            xt, wt = up(x), up(w)
            bt = None if b is None else up(b.reshape(1, N))
            got = {}
            for arm in ("ttnn", "pair_mm"):
                pair_mm.PAIR_MM_FUSED = arm == "pair_mm"
                runs = []
                for _ in range(a.repeat):
                    before = pair_mm.STATS[0]
                    if tb:
                        out = (pair_mm.matmul(xt, wt, None, ckc, transpose_b=True) if arm == "pair_mm"
                               else ttnn.matmul(xt, wt, transpose_b=True, compute_kernel_config=ckc))
                    else:
                        out = ops.linear(xt, wt, bt, compute_kernel_config=ckc)
                    served = pair_mm.STATS[0] - before
                    if out is None:
                        break
                    runs.append(ttnn.to_torch(out).double().reshape(ref.shape))
                if not runs:
                    got[arm] = {"served": 0}
                    continue
                err = torch.stack(runs) - ref
                bad = err.abs() > 0.25          # 15x the worst rounding error seen; only corruption crosses it
                got[arm] = {"served": served,
                            "rel_l2": [float(e.norm() / ref.norm()) for e in err],
                            "max_abs": [float(e.abs().max()) for e in err],
                            "n_bad": [int(b.sum()) for b in bad],
                            "repeats_identical": all(torch.equal(r, runs[0]) for r in runs),
                            "bad_at": [[int(i) for i in idx] for idx in bad[0].nonzero()[:4]]}
            row = {"tokens": n, "K": K, "N": N, "transpose_b": tb, "bias": b is not None, **got}
            print(json.dumps(row), flush=True)
            rows.append(row)
pair_mm.PAIR_MM_FUSED = False
pathlib.Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "rows": rows}, indent=1))
