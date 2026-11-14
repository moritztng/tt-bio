#!/usr/bin/env python3
"""Row statistics of the pair a TriangleMultiplication call normalises, over one real fold.

A fused LN-in (raw x @ gamma*W, then rstd*(. - mu*colsum) + beta@W) is exact at HiFi4 but its error under a
truncated-operand matmul grows with |mean|/std of each pair row (perf/spd_trimul/lnfold_sim.py). This folds the
512 fixture once through tt_baseline.build_fold and samples every --every-th trimul input to host.

    python perf/spd_trimul/pair_stats.py --model esmfold2 --out stats.json
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="esmfold2", choices=("esmfold2", "esmfold2-fast"))
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--every", type=int, default=37)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_baseline as B
from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                         _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd: os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

rows, n = [], [0]
call = T.TriangleMultiplication.__call__
def probe(self, x, *args, **kw):
    if n[0] % a.every == 0:
        z = ttnn.to_torch(x).double()
        z = z.reshape(-1, z.shape[-1])
        mu, sd = z.mean(-1), z.std(-1, unbiased=False)
        r = (mu.abs() / sd.clamp_min(1e-12))
        q = torch.quantile(r[torch.randperm(r.numel())[:1 << 20]], torch.tensor([0.5, 0.9, 0.99], dtype=r.dtype))
        rows.append({"call": n[0], "shape": list(x.shape), "ratio_median": q[0].item(), "ratio_p90": q[1].item(),
                     "ratio_p99": q[2].item(), "ratio_max": r.max().item(), "sd_median": sd.median().item()})
        print(json.dumps(rows[-1]), flush=True)
    n[0] += 1
    return call(self, x, *args, **kw)
T.TriangleMultiplication.__call__ = probe

fix = ROOT / "perf" / "size512" / "fixtures"
one_fold = B.build_fold(a.model, ROOT / f".msa_ab_{a.model}_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
                        fix / f"cdk2x2_{a.size}.a3m")[0]
one_fold()
a.out.write_text(json.dumps({"model": a.model, "size": a.size, "calls": n[0], "samples": rows}, indent=1))
print(f"{n[0]} calls, worst p99 ratio {max(r['ratio_p99'] for r in rows):.2f}", flush=True)
