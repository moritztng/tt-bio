"""Which arm of each collapsed `ops.linear` call is right, graded against float64, inside a fold.

The 512 protenix-v2 A/B moved each CDK2 copy by 2.5-3.3 A, over its 0.6-1.3 A seed floor,
and the rows-view arm landed closer to 1HCL. A fold cannot say which call did it, so this runs
one fold on the served (rank) arm and, at every call the rows view would take, also computes
the view and a float64 product of the same bf16 operands, and records both errors.

    python3 perf/bcx_oplin/callcheck.py --model protenix-v2 --out perf/bcx_oplin/callcheck_pxv2_512.json
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.ops as ops  # noqa: E402
from common import Clock, arm  # noqa: E402
from fold_ab import boltz2_cfg  # noqa: E402

ACT = {None: lambda t: t, "relu": torch.relu, "silu": torch.nn.functional.silu,
       "sigmoid": torch.sigmoid, "gelu": torch.nn.functional.gelu}


def host(t):
    return ttnn.to_torch(t).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_baseline as B
    one_fold, meta, _ = B.build_fold(a.model, Path(tempfile.mkdtemp(prefix="oplin-cc-")),
                                     a.target, a.a3m, extra_cfg=boltz2_cfg(a.model, B))
    real_linear, real_via2d = ops.linear, ops._via2d
    seen = {}
    rows = {}

    def spy(x, fn, kw=None):
        seen["hit"] = (x, kw or {})
        return fn(x)

    def linear(x, w, bias=None, *, activation=None, **k):
        seen.clear()
        with arm(False):
            ops._via2d = spy
            try:
                y_rank = real_linear(x, w, bias, activation=activation, **k)
            finally:
                ops._via2d = real_via2d
        if "hit" not in seen:
            return y_rank
        s = [int(d) for d in x.shape]
        kw = seen["hit"][1]
        if (len(s) <= 2 or math.prod(s[:-2]) <= 1 or s[-2] % 32 or x.layout != ttnn.TILE_LAYOUT
                or x.is_sharded() or kw.get("program_config") is not None
                or kw.get("memory_config") is not None):
            return y_rank
        with arm(True):
            y_view = real_linear(x, w, bias, activation=activation, **k)
        act = activation if isinstance(activation, str) or activation is None else str(activation)
        key = (tuple(s), tuple(int(d) for d in w.shape), bias is not None, act)
        r = rows.setdefault(key, dict(x=s, w=list(key[1]), bias=key[2], activation=act,
                                      calls=0, rank_rel=[], view_rel=[], rank_vs_view_rel=[]))
        r["calls"] += 1
        yr, yv = host(y_rank), host(y_view)
        yv = yv.reshape(yr.shape)
        r["rank_vs_view_rel"].append((yr - yv).norm().item() / (yr.norm().item() or 1.0))
        if act in ACT:
            ref = host(x) @ host(w).reshape(s[-1], -1)
            if bias is not None:
                ref = ref + host(bias).reshape(-1)
            ref = ACT[act](ref).reshape(yr.shape)
            n = ref.norm().item() or 1.0
            r["rank_rel"].append((yr - ref).norm().item() / n)
            r["view_rel"].append((yv - ref).norm().item() / n)
        ttnn.deallocate(y_view)
        return y_rank

    ops.linear = linear
    meta["job_cfg"]["seed"] = 0
    with Clock() as clk:
        t, m = one_fold()
    ops.linear = real_linear
    out = []
    for r in rows.values():
        for f in ("rank_rel", "view_rel", "rank_vs_view_rel"):
            v = r.pop(f)
            r[f + "_max"] = max(v) if v else None
            r[f + "_median"] = sorted(v)[len(v) // 2] if v else None
        out.append(r)
    out.sort(key=lambda r: -(r["rank_rel_max"] or 0))
    res = dict(model=a.model, target=str(a.target), wall_s=round(t, 2), plddt=m.get("plddt"),
               aiclk=clk.stats(), calls=out)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
