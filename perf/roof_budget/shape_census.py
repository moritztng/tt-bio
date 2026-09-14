#!/usr/bin/env python3
"""Every matmul the 512 aa fold issues, by (batch, M, K, N), weighted by calls per fold.

The arithmetic floor cannot be priced at the 8192-cube rate: no matmul in this fold is an 8192
cube, and the rate a shape reaches on this part spans 2.85 to 100.4 TFLOP/s. So enumerate the
shapes first, then measure the ones that carry the FLOPs.

Shapes come from the three disjoint top-level captures, so no matmul is counted twice.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exec_flops as EF                                                       # noqa: E402

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]


def shapes_of(nodes):
    """[(batch, M, K, N, flops)] for every matmul-kind op in a capture."""
    ops, ins, outs = EF.operands(nodes)
    out = []
    for i, op in enumerate(ops):
        name = op["name"]
        if name in EF.FREE:
            continue
        shapes = [s for _, s in ins[i]]
        if name in EF.MATMUL:
            o = outs[i]
            if not o:
                continue
            sh = max(o, key=EF._prod)
            if len(sh) < 2:
                continue
            M, N = sh[-2], sh[-1]
            batch = EF._prod(sh[:-2])
            acts = [s for s in shapes if len(s) >= 2 and EF._prod(s[:-1]) == batch * M]
            if acts:
                K = max(s[-1] for s in acts)
            else:
                w = [s for s in shapes if len(s) == 2 and s[-1] == N]
                if not w:
                    continue
                K = max(s[0] for s in w)
            out.append((batch, M, K, N, 2 * batch * M * N * K))
        elif name == "ttnn.generic_op":
            w2 = [s for s in shapes if len(s) == 2]
            if w2:
                for K, N in w2:
                    cand = [EF._prod(s) for s in shapes if s[-1] == K and len(s) >= 2]
                    if not cand:
                        continue
                    rows = max(cand) // K
                    out.append((1, rows, K, N, 2 * rows * K * N))
            else:
                mask = [s for s in shapes if len(s) == 4 and s[0] == 1 and s[-1] == s[-2]]
                qkv = [s for s in shapes if len(s) == 4 and s[0] != 1 and s[-1] != s[-2]]
                if mask and qkv:
                    b, h, S, d = max(qkv, key=EF._prod)
                    # QK^T is (S x d) @ (d x S) and AV is (S x S) @ (S x d), b*h of each
                    out.append((b * h, S, d, S, 2 * b * h * S * S * d))
                    out.append((b * h, S, S, d, 2 * b * h * S * S * d))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "attrib2_512_tip_qb2c2.json")
    ap.add_argument("--captures", type=Path, default=HERE / "captures")
    ap.add_argument("--out", type=Path, default=HERE / "shape_census.json")
    a = ap.parse_args()
    sigs = json.loads(a.run.read_text())["attrib"]["sigs"]

    agg = defaultdict(lambda: [0, 0])          # (b,M,K,N) -> [calls per fold, flops per fold]
    for cap in sorted(glob.glob(str(a.captures / "cap_*.json.gz"))):
        sig = Path(cap).name[len("cap_"):-len(".json.gz")].replace("__", "|")
        if sig not in TOP:
            continue
        calls = sigs[sig]["calls"]
        for b, M, K, N, f in shapes_of(EF.nodes_of(cap)):
            e = agg[(b, M, K, N)]
            e[0] += calls
            e[1] += calls * f

    tot = sum(v[1] for v in agg.values())
    rows = [{"batch": b, "M": M, "K": K, "N": N, "calls_per_fold": c,
             "TFLOP_per_fold": f / 1e12, "pct": 100 * f / tot}
            for (b, M, K, N), (c, f) in agg.items()]
    rows.sort(key=lambda r: -r["TFLOP_per_fold"])
    a.out.write_text(json.dumps({"total_TFLOP": tot / 1e12, "rows": rows}, indent=1))
    print("total matmul %.3f TFLOP over %d distinct shapes" % (tot / 1e12, len(rows)))
    cum = 0.0
    for r in rows[:26]:
        cum += r["pct"]
        print("  b=%-5d M=%-7d K=%-5d N=%-6d  calls %6d  %8.3f TF  %5.2f %%  cum %5.1f %%"
              % (r["batch"], r["M"], r["K"], r["N"], r["calls_per_fold"], r["TFLOP_per_fold"],
                 r["pct"], cum))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
