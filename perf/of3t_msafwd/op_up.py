#!/usr/bin/env python3
"""pair_transition op by op, upstream side: the float64 intermediates, and upstream 0.4.3 bf16's
own error at each op, each op teacher-forced from the float64 value of its input.

Ops: LayerNorm, linear_a, SiLU, linear_b, the gate multiply, linear_out, and the residual add
that lands the update on z. Writes the float64 intermediates for op_ours.py, so both sides are
scored against the same float64 tensors, one process per side.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from up_fwd import load_msa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("/home/ttuser/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--inter-out", type=Path, required=True, dest="inter_out")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    T = torch.load(a.trace, map_location="cpu", weights_only=False)
    B = torch.load(T["boundary"], map_location="cpu", weights_only=False)
    n = int(T["z_out"].shape[-2])
    tokm = torch.diagonal(B["inputs"]["kwargs"]["pair_mask"].reshape(n, n)) > 0
    rz = lambda x: x.reshape(n, n, -1)[tokm][:, tokm]
    rel = lambda x, r: float((rz(x.double()) - rz(r)).norm() / rz(r).norm())
    floor = lambda r: rel(r.to(torch.bfloat16), r)

    m64, m32 = load_msa(torch.float64, a.ckpt), load_msa(torch.float32, a.ckpt)
    inter, rep = {}, {"host": socket.gethostname(), "torch": torch.__version__, "blocks": {}}
    for i in range(len(m64.blocks)):
        t = T["sites"][f"b{i}.pair_transition"]
        nxt = (T["sites"][f"b{i+1}.opm"]["z"] if i + 1 < len(m64.blocks) else T["z_out"])
        p64, p32 = m64.blocks[i].pair_stack.pair_transition, m32.blocks[i].pair_stack.pair_transition
        x = t["z"]
        with torch.no_grad():
            ln = p64.layer_norm(x)
            pa = p64.swiglu.linear_a(ln)
            sa = p64.swiglu.swish(pa)
            pb = p64.swiglu.linear_b(ln)
            h = sa * pb
            out = p64.linear_out(h)
        check = rel(out, t["upd"])
        I = {"ln_w": p64.layer_norm.weight.detach().clone(),
             "ln_b": p64.layer_norm.bias.detach().clone(), "x": x, "ln": ln, "a": pa, "silu_a": sa, "b": pb, "h": h, "out": out,
             "z_after": nxt}
        inter[i] = I
        f32 = lambda v: v.to(torch.float32)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            u = {"layer_norm": (p32.layer_norm(f32(x)), ln),
                 "linear_a": (p32.swiglu.linear_a(f32(ln)), pa),
                 "silu": (p32.swiglu.swish(f32(pa).to(torch.bfloat16)), sa),
                 "linear_a+silu": (p32.swiglu.swish(p32.swiglu.linear_a(f32(ln))), sa),
                 "linear_b": (p32.swiglu.linear_b(f32(ln)), pb),
                 "gate_mul": (f32(sa).to(torch.bfloat16) * f32(pb).to(torch.bfloat16), h),
                 "linear_out": (p32.linear_out(f32(h)), out)}
            # Upstream's residual add: fp32 state plus the bf16 update autocast hands back.
            zsum = f32(x) + p32.linear_out(f32(h))
        row = {k: {"up_bf16": rel(v, r), "bf16_floor": floor(r), "dtype": str(v.dtype)}
               for k, (v, r) in u.items()}
        # The add's own error, apart from the update's: state + update in f64 on the same operands.
        row["residual_add"] = {"up_bf16": rel(zsum, x + out),
                               "bf16_floor": floor(x + out), "dtype": str(zsum.dtype),
                               "note": "vs float64 state + float64 update; includes the update's "
                                       "own error, which linear_out already reports"}
        rep["blocks"][i] = {"f64_decomposition_vs_trace_update": check, "ops": row}
        print(f"b{i}: f64 decomposition vs trace update {check:.3e}", flush=True)
        for k, r in row.items():
            print(f"   {k:14s} up bf16 {r['up_bf16']:.4e}  floor {r['bf16_floor']:.4e}  {r['dtype']}")
        if check > 1e-12:
            print("HARD FAILURE: the decomposition is not the module")
            return 4
    torch.save(inter, a.inter_out)
    a.report.write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
