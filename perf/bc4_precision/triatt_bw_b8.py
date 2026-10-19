#!/usr/bin/env python3
"""The fused triangle-attention backward (`triatt_bw.run`) on bf16 and bfp8 q/k/v, graded against
float64 autograd on the SAME operands (bfp8 ones dequantised): dq, dk, dv, dbias rel L2.

A correct kernel reads at the bf16 envelope on both dtypes. Written alongside lifting the kernel's
bf16-only gate, the a075fa138 lesson: a retyped CB is checked against float64 before any VJP grade.

    TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python perf/bc4_precision/triatt_bw_b8.py
"""
import json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import triatt_bw as TB
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    up = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,  # noqa: E731
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t, shape: torch.Tensor(ttnn.to_torch(t)).double()[tuple(slice(0, s) for s in shape)]  # noqa: E731
    rows = []
    for B, chunk in ((224, None), (32, None), (32, 2)):
        H, N, d = 4, 224, 32
        scale = d ** -0.5
        torch.manual_seed(0)
        q, k, v = (torch.randn(B, H, N, d) for _ in range(3))
        bias = torch.randn(1, H, N, N)
        g = (torch.randn(B, H, N, d) * 1e-2).bfloat16().float()
        for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
            qd, kd, vd = up(q, dt), up(k, dt), up(v, dt)
            bd, gd = up(bias, ttnn.bfloat16), up(g, ttnn.bfloat16)
            ok, why = TB.eligible(qd, kd, vd, bd)
            assert ok, why
            q64, k64, v64 = (down(t, (B, H, N, d)).requires_grad_() for t in (qd, kd, vd))
            b64 = down(bd, (1, H, N, N)).requires_grad_()
            o = torch.softmax(q64 @ k64.transpose(-1, -2) * scale + b64, -1) @ v64
            o.backward(g.double())
            dq, dk, dv, db = TB.run(dev, qd, kd, vd, bd, gd, scale, (ttnn.MathFidelity.HiFi4,),
                                    q_chunk_tiles=chunk)
            row = {"B": B, "q_chunk": chunk, "dtype": str(dt),
                   "dq": rel(down(dq, (B, H, N, d)), q64.grad), "dk": rel(down(dk, (B, H, N, d)), k64.grad),
                   "dv": rel(down(dv, (B, H, N, d)), v64.grad), "dbias": rel(down(db, (1, H, N, N)), b64.grad),
                   "out_dtypes": [str(t.dtype) for t in (dq, dk, dv)]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    worst = {dt: max(max(r[x] for x in ("dq", "dk", "dv", "dbias")) for r in rows if r["dtype"] == dt)
             for dt in {r["dtype"] for r in rows}}
    b16 = worst["DataType.BFLOAT16"]
    ok = all(w < max(3 * b16, 2e-2) for w in worst.values())
    print(f"worst rel L2 {worst}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
