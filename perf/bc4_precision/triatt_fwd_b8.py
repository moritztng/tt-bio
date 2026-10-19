#!/usr/bin/env python3
"""Triangle attention FORWARD on bf16 and bfp8 q/k/v, untaped and under the tape, against float64 on
the same operands (bfp8 ones dequantised): the head-major qkv projection and the SDPA route
`TriangleAttention` takes (`_tri_att_sdpa`), rel L2 per arm.

Written after the b8b + TT_BIO_TRIATT_QKV_B8 stack graded fwd_z 0.72 against 0.019 on the n=224
float64 VJP: a wrong value, not a precision loss, somewhere on the bfp8 q/k/v forward.

    TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python perf/bc4_precision/triatt_fwd_b8.py
"""
import contextlib, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TN, triatt_qkv as TQ, taped_ttnn as TT
    from tt_bio.autograd import precise_config
    dev = TN.get_device()
    up = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,  # noqa: E731
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    B, H, N, d, C = 224, 4, 224, 32, 128
    scale = d ** -0.5
    torch.manual_seed(0)
    ok = True
    rows = []
    for mode in ("untaped", "taped"):
        ctx = TT.tape() if mode == "taped" else contextlib.nullcontext()
        with ctx:
            # 1. head-major projection
            x = torch.randn(B, N, C)
            w = torch.randn(C, 3 * H * d) * C ** -0.5
            xd, wd = up(x, ttnn.bfloat16), up(w, ttnn.bfloat16)
            ref = (down(xd) @ down(wd)).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
            for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
                outs = TQ.qkv_heads(xd, wd, precise_config(), H, d, dt, TN._qkv_mm_config(xd, wd))
                if outs is None:
                    row = {"mode": mode, "op": "qkv_heads", "dtype": str(dt), "served": False}
                else:
                    outs = [getattr(o, "value", o) for o in outs]
                    row = {"mode": mode, "op": "qkv_heads", "dtype": str(dt), "served": True,
                           "rel": [rel(down(o), ref[s]) for s, o in enumerate(outs)],
                           "out": [str(o.dtype) for o in outs]}
                rows.append(row); print(json.dumps(row), flush=True)
            # 2. the SDPA route, all three operand sets
            q, k, v = (torch.randn(B, H, N, d) for _ in range(3))
            bias = torch.randn(1, H, N, N)
            for dt, bdt in ((ttnn.bfloat16,) * 2, (ttnn.bfloat8_b, ttnn.bfloat16), (ttnn.bfloat8_b,) * 2):
                qd, kd, vd, bd = up(q, dt), up(k, dt), up(v, dt), up(bias, bdt)
                q64, k64, v64, b64 = (down(t) for t in (qd, kd, vd, bd))
                o64 = torch.softmax((q64 @ k64.transpose(-1, -2) + b64) * scale, -1) @ v64
                o = TN._tri_att_sdpa(qd, kd, vd, bd, scale)
                o = getattr(o, "value", o)
                row = {"mode": mode, "op": "tri_att_sdpa", "dtype": str(dt), "bias": str(bdt),
                       "served": o is not None}
                if o is not None:
                    row.update(rel=rel(down(o)[:B, :H, :N, :d], o64), out=str(o.dtype))
                rows.append(row); print(json.dumps(row), flush=True)
    for mode in ("untaped", "taped"):
        for op in ("qkv_heads", "tri_att_sdpa"):
            r = [x for x in rows if x["mode"] == mode and x["op"] == op and x["served"]]
            if not r:
                continue
            base = max(r[0]["rel"]) if isinstance(r[0]["rel"], list) else r[0]["rel"]
            for x in r[1:]:
                worst = max(x["rel"]) if isinstance(x["rel"], list) else x["rel"]
                good = worst < max(3 * base, 2e-2)
                ok &= good
                print(mode, op, x["dtype"], x.get("bias", ""), "%.3e vs bf16 %.3e" % (worst, base),
                      "PASS" if good else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
