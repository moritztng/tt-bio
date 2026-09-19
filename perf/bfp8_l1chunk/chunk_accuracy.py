#!/usr/bin/env python3
"""Accuracy of each chunk_decomp arm against an fp64 reference of the SAME operands.

Required whatever the speed says: k_chunk sets the online-softmax reduction order, so arm C is not
bit-exact with arm A even at one dtype, and a config that returns NaN or garbage is broken rather
than fast. The bias/scale convention is the one bfp8-sdpa-unlock calibrated by letting the bf16 arm
pick among all four orderings: softmax((qk + bias) * scale).
"""
from __future__ import annotations

import argparse, json, os, socket, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--rows", type=int, default=8, help="batch rows scored (fp64 ref is huge)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.triatt_sdpa as TS
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    S, H, D, B = a.seq, a.heads, a.head_dim, a.rows
    scale = D ** -0.5

    g = torch.Generator().manual_seed(0)
    q_t = torch.randn(B, H, S, D, generator=g)
    k_t = torch.randn(B, H, S, D, generator=g)
    v_t = torch.randn(B, H, S, D, generator=g)
    b_t = torch.randn(1, H, S, S, generator=g)

    ref = torch.nn.functional.softmax(
        (q_t.double() @ k_t.double().transpose(-1, -2) + b_t.double()) * scale, dim=-1
    ) @ v_t.double()

    def to(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def rel_rms(x):
        return float(((x.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())

    arms = {"A_bf16_k256": (ttnn.bfloat16, 512, 256),
            "B_bfp8_k256": (ttnn.bfloat8_b, 512, 256),
            "C_bfp8_k512": (ttnn.bfloat8_b, 512, 512)}
    res = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "shape": {"B": B, "H": H, "S": S, "D": D},
           "convention": "softmax((qk + bias) * scale)",
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(), "arms": {}}
    for name, (dt, qc, kc) in arms.items():
        o = TS.sdpa(to(q_t, dt), to(k_t, dt), to(v_t, dt), to(b_t, dt), scale, qc, kc)
        assert o is not None, f"{name} declined"
        x = ttnn.to_torch(o)
        res["arms"][name] = {
            "rel_rms_vs_fp64": round(rel_rms(x), 6),
            "finite": bool(torch.isfinite(x).all()),
            "max_abs": round(float(x.abs().max()), 6),
        }
        print(name, res["arms"][name])
        ttnn.deallocate(o)

    base = res["arms"]["A_bf16_k256"]["rel_rms_vs_fp64"]
    for name in arms:
        res["arms"][name]["vs_bf16_control"] = round(
            res["arms"][name]["rel_rms_vs_fp64"] / base, 4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["arms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
