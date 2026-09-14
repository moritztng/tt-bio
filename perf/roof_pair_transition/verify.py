#!/usr/bin/env python3
"""What each Transition configuration costs in accuracy, against a float64 CPU reference.

Never against another approximation: the reference is the same bf16 inputs and weights promoted
to float64 and run through the same algebra on the host. Every device arm is scored against that,
so "bit-exact with the shipped arm" and "closer to the truth than the shipped arm" are separate
questions and both get answered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.device_lease import CardSetLease                                   # noqa: E402

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
H = W = 512
C = 128
HID = 512


def reference(x, w1, w2, w3, rows=32):
    """float64, host, row block at a time so the fp64 intermediates stay bounded."""
    out = torch.empty(1, H, W, C, dtype=torch.float64)
    for s in range(0, H, rows):
        c = x[:, s:s + rows].to(torch.float64)
        mu = c.mean(-1, keepdim=True)
        var = c.var(-1, unbiased=False, keepdim=True)
        xn = (c - mu) / torch.sqrt(var + 1e-5)
        a = xn @ w1.to(torch.float64)
        b = xn @ w2.to(torch.float64)
        out[:, s:s + rows] = (a * torch.sigmoid(a) * b) @ w3.to(torch.float64)
    return out


def score(got, ref):
    g = got.to(torch.float64)
    d = g - ref
    return {
        "rel_rms": float(torch.sqrt((d ** 2).mean()) / torch.sqrt((ref ** 2).mean())),
        "max_abs": float(d.abs().max()),
        "pcc": float(torch.corrcoef(torch.stack([g.flatten(), ref.flatten()]))[0, 1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "verify.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    xt = (torch.randn(1, H, W, C) * 0.5).to(torch.bfloat16)
    w1t = (torch.randn(C, HID) / C ** 0.5).to(torch.bfloat16)
    w2t = (torch.randn(C, HID) / C ** 0.5).to(torch.bfloat16)
    w3t = (torch.randn(HID, C) / HID ** 0.5).to(torch.bfloat16)
    ref = reference(xt, w1t, w2t, w3t)

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=0)
    try:
        kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
                else ttnn.types.BlackholeComputeKernelConfig)
        kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                  fp32_dest_acc_en=True, packer_l1_acc=True)
        cc = dev.compute_with_storage_grid_size()
        cg = ttnn.CoreGrid(y=min(10, cc.y), x=min(11, cc.x))

        def up(t, mc=DRAM):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        z, w1, w2, w3 = up(xt), up(w1t), up(w2t), up(w3t)
        nw = ttnn.from_torch(torch.ones(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        nb = ttnn.from_torch(torch.zeros(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)

        def run(h, fused):
            chunks = ttnn.chunk(z, -(-H // h), dim=1)
            parts = []
            for c in chunks:
                xn = ttnn.layer_norm(c, weight=nw, bias=nb, epsilon=1e-5,
                                     compute_kernel_config=kc, memory_config=L1)
                p = ttnn.linear(xn, w1, activation="silu" if fused else None,
                                compute_kernel_config=kc, memory_config=L1,
                                dtype=ttnn.bfloat16, core_grid=cg)
                if not fused:
                    p = ttnn.silu(p, memory_config=L1, output_tensor=p)
                q = ttnn.linear(xn, w2, compute_kernel_config=kc, memory_config=L1,
                                dtype=ttnn.bfloat16, core_grid=cg)
                ttnn.deallocate(xn)
                p = ttnn.multiply_(p, q)
                ttnn.deallocate(q)
                o = ttnn.linear(p, w3, compute_kernel_config=kc, memory_config=DRAM,
                                dtype=ttnn.bfloat16, core_grid=cg)
                ttnn.deallocate(p)
                parts.append(o)
                ttnn.deallocate(c)
            full = ttnn.concat(parts, dim=1)
            for p_ in parts:
                ttnn.deallocate(p_)
            t = ttnn.to_torch(full)
            ttnn.deallocate(full)
            return t

        arms = {"h16_fused_SHIPPED": (16, True), "h32_fused": (32, True),
                "h48_fused": (48, True), "h16_unfused": (16, False),
                "h32_unfused": (32, False), "h48_unfused": (48, False)}
        got, rows = {}, []
        for name, (h, f) in arms.items():
            try:
                got[name] = run(h, f)
            except Exception as e:                                            # noqa: BLE001
                rows.append({"arm": name, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"})
                continue
        base = got.get("h16_fused_SHIPPED")
        for name, t in got.items():
            r = {"arm": name}
            r.update(score(t, ref))
            r["bit_exact_vs_shipped"] = bool(torch.equal(t, base)) if base is not None else None
            if base is not None:
                d = (t.to(torch.float64) - base.to(torch.float64))
                r["rel_rms_vs_shipped"] = float(torch.sqrt((d ** 2).mean())
                                                / torch.sqrt((base.to(torch.float64) ** 2).mean()))
            rows.append(r)
    finally:
        ttnn.close_device(dev)
        lease.release()

    out = {"seed": a.seed, "ref": "float64 CPU, same bf16 inputs and weights", "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))
    for r in rows:
        if "error" in r:
            print("%-20s %s" % (r["arm"], r["error"]), flush=True)
        else:
            print("%-20s rel_rms %.3e  max_abs %.3e  pcc %.9f  bit-exact vs shipped %s  "
                  "rel_rms vs shipped %.3e" % (r["arm"], r["rel_rms"], r["max_abs"], r["pcc"],
                                               r["bit_exact_vs_shipped"],
                                               r["rel_rms_vs_shipped"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
