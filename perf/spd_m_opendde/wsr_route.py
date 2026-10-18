"""The pair Transition as the Pairformer calls it (`add_to_input=True`, the residual-L1 hint), per lever set and
token count, against a float32 reference z + T(z). Finds where a lever set's route breaks with size.

rel = rms(out - ref) / rms(T(z)); any bf16 output reads ~0.11 at |z| ~ 40 (one rounding of z + t).
Usage (one chip; TT_BIO_LEVERS must hold every lever an arm names, silu_f32 compiles at device open):
    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=<superset> [WSR_C=384 WSR_X=4] python wsr_route.py N[,N...] ARM=<levers> ...
WSR_C is the pair width, WSR_X the hidden expansion (OpenDDE's trunk 384 x 4; its template stack 64 x 2).
"""
import os
import sys
import torch
import ttnn
import tt_bio.tenstorrent as T

torch.set_num_threads(8)
sizes = [int(v) for v in sys.argv[1].split(",")]
arms = [a.split("=", 1) for a in sys.argv[2:]]
c, xh, zs = int(os.environ.get("WSR_C", 384)), int(os.environ.get("WSR_X", 4)), 40.0
g = torch.Generator().manual_seed(0)
sd = {"norm.weight": 1 + 0.1 * torch.randn(c, generator=g), "norm.bias": 0.1 * torch.randn(c, generator=g),
      "fc1.weight": torch.randn(xh * c, c, generator=g) * c ** -0.5,
      "fc2.weight": torch.randn(xh * c, c, generator=g) * c ** -0.5,
      "fc3.weight": torch.randn(c, xh * c, generator=g) * (xh * c) ** -0.5}
sd = {k: v.bfloat16().float() for k, v in sd.items()}
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def ref_rows(z):
    """float32 z + T(z), 32 rows at a time."""
    out = torch.empty_like(z)
    for i in range(0, z.shape[1], 32):
        x = z[:, i:i + 32]
        xn = torch.nn.functional.layer_norm(x, (c,), sd["norm.weight"], sd["norm.bias"], 1e-5)
        t = (torch.nn.functional.silu(xn @ sd["fc1.weight"].t()) * (xn @ sd["fc2.weight"].t())) @ sd["fc3.weight"].t()
        out[:, i:i + 32] = t
    return out


for n in sizes:
    z = (zs * torch.randn(1, n, n, c, generator=g)).bfloat16().float()
    t_ref = ref_rows(z)
    ref = z + t_ref
    trms = t_ref.pow(2).mean().sqrt()
    for name, spec in arms:
        with T.levers(spec):
            mod = T.Transition(sd, ckc)
            zt = ttnn.from_torch(z.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            mc = T._residual_update_memory_config(zt.shape, zt.dtype)
            out = mod(zt, memory_config=mc, add_to_input=True)
            o = ttnn.to_torch(out).float()
        e = o - ref
        bad = (e.abs() > 4).reshape(n, n, c).any(-1)
        rows = bad.any(1).nonzero().flatten().tolist()
        cols = bad.any(0).nonzero().flatten().tolist()
        print(f"c {c}x{xh} n {n} {name}: rel_rms {(e.pow(2).mean().sqrt() / trms).item():.5f} max|e| {e.abs().max().item():.3f} "
              f"bad (i,j) {int(bad.sum())} rows {rows[:3]}..{rows[-3:]} ({len(rows)}) cols {cols[:3]}..{cols[-3:]} "
              f"({len(cols)}) same_buffer {out.buffer_address() == zt.buffer_address() if out.is_allocated() else '-'}",
              flush=True)
        for t in {id(out): out, id(zt): zt}.values():
            if t.is_allocated():
                ttnn.deallocate(t)
