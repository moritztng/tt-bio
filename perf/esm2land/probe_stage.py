#!/usr/bin/env python3
"""Which op in the split-fc1 rewrite loses bit-exactness, and by how much?

`probe_parity.py` reports one bool for the whole FFN. That is not enough to gate on: at L=128 the
split arm is not `torch.equal` and at 192/256 the row-blocked arm is not, while every L from 298 to
768 is, divisible by 32 or not. So the divisibility theory is dead and the real discriminator has
to come from the ops themselves.

The rewrite makes two independent substitutions. This probe checks each on its own, against the
same layer_norm output, and reports the differing-element count and max abs error at every stage:

  fc1     lin(xn, w1)[:, :d_ff] vs lin(xn, w1a)          -- does halving N change the matmul?
  gate    multiply(silu(x1), x2) vs multiply(x1, x2, SILU) -- does the fused activation change?
  out     the fc2 output of each                          -- and what survives to the FFN output

Also reports the same for the row-blocked L1 arm, whose chunk heights differ from the full-size M.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def stat(a, b):
    """Differing-element count, max abs error and worst relative error between two host tensors."""
    d = (a.float() - b.float()).abs()
    n = int((d > 0).sum())
    scale = b.float().abs().max().clamp(min=1e-12)
    return {"equal": bool(n == 0), "n_diff": n, "n_total": int(d.numel()),
            "frac_diff": round(n / d.numel(), 8), "max_abs": float(d.max()),
            "max_abs_over_peak": float(d.max() / scale)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[128, 192, 256, 288, 298, 320, 512])
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    CZ, FF = 256, 1024

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    SILU = [ttnn.UnaryOpType.SILU]
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cz": CZ, "d_ff": FF, "rows": a.rows, "sizes": {}}
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    for L in a.L:
        torch.manual_seed(0)
        nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
        w1_full = torch.randn(2 * FF, CZ) * 0.02
        w1, w1a, w1b = f(w1_full.t()), f(w1_full[:FF].t()), f(w1_full[FF:].t())
        w2 = f((torch.randn(CZ, FF) * 0.02).t())
        z = f(torch.randn(1, L, L, CZ))

        def lin(x, w, **kw):
            return ttnn.linear(x, w, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                               core_grid=T.CORE_GRID_MAIN, **kw)

        xn = ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)

        h = lin(xn, w1)
        x1, x2 = ttnn.chunk(h, 2, dim=-1)
        h1, h2 = lin(xn, w1a), lin(xn, w1b)
        t_x1, t_x2 = ttnn.to_torch(x1), ttnn.to_torch(x2)
        t_h1, t_h2 = ttnn.to_torch(h1), ttnn.to_torch(h2)

        g_ref = ttnn.multiply(ttnn.silu(x1), x2)
        g_a = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU)
        # the fused activation on its own, fed the reference's own fc1 halves
        g_fuse_only = ttnn.multiply(x1, x2, input_tensor_a_activations=SILU)
        out_ref, out_a = lin(g_ref, w2), lin(g_a, w2)

        e = {"L": L, "m_tiles_full": -(-L // 32), "L_mod_32": L % 32,
             "fc1_a": stat(t_h1, t_x1), "fc1_b": stat(t_h2, t_x2),
             "gate_fuse_only": stat(ttnn.to_torch(g_fuse_only), ttnn.to_torch(g_ref)),
             "gate_split": stat(ttnn.to_torch(g_a), ttnn.to_torch(g_ref)),
             "out_split": stat(ttnn.to_torch(out_a), ttnn.to_torch(out_ref))}

        # the row-blocked L1 arm, end to end against the same reference
        parts = ttnn.chunk(z, -(-L // a.rows), dim=1)
        e["chunk_heights"] = [int(p.shape[1]) for p in parts]
        outs = []
        for p in parts:
            pn = ttnn.layer_norm(p, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
            p1, p2 = lin(pn, w1a), lin(pn, w1b)
            gt = ttnn.multiply(p1, p2, input_tensor_a_activations=SILU,
                               memory_config=ttnn.L1_MEMORY_CONFIG)
            outs.append(lin(gt, w2))
        e["out_rowblock"] = stat(ttnn.to_torch(ttnn.concat(outs, dim=1)), ttnn.to_torch(out_ref))

        # uniform 32-row slices plus one short tail, instead of ttnn.chunk's even split
        sl, i = [], 0
        while i < L:
            hi = min(i + a.rows, L)
            sl.append(ttnn.slice(z, [0, i, 0, 0], [1, hi, L, CZ]))
            i = hi
        e["slice_heights"] = [int(t.shape[1]) for t in sl]
        souts = []
        for p in sl:
            pn = ttnn.layer_norm(p, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)
            p1, p2 = lin(pn, w1a), lin(pn, w1b)
            gt = ttnn.multiply(p1, p2, input_tensor_a_activations=SILU,
                               memory_config=ttnn.L1_MEMORY_CONFIG)
            souts.append(lin(gt, w2))
        e["out_slice32"] = stat(ttnn.to_torch(ttnn.concat(souts, dim=1)), ttnn.to_torch(out_ref))

        R["sizes"][str(L)] = e
        print(f"L={L:4d} %32={L%32:2d} heights={sorted(set(e['chunk_heights']))} "
              f"fc1a={e['fc1_a']['equal']} fuse={e['gate_fuse_only']['equal']} "
              f"split={e['gate_split']['equal']} out={e['out_split']['equal']} "
              f"rowblk={e['out_rowblock']['equal']} "
              f"| out max_abs={e['out_split']['max_abs']:.3e} "
              f"slice32={e['out_slice32']['equal']} "
              f"| out d={e['out_split']['max_abs_over_peak']:.2e} "
              f"rowblk d={e['out_rowblock']['max_abs_over_peak']:.2e} "
              f"slice32 heights={sorted(set(e['slice_heights']))}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
