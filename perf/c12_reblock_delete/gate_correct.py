#!/usr/bin/env python3
"""Arm 1a correctness: does the fused gate epilogue compute `p * sigmoid(g)` into the right tiles?

Load-immune by construction -- it compares values, never time -- so it is admissible with a busy
sibling chip on the board pair. The N geometry is the production one (K=128, N=512, so 16 output
channel-tiles over 10 N-cores = 2 per core); only M is shrunk, because M is split across the other
axis and cannot change which (value, gate) tiles land together on a core.

Three arms on one x and one w:
  ref_mm   the shipped `minimal_matmul` in role-major column order, gated afterwards by the same
           `ttnn.multiply_(p, g, SIGMOID)` the four-way-split branch uses. This is today's answer.
  gate     the fused op: tile-interleaved weight, MM_GATE compute kernel, two destinations.
  torch    float32 on the host, the reference both are scored against.
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import ttnn
from tt_bio import mm_generic as G

TILE = 32
KERNEL_DIR = Path(__file__).resolve().parents[2] / "tt_bio" / "kernels" / "mm_split"
ROLES = ("p_a", "g_a", "p_b", "g_b")          # _GP_ROLES_SPLIT, the shipped order


def role_major_cols(C, group):
    """Column block order of the shipped fused projection: each role's `group` blocks of C."""
    return [(r, j) for r in ROLES for j in range(group)]


def tile_interleaved_cols(C, group):
    """(p t0, g t0, p t1, g t1, ...) per pair, which is what puts a (p, g) pair on ONE core."""
    out = []
    for p, g in (("p_a", "g_a"), ("p_b", "g_b")):
        for j in range(group):
            out.append((p, j))
            out.append((g, j))
    return out


def permute_w(w, src_order, dst_order, C):
    """Reorder `w`'s column blocks from `src_order` to `dst_order`. Pure permutation."""
    idx = {b: i for i, b in enumerate(src_order)}
    return torch.cat([w[:, idx[b] * C:(idx[b] + 1) * C] for b in dst_order], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=128, help="pair-rep side; M = h*h")
    ap.add_argument("--k", type=int, default=128, help="c_z, the contraction")
    ap.add_argument("--slice-c", type=int, default=128)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", 0)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    C = a.slice_c // a.group                      # 32 channels = 1 tile
    n_blocks = 4 * a.group                        # 16 column blocks in the fused projection
    N = n_blocks * C
    M = a.h * a.h
    assert N // TILE == n_blocks, (N, n_blocks)

    torch.manual_seed(0)
    x_t = (torch.randn(1, a.h, a.h, a.k) * 0.5).bfloat16()
    src = role_major_cols(C, a.group)
    dst = tile_interleaved_cols(C, a.group)
    w_major = (torch.randn(a.k, N) * 0.1).bfloat16()
    w_inter = permute_w(w_major, src, dst, C)

    # ---- host reference, float32 --------------------------------------------------------------
    acc = (x_t.float().reshape(M, a.k) @ w_major.float()).reshape(1, a.h, a.h, N)
    col = {b: i for i, b in enumerate(src)}

    def gather(role):
        return torch.cat([acc[..., col[(role, j)] * C:(col[(role, j)] + 1) * C]
                          for j in range(a.group)], dim=-1)

    ref_a = gather("p_a") * torch.sigmoid(gather("g_a"))
    ref_b = gather("p_b") * torch.sigmoid(gather("g_b"))

    # tt-bio's own opener, not ttnn.open_device: a LONE p300 chip is a CUSTOM cluster and
    # open_device is a TT_FATAL without a mesh graph descriptor. get_device sets it, takes the
    # device-init lock and honours TT_BIO_LEASE_CARDS.
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    res = {"shape": {"h": a.h, "k": a.k, "N": N, "M": M, "C": C, "group": a.group},
           "card": a.card}
    try:
        from tt_bio.tenstorrent import _MM_DEFAULT, COMPUTE_GRID_MAIN
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        cfg = (_MM_DEFAULT, tuple(COMPUTE_GRID_MAIN))
        res["cfg"] = {"block": list(_MM_DEFAULT), "grid": list(COMPUTE_GRID_MAIN)}

        def dev_t(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = dev_t(x_t)

        # ---- arm ref_mm: today's path, shipped op then the shipped gate --------------------
        wm = dev_t(w_major)
        fused = ttnn.experimental.minimal_matmul(
            x, wm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc)
        q = dict(zip(ROLES, ttnn.chunk(fused, chunks=4, dim=-1)))
        got_ref_a = ttnn.to_torch(ttnn.multiply(
            q["p_a"], q["g_a"], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])).float()
        got_ref_b = ttnn.to_torch(ttnn.multiply(
            q["p_b"], q["g_b"], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])).float()
        ttnn.deallocate(fused)

        # ---- arm gate: the fused epilogue ---------------------------------------------------
        wi = dev_t(w_inter)
        outs = [ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, a.h, a.h, a.slice_c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
            ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
        G.generic_minimal_matmul(
            dev, x, wi, outs, cfg, G.ckc_args(ckc),
            {"MM_DUAL_NOC": 1}, KERNEL_DIR, None, ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            [a.slice_c // TILE, a.slice_c // TILE], KERNEL_DIR, True)
        got_a = ttnn.to_torch(outs[0]).float()
        got_b = ttnn.to_torch(outs[1]).float()

        def score(name, got, ref):
            d = (got - ref).abs()
            den = ref.abs().mean().item()
            g, r = got.flatten(), ref.flatten()
            pcc = torch.corrcoef(torch.stack([g, r]))[0, 1].item()
            return {"arm": name, "max_abs": d.max().item(), "mean_abs": d.mean().item(),
                    "rel_mean": d.mean().item() / den if den else None, "pcc": pcc}

        res["scores"] = [
            score("ref_mm.a", got_ref_a, ref_a), score("ref_mm.b", got_ref_b, ref_b),
            score("gate.a", got_a, ref_a), score("gate.b", got_b, ref_b),
        ]
        res["gate_vs_refmm"] = [
            {"arm": "a", "max_abs": (got_a - got_ref_a).abs().max().item(),
             "bit_exact": bool(torch.equal(got_a, got_ref_a))},
            {"arm": "b", "max_abs": (got_b - got_ref_b).abs().max().item(),
             "bit_exact": bool(torch.equal(got_b, got_ref_b))},
        ]
    finally:
        ttnn.close_device(dev)

    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
