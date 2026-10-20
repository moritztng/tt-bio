"""Packer L1 accumulation in fp32 (Wormhole, fp32 dest, packer_l1_acc) against fp32 sequential sums, on known operands.

rowcol_probe found fc1's normal-mode wrong element ([2,447,2411], off by exactly -4.0) only with packer_l1_acc on, and
every K chunk's product right on its own: the fault is in adding K blocks' partials in L1. Here w is nb stacked 32 x 32
identities and in0_block_w is 1, so out = x[:, 0:32] + x[:, 32:64] + ... with one packer add per block after the first,
for ~1M operand tuples a call. Wrong sums are recorded with their operands.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/pl1_probe.py --out probe.jsonl
"""
import argparse
import json

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", default="1", help="comma list: K tiles a block (in0_block_w); >1 gives each block a long-mantissa dest sum")
    ap.add_argument("--nbs", default="2,3,4,8")
    ap.add_argument("--dists", default="wide,mixed,same_sign")
    ap.add_argument("--per-core", type=int, default=14, help="M tiles a core")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent as T
    T.set_fast_mode(False)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    out = open(a.out, "a")
    gen = torch.Generator().manual_seed(1)
    mt = g.x * g.y * a.per_core   # M tiles: divides over the grid (Galaxy 8 x 9) for both programs
    M = mt * 32
    for inner, nb, dist in [(i, n, d) for i in map(int, a.inner.split(",")) for n in map(int, a.nbs.split(","))
                            for d in a.dists.split(",")]:
        if True:
            x = torch.randn(M, 32 * nb * inner, generator=gen)
            if dist == "wide":     # exponents 2^-6 .. 2^6, both signs
                x = x * torch.pow(2.0, torch.randint(-6, 7, x.shape, generator=gen).float())
            elif dist == "same_sign":
                x = x.abs()
            x = x.to(torch.bfloat16)
            w = torch.eye(32).repeat(nb * inner, 1).to(torch.bfloat16)
            parts = x.float().view(M, nb, inner, 32).sum(2)   # a block's dest sum, fp32
            seq = parts[:, 0].clone()
            for i in range(1, nb):
                seq = seq + parts[:, i]   # fp32, in K order
            exact = x.double().view(M, nb * inner, 32).sum(1)
            xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            for pl1 in (True, False):
                ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True, packer_l1_acc=pl1)
                rec = dict(inner=inner, nb=nb, dist=dist, pl1=pl1, n=M * 32)
                o = None
                for name, pc in (
                        ("reuse", lambda: ttnn.MatmulMultiCoreReuseProgramConfig(
                            compute_with_storage_grid_size=(g.x, g.y), in0_block_w=inner, out_subblock_h=1, out_subblock_w=1,
                            per_core_M=mt // (g.x * g.y), per_core_N=1)),
                        ("mcast2d", lambda: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(1, g.y), in0_block_w=inner, out_subblock_h=1, out_subblock_w=1,
                            per_core_M=mt // g.y, per_core_N=1, transpose_mcast=False, fused_activation=None))):
                    try:
                        o = ttnn.to_torch(ttnn.matmul(xd, wd, program_config=pc(), compute_kernel_config=ck,
                                                      dtype=ttnn.float32)).float()
                        rec["prog"] = name
                        break
                    except RuntimeError as exc:
                        rec[f"{name}_error"] = str(exc).splitlines()[0][:160]
                if o is None:
                    print(json.dumps(rec), flush=True)
                    out.write(json.dumps(rec) + "\n")
                    continue
                e = (o.double() - seq.double())
                bad = e.abs() > 1e-3 * (exact.abs() + 1e-3)
                rec["n_bad"] = int(bad.sum())
                big = e.abs() > 0.25 + 0.01 * exact.abs()   # a wrap, not rounding
                rec["n_big"] = int(big.sum())
                if rec["n_big"]:
                    bad = big
                rec["max_rel_vs_seq"] = float((e.abs() / (seq.double().abs() + 1e-6)).max())
                rec["n_exact_vs_seq"] = int((e == 0).sum())
                idx = bad.nonzero()[:24].tolist()
                rec["bad"] = [dict(parts=[float(v) for v in parts[r, :, c]], seq=float(seq[r, c]), dev=float(o[r, c]),
                                   err=float(e[r, c])) for r, c in idx]
                print(json.dumps({k: v for k, v in rec.items() if k != "bad"}), rec["bad"][:2], flush=True)
                out.write(json.dumps(rec) + "\n")
                out.flush()
            ttnn.deallocate(xd), ttnn.deallocate(wd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
