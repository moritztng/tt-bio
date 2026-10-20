#!/usr/bin/env python3
"""bc4-bet-layout: where the transposed OPM layout (opm_layouts.py `pb`) leaves per_row's bits, and both against
float64. Stage-by-stage equality (mm1 values, relayouted operand, mm2 out) and rel L2 / max abs of each output vs
an einsum on the same bf16 inputs.
    python perf/bc4_layout/opm_pb_diag.py out.json [--ns 224,864]
"""
import argparse, json, os, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ns", default="224,864")
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    S, C, D, cz = 2, 32, 32, 128
    torch.manual_seed(0)
    W = (torch.randn(C, D, cz) * 0.05).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    th = lambda t: ttnn.to_torch(t).double()
    blt = torch.zeros(cz, S, D, S, C, dtype=W.dtype)
    for s in range(S):
        blt[:, s, :, s, :] = W.permute(2, 1, 0)
    w_row, w_t = up(W.reshape(C, D * cz)), up(blt.reshape(cz * S * D, S * C))
    mm = lambda x, y: ttnn.matmul(x, y, compute_kernel_config=cfg)
    res = {}
    for n in (int(x) for x in A.ns.split(",")):
        a0, b0 = torch.randn(S, n, C).to(torch.bfloat16), torch.randn(S, n, D).to(torch.bfloat16)
        a, b = up(a0), up(b0)
        I = J = n
        # per_row stages
        Ar = [mm(ttnn.reshape(a[s:s + 1], (I, C)), w_row) for s in range(S)]          # [I, D*cz] order (d,k)
        Arr = [ttnn.permute(ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (I, D, cz)),
                                           ttnn.TILE_LAYOUT), (0, 2, 1)) for x in Ar]   # [I, cz, D]
        Xr = ttnn.concat(Arr, dim=-1)
        btr = ttnn.concat([ttnn.permute(ttnn.reshape(b[s:s + 1], (J, D)), (1, 0)) for s in range(S)], dim=0)
        Or = mm(Xr, btr)
        # pb stages
        T = mm(w_t, ttnn.reshape(ttnn.permute(a, (0, 2, 1)), (S * C, I)))                 # [(k,s,d), I]
        Xp = ttnn.permute(ttnn.reshape(T, (cz, S * D, I)), (2, 0, 1))                     # [I, cz, S*D]
        btp = ttnn.reshape(ttnn.permute(b, (0, 2, 1)), (S * D, J))
        Op = mm(Xp, btp)
        # host views of the stages in one order
        mm1_r = torch.stack([th(x).reshape(I, D, cz) for x in Ar])                       # [S, I, D, cz]
        mm1_p = th(T).reshape(cz, S, D, I).permute(1, 3, 2, 0)                           # [S, I, D, cz]
        A64 = torch.einsum("sic,cdk->sidk", a0.double(), W.double())
        O64 = torch.einsum("sic,sjd,cdk->ikj", a0.double(), b0.double(), W.double())
        rel = lambda x, r: float((x - r).norm() / r.norm())
        row = {"mm1_equal": bool(torch.equal(mm1_r, mm1_p)), "mm1_max_abs": float((mm1_r - mm1_p).abs().max()),
               "mm1_rel_f64": {"per_row": rel(mm1_r, A64), "pb": rel(mm1_p, A64)},
               "operand_equal": bool(torch.equal(th(Xr), th(Xp))), "bt_equal": bool(torch.equal(th(btr), th(btp))),
               "out_equal": bool(torch.equal(th(Or), th(Op))), "out_max_abs": float((th(Or) - th(Op)).abs().max()),
               "out_rel_f64": {"per_row": rel(th(Or), O64), "pb": rel(th(Op), O64)},
               "out_frac_differing": float((th(Or) != th(Op)).double().mean())}
        # is the difference the bf16 rounding of mm1 (pb rounds once per output, per_row the same)? check pb mm1 vs
        # bf16(round(A64)): an exact fp32 accumulate then one rounding is what both should give
        r64 = A64.to(torch.bfloat16).double()
        row["mm1_frac_eq_rounded_f64"] = {"per_row": float((mm1_r == r64).double().mean()),
                                          "pb": float((mm1_p == r64).double().mean())}
        res[n] = row
        print(n, json.dumps(row), flush=True)
    pathlib.Path(A.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
