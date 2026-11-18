"""The token DiT's two gated residuals with the gate and the residual add in the output projection's epilogue
(tt_bio/mm_gres), against today's path: ttnn.linear then addcmul_at reading the gate window in place. BoltzGen's
token shape (514 tokens, c_s 768), the attention output projection (K 768) and the transition's b_to_a (K 1536), the
gate a column window of a stepped conditioning tensor as dit_cond_steps lays it out. Every arm against float64,
rel_l2 and max_abs, then ms a call (synchronized, median of --calls).

    TT_VISIBLE_DEVICES=N python perf/spd/bg_gres_check.py --out check.jsonl [--cfgs "3,4,3,1,3;..."]
"""
import argparse
import json
import math
import statistics
import time

import torch
import ttnn

DEFAULT_CFGS = "3,4,3,1,3;3,4,3,3,1;2,4,3,2,1;3,8,3,1,3;5,4,3,1,3;3,4,6,1,3;2,8,4,2,2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=514)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4, help="steps the stacked gate tensor holds (dit_cond_steps)")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--cfgs", default=DEFAULT_CFGS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio import tenstorrent as T, mm_gres as MGR
    from tt_bio.addcmul_at import addcmul_at
    dev = T.get_device()
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    g = dev.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    n, d, S = a.tokens, a.dim, a.steps
    gen = torch.Generator().manual_seed(0)
    out = open(a.out, "a")
    # the stepped conditioning tensor [S, n, 4 d] as dit_cond_steps stacks it; the gate window is step 2, column 2 d
    cond = torch.sigmoid(torch.randn(S, n, 4 * d, generator=gen)).to(torch.bfloat16)
    cd = ttnn.from_torch(cond, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    row, off = 2, 2 * d
    gate = cond[row, :, off:off + d]
    for name, K in (("o_proj", d), ("b_to_a", 2 * d)):
        x = torch.randn(1, n, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, d, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        r = (torch.randn(1, n, d, generator=gen) * 2).to(torch.bfloat16)
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        rd = ttnn.from_torch(r, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        exact = r.double() + gate.double() * (x.double() @ w.double())
        sc = exact.norm()
        rec = dict(shape=name, tokens=n, K=K, N=d, grid=grid, fidelity="HiFi4")

        def ref():
            p = ttnn.linear(xd, wd, compute_kernel_config=ck, core_grid=T.CORE_GRID_MAIN)
            o = addcmul_at(rd, p, cd, d, z_off=off, z_row=row)
            ttnn.deallocate(p)
            return o
        fns = {"ref": ref}
        for c in filter(None, a.cfgs.split(";")):
            bc = tuple(int(v) for v in c.split(","))
            fns["fused_" + "_".join(map(str, bc))] = (
                lambda bc=bc: MGR.matmul_gate_add(xd, wd, None, (cd, off, d, row), rd, bc, grid, ck))
        for arm, fn in list(fns.items()):
            try:
                o = fn()
                if o is None:
                    rec[f"{arm}_error"] = "declined"
                    del fns[arm]
                    continue
                o = ttnn.to_torch(o).double().reshape(exact.shape)
            except RuntimeError as exc:   # a block that does not place
                rec[f"{arm}_error"] = str(exc).splitlines()[0][:200]
                del fns[arm]
                continue
            e = o - exact
            rec[f"{arm}_rel_l2"] = float(e.norm() / sc)
            rec[f"{arm}_max_abs"] = float(e.abs().max())
        for arm, fn in fns.items():
            for _ in range(3):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(a.calls):
                t0 = time.perf_counter()
                o = fn()
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
            rec[f"{arm}_ms"] = round(statistics.median(ts), 4)
        try:
            rec["aiclk"] = open(f"/sys/class/tenstorrent/tenstorrent!{T.visible_device_ids()[0]}/tt_aiclk").read().strip()
        except Exception:   # the clock is recorded by run_cell.sh's sampler too
            pass
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
