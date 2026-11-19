"""The token DiT's two gated residuals with the gate and the residual add in the output projection's epilogue
(tt_bio/mm_gres), against today's path: ttnn.linear then addcmul_at reading the gate window in place. BoltzGen's
token shape (514 tokens, c_s 768), the attention output projection on its padded head lanes (dit_qkvf, in both WH
sets: 16 heads x 64 lanes = K 1024; gres1-3 measured K 768, which the design never runs) and the transition's b_to_a
(K 1536), the
gate a column window of a stepped conditioning tensor as dit_cond_steps lays it out. Every arm against float64,
rel_l2 and max_abs, then ms a call (synchronized, median of --calls).

    TT_VISIBLE_DEVICES=N python perf/spd/bg_gres_check.py --mode {fast,normal} --out check.jsonl [--cfgs "2,4,4,2,4;..."]
"""
import argparse
import json
import math
import statistics
import time

import torch
import ttnn

# _DIT_STEP_MM's measured blocks first, then neighbours. K block 1 under fp32 accumulation (normal): wider K blocks
# return wrong elements on Wormhole (tenstorrent._DIT_STEP_MM).
CFGS = {"normal": "2,1,4,1,4;4,1,4,1,4;3,1,4,1,4;2,1,8,1,4", "fast": "2,4,4,2,4;4,4,4,2,4;2,8,4,2,4;3,4,4,1,4"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=514)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=16, help="token DiT heads: o_proj reads heads x ceil32(dim/heads) lanes")
    ap.add_argument("--steps", type=int, default=4, help="steps the stacked gate tensor holds (dit_cond_steps)")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--mode", choices=sorted(CFGS), default="fast")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pass-file", default="", help="written when the lever's own block (the mode's first cfg, the one "
                    "_DIT_STEP_MM serves) at the attention's o_proj shape is no less accurate than today's path "
                    "(rel_l2 <= 1.1x ref's) and >= --min-save-us faster a call; a grid cell chains the A/B on it")
    ap.add_argument("--min-save-us", type=float, default=10.0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio import tenstorrent as T, mm_gres as MGR
    from tt_bio.addcmul_at import addcmul_at
    dev = T.get_device()
    # the formats BoltzGen's token DiT runs (tenstorrent.dit_lowp): fast = dit_b8 on a bf16 DiT (bfp8 operands, HiFi2,
    # bf16 accumulation and stream), normal = dit_mm16 on an fp32 DiT (bf16 operands, fp32 accumulation and stream)
    normal = a.mode == "normal"
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4 if normal else ttnn.MathFidelity.HiFi2,
                                          math_approx_mode=False, fp32_dest_acc_en=normal, packer_l1_acc=normal)
    op_dt = ttnn.bfloat16 if normal else ttnn.bfloat8_b
    st_dt = ttnn.float32 if normal else ttnn.bfloat16
    cfgs = a.cfgs or CFGS[a.mode]
    grid = tuple(T.COMPUTE_GRID_MAIN)
    n, d, S = a.tokens, a.dim, a.steps
    gen = torch.Generator().manual_seed(0)
    out = open(a.out, "a")
    # the stepped conditioning tensor [S, n, 4 d] as dit_cond_steps stacks it; the gate window is step 2, column 2 d
    cond = torch.sigmoid(torch.randn(S, n, 4 * d, generator=gen)).to(torch.bfloat16)
    cd = ttnn.from_torch(cond, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    row, off = 2, 2 * d
    gate = cond[row, :, off:off + d]
    lanes = a.heads * -(-(d // a.heads) // 32) * 32
    for name, K in (("o_proj", lanes), ("b_to_a", 2 * d)):
        x = torch.randn(1, n, K, generator=gen)
        w = torch.randn(K, d, generator=gen) / math.sqrt(K)
        r = torch.randn(1, n, d, generator=gen) * 2
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=op_dt)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=op_dt)
        rd = ttnn.from_torch(r, layout=ttnn.TILE_LAYOUT, device=dev, dtype=st_dt)
        # float64 of the operands as the device holds them (block-fp8 rounding is the model's, not the op's)
        xq, wq, rq = (ttnn.to_torch(t).double().reshape(s) for t, s in ((xd, x.shape), (wd, w.shape), (rd, r.shape)))
        exact = rq + gate.double() * (xq @ wq)
        sc = exact.norm()
        rec = dict(mode=a.mode, shape=name, tokens=n, K=K, N=d, grid=grid, fidelity=str(ck.math_fidelity))

        def ref():
            p = (T.k1_linear(xd, wd, compute_kernel_config=ck, dtype=st_dt) if normal else
                 ttnn.linear(xd, wd, dtype=st_dt, compute_kernel_config=ck, core_grid=T.CORE_GRID_MAIN))
            o = addcmul_at(rd, p, cd, d, z_off=off, z_row=row)
            if o is None:   # the model's own fallback (DiffusionTransformerLayer.gate): cut the window, one addcmul
                g = ttnn.typecast(ttnn.reshape(ttnn.slice(cd, [row, 0, off], [row + 1, n, off + d]), (1, n, d)), st_dt)
                o = ttnn.addcmul(rd, g, p, value=1.0)
                ttnn.deallocate(g)
            ttnn.deallocate(p)
            return o
        fns = {"ref": ref}
        for c in filter(None, cfgs.split(";")):
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
        if name == "o_proj":
            lead = "fused_" + cfgs.split(";")[0].replace(",", "_")
            ok = (f"{lead}_ms" in rec and "ref_ms" in rec
                  and rec[f"{lead}_rel_l2"] <= 1.1 * rec["ref_rel_l2"]
                  and (rec["ref_ms"] - rec[f"{lead}_ms"]) * 1e3 >= a.min_save_us)
            verdict = dict(verdict="PASS" if ok else "FAIL", arm=lead, **{k: rec.get(k) for k in (
                f"{lead}_ms", "ref_ms", f"{lead}_rel_l2", "ref_rel_l2")})
            print(json.dumps(verdict), flush=True)
            out.write(json.dumps(verdict) + "\n")
            if ok and a.pass_file:
                open(a.pass_file, "w").write(json.dumps(verdict) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
