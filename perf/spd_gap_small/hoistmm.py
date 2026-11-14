"""Device time and float64 error of the token DiT's three wide conditioning linears (dit_shoist / the cond hoist).

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/hoistmm.py --out hoistmm.jsonl

Boltz-2 runs [512, 768] x [768, 36864] three times a denoise step (scale with bias + sigmoid, shift, output gate with
bias + sigmoid): 56.6 MB of bf16 weight each. cen8 (WH, 1000 MHz) reads 1.43-1.98 ms a call, ~28 GB/s of weight
against ~248 GB/s of DRAM and a ~0.45 ms HiFi4 math floor. ttnn's own program for this shape is a 2D multicast in
which one row of cores pulls every weight column; the 1D arms multicast the 512-row activation instead and let every
core read its own weight columns. `fold` is the call exactly as tenstorrent.py makes it.

Device time is a captured trace replayed back to back (op_probe.dev_time), median of 7 batches, AICLK of the opened
node read while replays run. Error is against float64 with op_probe's wrong-pixel rule.
"""
import argparse, json, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import aiclk, dev_time, opened_nodes, wrong  # noqa: E402

M, K, N = 512, 768, 36864


def programs(grid, acc, act):
    """1D in0-multicast programs: all M rows on every core, N split over the cores (the four largest core counts
    that divide the N tiles and fit the grid, any rectangle)."""
    gx, gy = grid
    mt, kt, nt = M // 32, K // 32, N // 32
    cap = 4 if acc else 8
    out = []
    shapes = sorted({(cx, n // cx) for n in range(32, gx * gy + 1) if nt % n == 0
                     for cx in range(1, gx + 1) if n % cx == 0 and n // cx <= gy}, key=lambda c: -c[0] * c[1])
    for cg in [c for i, c in enumerate(shapes) if all(c[0] * c[1] != d[0] * d[1] for d in shapes[:i])][:4]:
        ncores = cg[0] * cg[1]
        pn = nt // ncores
        for ibw in (1, 2, 3, 4, 6):
            if kt % ibw:
                continue
            subs = sorted(((h, w) for h in (1, 2, 4) for w in range(1, pn + 1)
                           if mt % h == 0 and pn % w == 0 and h * w <= cap), key=lambda s: -s[0] * s[1])[:2]
            for h, w in subs:
                out.append((f"1d_c{ncores}_k{ibw}_pn{pn}_sb{h}x{w}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=cg, in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
                    out_block_h=mt, out_block_w=pn, per_core_M=mt, per_core_N=pn, fuse_batch=True,
                    fused_activation=act, mcast_in0=True)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--acc", default="1,0", help="fp32_dest_acc_en settings to sweep")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = T.get_device(trace="protenix")
    g = dev.compute_with_storage_grid_size(); grid = (g.x, g.y)
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=T.arch_name(), grid=grid, t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    emit(ev="start", m=M, k=K, n=N)
    torch.manual_seed(7)
    A = torch.randn(1, M, K); W = torch.randn(K, N) / K ** 0.5; Bv = torch.randn(N) * 0.1
    R = A.double() @ W.double() + Bv.double()
    Rs = torch.sigmoid(R)
    ta = ttnn.from_torch(A, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tw = ttnn.from_torch(W, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(Bv.reshape(1, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sig = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
    for acc in (bool(int(x)) for x in a.acc.split(",")):
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=True)
        arms = [("fold", None, False), ("fold_sig", None, True)]
        arms += [(lab, pc, False) for lab, pc in programs(grid, acc, None)]
        best = None
        for lab, pc, s in arms:
            kw = dict(bias=tb, compute_kernel_config=ckc)
            if pc is None:
                if s:
                    kw["activation"] = "sigmoid"
            else:
                kw["program_config"] = pc
            fn = lambda kw=kw: ttnn.linear(ta, tw, **kw)
            try:
                y = fn()
            except Exception as e:
                emit(ev="arm", acc=acc, arm=lab, error=str(e).splitlines()[0][:240])
                continue
            Y = ttnn.to_torch(y).double().reshape(R.shape); ttnn.deallocate(y)
            nb, mx, rel = wrong(Rs if s else R, Y)
            us, sp, ck = dev_time(dev, fn, nodes=nodes)
            gbs = K * N * 2 / us / 1e3
            emit(ev="arm", acc=acc, arm=lab, us=round(us, 1), spread=round(sp, 3), w_gbs=round(gbs, 1), wrong=nb,
                 max_err=round(mx, 5), rel_rms=float(f"{rel:.3e}"), aiclk=ck)
            if pc is not None and nb == 0 and (best is None or us < best[0]):
                best = (us, lab, pc)
        if best:
            # the best plain program again with the sigmoid fused, as the scale and gate linears need it
            us0, lab, pc = best
            pcs = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=pc.compute_with_storage_grid_size, in0_block_w=pc.in0_block_w,
                out_subblock_h=pc.out_subblock_h, out_subblock_w=pc.out_subblock_w, out_block_h=pc.out_block_h,
                out_block_w=pc.out_block_w, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N, fuse_batch=True,
                fused_activation=sig, mcast_in0=True)
            fn = lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc, program_config=pcs)
            y = fn(); Y = ttnn.to_torch(y).double().reshape(R.shape); ttnn.deallocate(y)
            nb, mx, rel = wrong(Rs, Y)
            us, sp, ck = dev_time(dev, fn, nodes=nodes)
            emit(ev="arm", acc=acc, arm=lab + "_sig", us=round(us, 1), spread=round(sp, 3), wrong=nb,
                 max_err=round(mx, 5), rel_rms=float(f"{rel:.3e}"), aiclk=ck)
    emit(ev="end")


if __name__ == "__main__":
    main()
