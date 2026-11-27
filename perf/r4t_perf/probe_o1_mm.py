#!/usr/bin/env python3
"""30-second chip-9 probe:
1. Exact per-op breakdown of 1 iter of newton_schulz_device on (14, 6144, 768) and (36, 768, 768),
   plus ns_mode='fp32' speed AND accuracy vs float64 NS on rank-2+noise realistic gradients!
2. P3 ttnn.rms_norm + HiFi4 proj accuracy vs float64 and wall ms.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.rfd4 import engine as E, network as N  # noqa: E402
from tt_bio.rfd4.pair import PairSplit  # noqa: E402
from tt_bio.train.device_optim import newton_schulz_device, _split2_bf16, _sum_sq_last2_exact_f32  # noqa: E402
from tt_bio.train.rfd4_data import load_batch  # noqa: E402
from tt_bio.train.rfd4_pair import DeviceTrainPair, _split3_bf16  # noqa: E402


def _ckc(fid):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _rel(a, b):
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a64 - b64) / (np.linalg.norm(b64) + 1e-30))


def _ns_f64(x: np.ndarray) -> np.ndarray:
    a, b, c = 3.4445, -4.775, 2.0315
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        cur = x[i].astype(np.float64)
        fl = cur.shape[0] > cur.shape[1]
        if fl:
            cur = cur.T
        cur = cur / max(float(np.linalg.norm(cur)), 1e-7)
        for _ in range(5):
            g = cur @ cur.T
            p = b * g + c * (g @ g)
            cur = a * cur + p @ cur
        out[i] = cur.T if fl else cur
    return out


def bench(dev, fn, reps=5):
    y = fn()
    ttnn.synchronize_device(dev)
    if isinstance(y, (list, tuple)):
        for t in y:
            if t is not None: ttnn.deallocate(t)
    elif y is not None:
        ttnn.deallocate(y)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = fn()
        ttnn.synchronize_device(dev)
        ts.append(1000 * (time.perf_counter() - t0))
        if isinstance(y, (list, tuple)):
            for t in y:
                if t is not None: ttnn.deallocate(t)
        elif y is not None:
            ttnn.deallocate(y)
    return float(np.median(ts))


def main():
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    try:
        c4 = _ckc(ttnn.MathFidelity.HiFi4)
        c2 = _ckc(ttnn.MathFidelity.HiFi2)
        rng = np.random.default_rng(7)
        # Near-rank-2 + dense noise (real RFD4 gradient stress test!)
        x_np = np.zeros((14, 6144, 768), dtype=np.float32)
        for i in range(14):
            u = rng.standard_normal((6144, 2)).astype(np.float32) * 0.05
            v = rng.standard_normal((2, 768)).astype(np.float32) * 0.05
            x_np[i] = u @ v + rng.standard_normal((6144, 768)).astype(np.float32) * 0.005
        ref_ns = _ns_f64(x_np)
        xt32 = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        # Time + grade full NS modes on (14, 6144, 768)
        for mode, tall in [("split2_fast", True), ("fp32", True)]:
            y = newton_schulz_device(xt32, mode=mode, tall=tall)
            ttnn.synchronize_device(dev)
            rel = _rel(ttnn.to_torch(y).float().numpy(), ref_ns)
            ttnn.deallocate(y)
            ms = bench(dev, lambda: newton_schulz_device(xt32, mode=mode, tall=tall), reps=3)
            print(f"[NS (14,6144,768) {mode:12s}] {ms:6.2f} ms | rel_vs_f64={rel:.3e}", flush=True)

        # Per-op breakdown of ONE iter of split2_fast on (14, 6144, 768)
        print(f"  _sum_sq_last2 (14,6144,768): {bench(dev, lambda: _sum_sq_last2_exact_f32(xt32)):.2f} ms", flush=True)
        print(f"  _split2_bf16  (14,6144,768): {bench(dev, lambda: _split2_bf16(xt32)):.2f} ms", flush=True)
        xh, xl = _split2_bf16(xt32)
        print(f"  G  p0=xh^T@xh HiFi2: {bench(dev, lambda: ttnn.matmul(xh, xh, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  G  p1=xh^T@xl HiFi2: {bench(dev, lambda: ttnn.matmul(xh, xl, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        p1 = ttnn.matmul(xh, xl, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=c2)
        print(f"  p1 transpose (14,768,768) fp32: {bench(dev, lambda: ttnn.transpose(p1, -2, -1)):.2f} ms", flush=True)
        print(f"  G  split2_bf16 (14,768,768): {bench(dev, lambda: _split2_bf16(p1)):.2f} ms", flush=True)
        gh, gl = _split2_bf16(p1)
        print(f"  GG p0=gh@gh   HiFi2: {bench(dev, lambda: ttnn.matmul(gh, gh, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  XP p0=xh@gh   HiFi2: {bench(dev, lambda: ttnn.matmul(xh, gh, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  XP p1=xh@gl   HiFi2: {bench(dev, lambda: ttnn.matmul(xh, gl, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  XP p2=xl@gh   HiFi2: {bench(dev, lambda: ttnn.matmul(xl, gh, dtype=ttnn.float32, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  add(xt32, xt32) fp32: {bench(dev, lambda: ttnn.add(xt32, xt32)):.2f} ms", flush=True)
        print(f"  mul(xt32, 3.44) fp32: {bench(dev, lambda: ttnn.multiply(xt32, 3.4445)):.2f} ms", flush=True)
        # Check direct fp32 matmul!
        print(f"  fp32 G=X^T@X  HiFi4: {bench(dev, lambda: ttnn.matmul(xt32, xt32, transpose_a=True, dtype=ttnn.float32, compute_kernel_config=c4)):.2f} ms", flush=True)
        print(f"  fp32 XP=X@P   HiFi4: {bench(dev, lambda: ttnn.matmul(xt32, p1, dtype=ttnn.float32, compute_kernel_config=c4)):.2f} ms", flush=True)
        # Check core_grid on 3D matmul!
        cg = ttnn.CoreGrid(y=10, x=12)
        print(f"  core_grid G  xh^T@xh HiFi2: {bench(dev, lambda: ttnn.matmul(xh, xh, transpose_a=True, dtype=ttnn.float32, core_grid=cg, compute_kernel_config=c2)):.2f} ms", flush=True)
        print(f"  core_grid XP xh@gh   HiFi2: {bench(dev, lambda: ttnn.matmul(xh, gh, dtype=ttnn.float32, core_grid=cg, compute_kernel_config=c2)):.2f} ms", flush=True)
        # Check bf16 direct matmul (p0 -> bf16? No, check bfp8 cross-terms!)
        xl8 = ttnn.typecast(xl, ttnn.bfloat8_b)
        gl8 = ttnn.typecast(gl, ttnn.bfloat8_b)
        print(f"  bfp8 XP p1=xh@gl8 LoFi: {bench(dev, lambda: ttnn.matmul(xh, gl8, dtype=ttnn.float32, compute_kernel_config=_ckc(ttnn.MathFidelity.LoFi))):.2f} ms", flush=True)
        for t in (xt32, xh, xl, p1, gh, gl, xl8, gl8):
            ttnn.deallocate(t)

        # Also check (36, 768, 768) NS split2_fast vs fp32!
        s_np = np.zeros((36, 768, 768), dtype=np.float32)
        for i in range(36):
            u = rng.standard_normal((768, 2)).astype(np.float32) * 0.05
            v = rng.standard_normal((2, 768)).astype(np.float32) * 0.05
            s_np[i] = u @ v + rng.standard_normal((768, 768)).astype(np.float32) * 0.005
        sref = _ns_f64(s_np)
        st32 = ttnn.from_torch(torch.from_numpy(s_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for mode in ("split2_fast", "fp32"):
            y = newton_schulz_device(st32, mode=mode, tall=False)
            ttnn.synchronize_device(dev)
            rel = _rel(ttnn.to_torch(y).float().numpy(), sref)
            ttnn.deallocate(y)
            ms = bench(dev, lambda: newton_schulz_device(st32, mode=mode, tall=False), reps=3)
            print(f"[NS (36,768,768)  {mode:12s}] {ms:6.2f} ms | rel_vs_f64={rel:.3e}", flush=True)
        ttnn.deallocate(st32)

        # --- PART 2: P3 ttnn.rms_norm + HiFi4 PROJ ---
        state, cfg = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
        glue = N.FlowGlue(state, cfg)
        sp = PairSplit(glue.pair_repr_builder)
        bt = load_batch("/home/ttuser/fleet/r4t/step/batch00.npz")[0]
        with torch.no_grad():
            c_ref = sp.constant(bt)
            p_ref = sp.inputs(bt, c_ref)
            z_ref_f64 = sp.assemble(p_ref).numpy()
            gk, bk = sp.modulation(bt)
        dp = DeviceTrainPair(glue.pair_repr_builder, dev)
        w_h = glue.pair_repr_builder.init_repr_factory.linear_out.weight.detach().float()
        lnw_h = glue.pair_repr_builder.init_repr_factory.ln_out.weight.detach().float()
        dm = {
            dp.W_KEY: ttnn.from_torch(w_h.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
            dp.LNW_KEY: ttnn.from_torch(lnw_h.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
        }
        dp.prepare_batch(bt, dm)
        st = dp._step_state
        # Check proj HiFi4
        w_rm = ttnn.to_layout(dm[dp.W_KEY], ttnn.ROW_MAJOR_LAYOUT)
        w_c_rm = ttnn.slice(w_rm, [0, dp.const_a], [dp.c, dp.const_b]); ttnn.deallocate(w_rm)
        w_c_pad = ttnn.pad(w_c_rm, [(0, 0), (0, dp.const_kpad - dp.const_dim)], 0.0); ttnn.deallocate(w_c_rm)
        w_c_tl = ttnn.to_layout(w_c_pad, ttnn.TILE_LAYOUT); ttnn.deallocate(w_c_pad)
        w0, w1, w2 = _split3_bf16(w_c_tl); ttnn.deallocate(w_c_tl)
        wt0 = ttnn.transpose(w0, -2, -1); wt1 = ttnn.transpose(w1, -2, -1); wt2 = ttnn.transpose(w2, -2, -1)
        for t in (w0, w1, w2): ttnn.deallocate(t)
        fhi = st["f_const_hi"]
        p0 = ttnn.matmul(fhi, wt0, dtype=ttnn.float32, compute_kernel_config=c4)
        p1 = ttnn.matmul(fhi, wt1, dtype=ttnn.float32, compute_kernel_config=c4)
        p2 = ttnn.matmul(fhi, wt2, dtype=ttnn.float32, compute_kernel_config=c4)
        s01 = ttnn.add(p0, p1); proj4 = ttnn.add(s01, p2)
        proj4_4d = ttnn.reshape(proj4, [1, 256, 256, 384])
        print(f"P3 PROJ HiFi4 rel vs host: {_rel(ttnn.to_torch(proj4_4d).float().numpy(), c_ref.proj[:1].numpy()):.3e}", flush=True)

        x = dp._gather_x(proj4_4d, st["table"], st["bins_tt"], 4, 256)
        x_ref = (c_ref.proj.double() + sp.w_step.double()[p_ref.bins]).numpy()
        print(f"P3 X (pair_build with HiFi4 proj) rel vs f64: {_rel(ttnn.to_torch(x).float().numpy(), x_ref):.3e}", flush=True)

        lnw_1d = ttnn.reshape(dm[dp.LNW_KEY], [1, 1, 1, 384])
        g_tt = ttnn.from_torch(gk.reshape(4, 1, 1, 384).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        b_tt = ttnn.from_torch(bk.reshape(4, 1, 1, 384).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        def do_rmsnorm():
            a = ttnn.rms_norm(x, weight=lnw_1d, epsilon=dp.e1, compute_kernel_config=c4)
            bb = ttnn.rms_norm(a, epsilon=dp.e2, compute_kernel_config=c4); ttnn.deallocate(a)
            sc = ttnn.multiply(bb, g_tt); ttnn.deallocate(bb)
            z = ttnn.add(sc, b_tt); ttnn.deallocate(sc)
            return z
        z_rn = do_rmsnorm()
        print(f"P3 ttnn.rms_norm (HiFi4) rel vs f64: {_rel(ttnn.to_torch(z_rn).float().numpy(), z_ref_f64):.3e}", flush=True)
        ttnn.deallocate(z_rn)
        print(f"P3 ttnn.rms_norm (4x256x256x384) time: {bench(dev, do_rmsnorm):.2f} ms", flush=True)
        print(f"P3 pair_build (4x256x256x384) time: {bench(dev, lambda: dp._gather_x(proj4_4d, st['table'], st['bins_tt'], 4, 256)):.2f} ms", flush=True)

        for t in (wt0, wt1, wt2, p0, p1, p2, s01, proj4, x, g_tt, b_tt): ttnn.deallocate(t)
        dp.release()
        for t in dm.values(): ttnn.deallocate(t)
    finally:
        pass


if __name__ == "__main__":
    main()
