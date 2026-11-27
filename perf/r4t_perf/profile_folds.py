#!/usr/bin/env python3
"""Profile DeviceFolds.tape_backward & refold by op type and verify F1 fast transpose parity."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import autograd as ag  # noqa: E402
from tt_bio.rfd4 import engine as E, network as N  # noqa: E402
from tt_bio.rfd4.heads import DeviceHeads  # noqa: E402
from tt_bio.train.device_optim import transpose_exact_f32  # noqa: E402
from tt_bio.train.folds import DeviceFolds, make_rfd4_refold_hook  # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else ""
    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    glue.requires_grad_(True)
    state_res = dict(state)
    for gn, gp in glue.named_parameters():
        if gp.requires_grad:
            state_res[f"glue.{gn}"] = gp.detach()

    ag.install()
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    try:
        # 1. Verify F1 fast transpose bit-exactness for bf16 and bfp8 on RFD4 shapes
        rng = np.random.default_rng(123)
        t_checks = {}
        for shp in [(6144, 768), (768, 3072), (768, 768), (256, 384), (12, 384)]:
            arr = rng.standard_normal(shp).astype(np.float32) * 0.1
            x = ttnn.from_torch(torch.from_numpy(arr), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            ref32 = transpose_exact_f32(x)
            ref16 = ttnn.typecast(ref32, ttnn.bfloat16)
            ref8 = ttnn.typecast(ref32, ttnn.bfloat8_b)

            s16 = ttnn.typecast(x, ttnn.bfloat16)
            f1_16 = ttnn.transpose(s16, -2, -1)
            ttnn.deallocate(s16)
            t32 = ttnn.transpose(x, -2, -1)
            f1_8 = ttnn.typecast(t32, ttnn.bfloat8_b)
            ttnn.deallocate(t32)

            r16_np = ttnn.to_torch(ref16).float().numpy()
            f16_np = ttnn.to_torch(f1_16).float().numpy()
            r8_np = ttnn.to_torch(ref8).float().numpy()
            f8_np = ttnn.to_torch(f1_8).float().numpy()
            d16 = float(np.max(np.abs(r16_np - f16_np)))
            d8 = float(np.max(np.abs(r8_np - f8_np)))
            rel8 = float(np.linalg.norm(r8_np - f8_np) / (np.linalg.norm(r8_np) + 1e-30))
            t_checks[str(shp)] = dict(bf16_max_diff=d16, bfp8_max_diff=d8, bfp8_rel_l2=rel8)
            print(f"[F1 TRANSPOSE {shp}] bf16_max_diff={d16:.3e} bfp8_max_diff={d8:.3e} bfp8_rel={rel8:.3e}", flush=True)
            for t in (x, ref32, ref16, ref8, f1_16, f1_8):
                ttnn.deallocate(t)

        # 2. Record RFD4 DeviceFolds and inspect DAG op census + refold / tape_backward timing
        def pack(v, *, dtype=None, layout=None, memory_config=None):
            dt = ttnn.bfloat16 if dtype is None else dtype
            lay = ttnn.TILE_LAYOUT if layout is None else layout
            mc = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
            return ttnn.from_torch(v.detach().float().contiguous(), dtype=dt, layout=lay, device=dev, memory_config=mc)

        df = DeviceFolds(state_res, dev)
        with df.recording(device_pack=pack) as sd:
            trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=ttnn.bfloat16, attn_fp32=True)
            for pu in trunk.pair_updates.values():
                for attr in ("tri_out", "tri_in"):
                    m = getattr(pu, attr, None)
                    if m is not None and hasattr(m, "prewarm"):
                        m.prewarm(256, 1)
            dheads = DeviceHeads(glue, dev, state=sd)
        df.bind(trunk, dheads)

        # Build dev_masters dict
        dev_masters = {}
        for nm, t in df.leaves.items():
            dev_masters[nm] = ttnn.from_torch(t.detach().float().contiguous(), dtype=ttnn.float32,
                                              layout=ttnn.TILE_LAYOUT, device=dev)

        # Count DAG ops & spec dtypes
        op_cnt = defaultdict(int)
        for nd in df.nodes:
            op_cnt[nd.op] += 1
        dtype_cnt = defaultdict(int)
        for sp in df.specs:
            dtype_cnt[str(sp.dtype)] += 1
        print(f"[FOLDS CENSUS] nodes={len(df.nodes)} ops={dict(op_cnt)} specs={len(df.specs)} dtypes={dict(dtype_cnt)}", flush=True)

        # Time refold + hook (3 warm runs)
        hook_no_pair = make_rfd4_refold_hook(trunk, glue, host_glue_params={}, pair_card_ref=None, device=dev)
        refold_ms = []
        hook_ms = []
        for i in range(4):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            df.refold(dev_masters, on_refold=None)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            hook_no_pair(dev_masters)
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            if i >= 1:
                refold_ms.append(1000 * (t1 - t0))
                hook_ms.append(1000 * (t2 - t1))

        # Time tape_backward (populate synthetic grads on all specs, 3 warm runs)
        bwd_ms = []
        for i in range(4):
            for sp in df.specs:
                leaf = ag.parameter_for(sp.handle)
                if leaf is not None:
                    leaf.grad = ttnn.zeros_like(sp.handle, dtype=ttnn.float32)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            dg, idle = df.tape_backward(dev_masters)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            if i >= 1:
                bwd_ms.append(1000 * (t1 - t0))
            for t in dg.values():
                ttnn.deallocate(t)

        print(f"[FOLDS TIMING] refold={np.median(refold_ms):.1f} ms | hook(no_pair)={np.median(hook_ms):.1f} ms | "
              f"tape_backward={np.median(bwd_ms):.1f} ms", flush=True)

        # Op-by-op timing breakdown inside tape_backward (1 pass)
        for sp in df.specs:
            leaf = ag.parameter_for(sp.handle)
            if leaf is not None:
                leaf.grad = ttnn.zeros_like(sp.handle, dtype=ttnn.float32)
        import tt_bio.train.folds as Fmod
        op_time = defaultdict(float)
        op_calls = defaultdict(int)
        def wrap(name):
            orig = getattr(Fmod, name)
            def wrapped(*a, **kw):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = orig(*a, **kw)
                ttnn.synchronize_device(dev)
                op_time[name] += 1000 * (time.perf_counter() - t0)
                op_calls[name] += 1
                return r
            return orig, wrapped
        fns = ["_dev_transpose_2d", "_dev_slice_f32", "_dev_concat_f32", "_dev_reshape_f32",
               "_dev_mul_f32", "_dev_reduce_to_f32", "_dev_expand_f32"]
        saved = {}
        for fn in fns:
            o, w = wrap(fn)
            saved[fn] = o
            setattr(Fmod, fn, w)
        dg, _ = df.tape_backward(dev_masters)
        for fn, o in saved.items():
            setattr(Fmod, fn, o)
        for t in dg.values():
            ttnn.deallocate(t)
        print("[TAPE_BWD BY HELPER]", {k: f"{op_time[k]:.1f}ms ({op_calls[k]}x)" for k in sorted(op_time, key=lambda x: -op_time[x])}, flush=True)

        res = dict(
            transpose_checks=t_checks,
            census=dict(nodes=len(df.nodes), ops=dict(op_cnt), specs=len(df.specs), dtypes=dict(dtype_cnt)),
            refold_ms=round(float(np.median(refold_ms)), 2),
            hook_ms=round(float(np.median(hook_ms)), 2),
            tape_backward_ms=round(float(np.median(bwd_ms)), 2),
            bwd_helpers={k: dict(ms=round(op_time[k], 2), calls=op_calls[k]) for k in op_time},
        )
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
    finally:
        pass


if __name__ == "__main__":
    main()
