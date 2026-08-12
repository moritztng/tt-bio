#!/usr/bin/env python3
"""SCREEN: how much of a DiT block's 4.969 ms is the per-block pair-bias projection?

`_DiTBlock.__call__` (tt_bio/openfold3_diffusion_transformer.py) opens with

    z  = layer_norm(z)            # [1,512,512,128] fp32
    zb = linear(z, w_lin_z)       # -> [1,512,512,16]
    zb = to_layout(RM); permute(0,3,1,2); to_layout(TILE)
    zb = add_(zb, mask_bias)      # [1,16,512,512]

`z` is the CONDITIONED pair. It does not depend on the diffusion step, so all 24 blocks x 200 steps
compute 24 distinct tensors 200 times each. This prices the sequence with the real shapes and
dtypes, against the rest of the block (the token attention), so the hoist has a predicted landing
before anything is built. Random weights: shapes set the cost, values do not.
"""
import argparse, json, os, time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import CORE_GRID_MAIN
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    F32 = ttnn.float32
    N, H = a.n, 16
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                           packer_l1_acc=True) \
        if hasattr(ttnn, "WormholeComputeKernelConfig") else None

    z = ttnn.from_torch(torch.randn(1, N, N, 128), layout=ttnn.TILE_LAYOUT, device=dev, dtype=F32)
    lnw = ttnn.from_torch(torch.randn(128), layout=ttnn.TILE_LAYOUT, device=dev, dtype=F32)
    wz = ttnn.from_torch(torch.randn(128, H), layout=ttnn.TILE_LAYOUT, device=dev, dtype=F32)
    mb = ttnn.from_torch(torch.zeros(1, 1, 1, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=F32)

    def sync():
        ttnn.synchronize_device(dev)

    def timeit(fn, iters):
        sync(); fn(); sync()
        ts = []
        for _ in range(iters):
            sync(); t0 = time.perf_counter(); fn(); sync()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    def pair_bias():
        zl = ttnn.layer_norm(z, weight=lnw, epsilon=1e-5, compute_kernel_config=ckc)
        zb = ttnn.linear(zl, wz, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(zl)
        zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
        zb = ttnn.permute(zb, (0, 3, 1, 2))
        zb = ttnn.to_layout(zb, ttnn.TILE_LAYOUT)
        zb = ttnn.add_(zb, mb)
        ttnn.deallocate(zb)

    def ln_only():
        zl = ttnn.layer_norm(z, weight=lnw, epsilon=1e-5, compute_kernel_config=ckc)
        ttnn.deallocate(zl)

    def ln_lin():
        zl = ttnn.layer_norm(z, weight=lnw, epsilon=1e-5, compute_kernel_config=ckc)
        zb = ttnn.linear(zl, wz, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(zl); ttnn.deallocate(zb)

    def clone_z():
        y = ttnn.clone(z)
        ttnn.deallocate(y)

    rows = {}
    for label, fn in (("pair_bias_full", pair_bias), ("layer_norm_only", ln_only),
                      ("layer_norm+linear", ln_lin), ("clone_z (roof ref)", clone_z)):
        ms = timeit(fn, a.iters) * 1e3
        rows[label] = round(ms, 3)
        print(f"  {label:22s} {ms:8.3f} ms", flush=True)

    zbytes = N * N * 128 * 4
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "n": N, "blocks": a.blocks, "steps": a.steps,
           "z_MB": round(zbytes / 1e6, 1),
           "clone_implied_GBs": round(2 * zbytes / (rows["clone_z (roof ref)"] / 1e3) / 1e9, 1),
           "ms": rows,
           "pair_bias_per_step_ms": round(rows["pair_bias_full"] * a.blocks, 2),
           "hoistable_s_over_rollout": round(rows["pair_bias_full"] * a.blocks
                                             * (a.steps - 1) / 1e3, 3)}
    print(f"\n  pair bias x{a.blocks} blocks = {res['pair_bias_per_step_ms']:.2f} ms/step; "
          f"hoisting 199/200 of it = {res['hoistable_s_over_rollout']:.3f} s over the rollout",
          flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
