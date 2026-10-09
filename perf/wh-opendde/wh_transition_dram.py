#!/usr/bin/env python3
"""Screen: OpenDDE's c_z=384 pair Transition with its hidden activations in DRAM and tall row blocks.

The shipped module keeps x_norm, x_1 and x_2 in L1, which caps the row block on Wormhole at 2-4 rows
of a 736-wide pair (c=384, hidden 1536): ~184 blocks of six small ops per call, 139-153 ms against a
~0.2 ms-per-block matmul roof. A DRAM block can be as tall as we like, so each linear is one large
matmul; what it costs is DRAM traffic for the two hidden tensors. This measures that trade: the
shipped module against a DRAM swiglu at row heights `--heights`, same weights, same compute kernel
config, same per-op arguments as `Transition.swiglu`, so only the memory config and the height differ.
Every arm is compared to the shipped output (torch.equal, max |diff|, rel rms).

Usage: TT_VISIBLE_DEVICES=<chip> python3 perf/wh-opendde/wh_transition_dram.py --out results/x.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def build_transition(c, n=4, seed=0, fidelity="HiFi3"):
    g = torch.Generator().manual_seed(seed)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(c, generator=g), "norm.bias": 0.1 * torch.randn(c, generator=g),
          "fc1.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc2.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc3.weight": torch.randn(c, n * c, generator=g) * ((n * c) ** -0.5)}
    ckc = ttnn.init_device_compute_kernel_config(
        T.get_device().arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        fp32_dest_acc_en=True, packer_l1_acc=True)
    return T.Transition(sd, ckc)


def dram_transition(tr, x, h):
    """`Transition.swiglu` over row blocks of height h, every intermediate in DRAM."""
    DR = ttnn.DRAM_MEMORY_CONFIG
    dtype = tr.dtype if tr.dtype is not None else T._dtype()
    hidden = ttnn.bfloat8_b if tr._hidden_b8 else dtype
    H, W, c = x.shape[1], x.shape[2], x.shape[3]
    outs = []
    for r0 in range(0, H, h):
        r1 = min(H, r0 + h)
        xs = ttnn.slice(x, [0, r0, 0, 0], [1, r1, W, c], memory_config=DR)
        xn = ttnn.layer_norm(xs, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                             compute_kernel_config=tr.compute_kernel_config, memory_config=DR)
        ttnn.deallocate(xs)
        x1 = ttnn.linear(xn, tr.fc1_weight, activation="silu",
                         compute_kernel_config=T.silu_ckc(tr.compute_kernel_config),
                         memory_config=DR, dtype=hidden, core_grid=T.CORE_GRID_MAIN)
        x2 = ttnn.linear(xn, tr.fc2_weight, compute_kernel_config=tr.compute_kernel_config,
                         memory_config=DR, dtype=hidden, core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(xn)
        x1 = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        outs.append(ttnn.linear(x1, tr.fc3_weight, compute_kernel_config=tr.compute_kernel_config,
                                dtype=dtype, core_grid=T.CORE_GRID_MAIN, memory_config=DR))
        ttnn.deallocate(x1)
    if len(outs) == 1:
        return outs[0]
    out = ttnn.concat(outs, dim=1, memory_config=DR)
    for o in outs:
        ttnn.deallocate(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="736,1024")
    ap.add_argument("--c", type=int, default=384)
    ap.add_argument("--heights", default="8,16,32,64,128")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--fidelity", default="HiFi3")
    ap.add_argument("--levers", default="normal")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "arch": str(dev.arch()).rsplit(".", 1)[-1], "c": a.c, "fidelity": a.fidelity,
           "levers": a.levers, "rows": []}
    with T.levers(a.levers):
        tr = build_transition(a.c, fidelity=a.fidelity)
        for W in [int(s) for s in a.sizes.split(",")]:
            x_t = torch.randn(1, W, W, a.c) * 0.5
            ref = None
            for arm in ["shipped"] + [f"dram{h}" for h in a.heights.split(",")]:
                row = {"W": W, "arm": arm}
                try:
                    walls = []
                    for i in range(a.warm + a.iters):
                        xt = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        out = tr(xt) if arm == "shipped" else dram_transition(tr, xt, int(arm[4:]))
                        ttnn.synchronize_device(dev)
                        if i >= a.warm:
                            walls.append(time.perf_counter() - t0)
                        ho = ttnn.to_torch(out).float()
                        ttnn.deallocate(out)
                        ttnn.deallocate(xt)
                    row["ms"] = round(st.median(walls) * 1e3, 3)
                    row["spread_pct"] = round(100 * (max(walls) - min(walls)) / st.median(walls), 2)
                    if ref is None:
                        ref = ho
                    row["equal"] = bool(torch.equal(ho, ref))
                    row["max_abs"] = float((ho - ref).abs().max())
                    row["rel_rms"] = float(((ho - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
                    print(f"W={W} {arm:9s} {row['ms']:9.3f} ms (spread {row['spread_pct']:.1f}%) "
                          f"equal={row['equal']} max_abs={row['max_abs']:.3g} rel_rms={row['rel_rms']:.3g}",
                          flush=True)
                except Exception as e:                                          # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    print(f"W={W} {arm} FAILED {row['error']}", flush=True)
                res["rows"].append(row)
                a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
