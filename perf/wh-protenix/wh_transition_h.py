#!/usr/bin/env python3
"""Protenix-v2's 298 aa Wormhole crash, reduced to the one op that throws, plus the h sweep
that prices the fix.

The fold dies in `Transition.__call__`'s eager chunked path (tenstorrent.py:3536) on
`ttnn.linear` (:3432) with a statically-allocated-CB-vs-L1-buffer clash across the whole 8x9
grid. This rebuilds exactly that loop -- same shapes, same memory configs, same compute kernel
config, same grid -- with synthetic weights, and sweeps the row-chunk h. It answers three
things a fold cannot answer cheaply:

  * which h clashes at the failing width and which h survives,
  * what each surviving h costs, so the fix is the largest h that fits and not the smallest,
  * whether h is a bit-exact partition (it should be: swiglu is row-local).

No fold, no weights, one card, seconds per point.
"""
import argparse, json, os, statistics as st, sys, time


def build(dev, ttnn, torch, H, W, c, hidden, seed=0):
    torch.manual_seed(seed)
    x = ttnn.from_torch(torch.randn(1, H, W, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = {}
    for k, shp in (("norm_w", (c,)), ("norm_b", (c,)), ("fc1", (c, hidden)),
                   ("fc2", (c, hidden)), ("fc3", (hidden, c))):
        t = torch.randn(*shp) * 0.02 if len(shp) == 2 else torch.ones(*shp)
        w[k] = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x, w


def swiglu(ttnn, ckc, cg, x, w):
    """Byte-for-byte the shipped tt_bio Transition.swiglu (tenstorrent.py:3422-3463)."""
    x_norm = ttnn.layer_norm(x, weight=w["norm_w"], bias=w["norm_b"], epsilon=1e-5,
                             compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_1 = ttnn.linear(x_norm, w["fc1"], activation="silu", compute_kernel_config=ckc,
                      memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=cg)
    x_2 = ttnn.linear(x_norm, w["fc2"], compute_kernel_config=ckc,
                      memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=cg)
    ttnn.deallocate(x_norm)
    y = ttnn.multiply_(x_1, x_2)
    ttnn.deallocate(x_2)
    out = ttnn.linear(y, w["fc3"], compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                      core_grid=cg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(y)
    return out


def run_h(ttnn, ckc, cg, x, w, H, h):
    """The shipped eager path: ttnn.chunk on dim 1, swiglu per chunk, concat."""
    chunks = ttnn.chunk(x, -(-H // h), dim=1)
    out = ttnn.concat([swiglu(ttnn, ckc, cg, c, w) for c in chunks], dim=1)
    for c in chunks:
        ttnn.deallocate(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=320, help="pair width W (298 aa pads to 320)")
    ap.add_argument("--h-total", type=int, default=0, help="rows H; default = W (square pair)")
    ap.add_argument("--c", type=int, default=256, help="protenix-v2 c_z")
    ap.add_argument("--hidden", type=int, default=1024, help="transition n=4 -> 4*c_z")
    ap.add_argument("--hs", default="4,8,12,16,20,24,25,32,40")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch, ttnn
    dev = ttnn.open_device(device_id=0)
    try:
        g = dev.compute_with_storage_grid_size()
        cg = ttnn.CoreGrid(y=int(g.y), x=int(g.x))
        arch = ttnn.get_arch_name()
        kcls = (ttnn.types.WormholeComputeKernelConfig if arch == "wormhole_b0"
                else ttnn.types.BlackholeComputeKernelConfig)
        ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=True)
        l1 = ttnn.get_max_worker_l1_unreserved_size()
        H = a.h_total or a.w
        meta = {"arch": arch, "grid": [int(g.x), int(g.y)], "cores": int(g.x) * int(g.y),
                "l1_per_core": int(l1), "H": H, "W": a.w, "c": a.c, "hidden": a.hidden,
                "host": os.uname().nodename}
        print(json.dumps(meta), flush=True)

        x, w = build(dev, ttnn, torch, H, a.w, a.c, a.hidden)
        rows, ref = [], None
        for h in [int(v) for v in a.hs.split(",")]:
            nchunk = -(-H // h)
            # per-chunk L1 elements the shipped code prices, and the per-core bytes it lands as
            elem = h * a.w * a.c
            per_core = (elem + a.hidden * h * a.w * 2) * 2 / (int(g.x) * int(g.y))
            r = {"h": h, "n_chunks": nchunk, "chunk_elems": elem,
                 "est_l1_bytes_per_core": round(per_core)}
            try:
                for _ in range(a.warm):
                    o = run_h(ttnn, ckc, cg, x, w, H, h)
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                ms = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    o = run_h(ttnn, ckc, cg, x, w, H, h)
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3)
                    got = ttnn.to_torch(o)
                    ttnn.deallocate(o)
                r["status"] = "ok"
                r["ms_median"] = round(st.median(ms), 4)
                r["ms_all"] = [round(v, 4) for v in ms]
                if ref is None:
                    ref = got
                    r["bit_exact_vs_ref"] = None
                    r["ref_h"] = h
                else:
                    r["bit_exact_vs_ref"] = bool(torch.equal(ref, got))
            except Exception as e:
                msg = str(e).replace("\n", " ")
                r["status"] = "THROW"
                r["error"] = msg[:400]
                r["is_cb_clash"] = "circular buffer" in msg.lower()
            rows.append(r)
            print(json.dumps(r), flush=True)
        out = {"meta": meta, "rows": rows}
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1)
            print("wrote " + a.out, flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
