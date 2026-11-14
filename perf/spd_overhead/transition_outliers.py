"""Where the pair Transition's large errors against float64 come from: which pixels, which stage.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/transition_outliers.py --out OUT.json [--shape 736x256x1024]

Same inputs as swiglu_bench.py (seed 0). Runs the real module, lists the pixels whose output is off
by more than 0.1, then reruns the stages by hand on the row block that holds the worst pixel, with
the module's own weights and kernel config, and compares each stage with float64 math on the
device's own bf16 input to that stage, so an error is pinned to the op that made it.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shape", default="736x256x1024")
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    dev = T.get_device()
    arch = "wormhole" if T.is_wormhole() else "blackhole"
    ckc = (ttnn.types.BlackholeComputeKernelConfig if arch == "blackhole"
           else ttnn.WormholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    S, C, HID = map(int, a.shape.split("x"))
    g = torch.Generator().manual_seed(0)
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)
    sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
          "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
          "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
    zt = bf(torch.randn(1, S, S, C, generator=g))
    F = torch.nn.functional
    xn = F.layer_norm(zt, (C,), sd["norm.weight"], sd["norm.bias"], 1e-5)
    ref = (F.silu(xn @ sd["fc1.weight"].T) * (xn @ sd["fc2.weight"].T)) @ sd["fc3.weight"].T
    tr = T.Transition({k: v.float() for k, v in sd.items()}, ckc)
    z = ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = tr(z)
    out = ttnn.to_torch(o).to(torch.float64)
    ttnn.deallocate(o)
    h = [k[2] for k in T.TRANSITION_H_CHUNK_SHAPES][0]
    err = (out - ref).abs().amax(-1)[0]                       # per pixel [S, S]
    bad = (err > 0.1).nonzero().tolist()
    res = {"arch": arch, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "shape": a.shape, "h": h,
           "n_bad_px": len(bad), "bad_px": bad[:64],
           "bad_rows": sorted({r for r, _ in bad}), "bad_cols": sorted({c for _, c in bad}),
           "input_absmax_bad": [float(zt[0, r, c].abs().max()) for r, c in bad[:8]],
           "input_std_bad": [float(zt[0, r, c].std()) for r, c in bad[:8]]}
    print(json.dumps(res), flush=True)
    if bad:
        r, c = max(bad, key=lambda p: float(err[p[0], p[1]]))
        s = (r // h) * h
        blk = z[:, s:min(s + h, S)]
        st = {}
        dt = lambda t: ttnn.to_torch(t).to(torch.float64)
        x_norm = ttnn.layer_norm(blk, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                                 compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        xb = dt(blk)
        xnd = dt(x_norm)
        want = F.layer_norm(xb, (C,), sd["norm.weight"], sd["norm.bias"], 1e-5)
        st["layer_norm"] = float((xnd - want)[0, r - s, c].abs().max())
        mm = lambda x, w, act=None: ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                                dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
        x1 = mm(x_norm, tr.fc1_weight, "silu"); x2 = mm(x_norm, tr.fc2_weight)
        x1d, x2d = dt(x1), dt(x2)
        st["fc1_silu"] = float((x1d - F.silu(xnd @ sd["fc1.weight"].T))[0, r - s, c].abs().max())
        st["fc2"] = float((x2d - xnd @ sd["fc2.weight"].T)[0, r - s, c].abs().max())
        x1n = mm(x_norm, tr.fc1_weight)
        st["fc1_nosilu"] = float((dt(x1n) - xnd @ sd["fc1.weight"].T)[0, r - s, c].abs().max())
        p = ttnn.multiply(x1, x2)
        pd = dt(p)
        st["multiply"] = float((pd - x1d * x2d)[0, r - s, c].abs().max())
        y = ttnn.linear(p, tr.fc3_weight, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                        core_grid=T.CORE_GRID_MAIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        st["fc3"] = float((dt(y) - pd @ sd["fc3.weight"].T)[0, r - s, c].abs().max())
        st["end_to_end_block"] = float((dt(y) - ref[:, s:s + h])[0, r - s, c].abs().max())
        st["module_px"] = float(err[r, c])
        # The pixel's own numbers, to see whether it is an input-conditioning effect.
        st["px"] = [r, c]
        st["x_norm_absmax"] = float(xnd[0, r - s, c].abs().max())
        st["fc1_absmax"] = float(x1d[0, r - s, c].abs().max())
        st["fc2_absmax"] = float(x2d[0, r - s, c].abs().max())
        st["prod_absmax"] = float(pd[0, r - s, c].abs().max())
        st["ref_absmax"] = float(ref[0, r, c].abs().max())
        e2 = (x2d - xnd @ sd["fc2.weight"].T)[0, r - s, c]
        j = int(e2.abs().argmax())
        st["fc2_at"] = {"channel": j, "device": float(x2d[0, r - s, c, j]),
                        "want": float((xnd @ sd["fc2.weight"].T)[0, r - s, c, j])}
        res["stages"] = st
        print(json.dumps(st), flush=True)
        # fc2 alone over the whole block under each kernel config: count elements off by > 0.5.
        want2 = xnd @ sd["fc2.weight"].T
        sweep = []
        for fid in ("HiFi4", "HiFi3", "HiFi2"):
            for f32 in (True, False):
                for l1acc in (True, False):
                    k = type(ckc)(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=True,
                                  fp32_dest_acc_en=f32, packer_l1_acc=l1acc)
                    for odt in ("bf16", "fp32"):
                        y2 = ttnn.linear(x_norm, tr.fc2_weight, compute_kernel_config=k,
                                         memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN,
                                         dtype=ttnn.bfloat16 if odt == "bf16" else ttnn.float32)
                        d2 = (dt(y2) - want2).abs()
                        ttnn.deallocate(y2)
                        sweep.append({"fid": fid, "fp32_acc": f32, "l1_acc": l1acc, "out": odt,
                                      "n_over_0.5": int((d2 > 0.5).sum()), "max": float(d2.max())})
                        print(json.dumps(sweep[-1]), flush=True)
        res["fc2_sweep"] = sweep
        # The device's own copy of the weight against the bf16 values it was built from.
        w2 = dt(tr.fc2_weight).reshape(-1, HID)[:C]
        res["fc2_weight_vs_sd"] = float((w2 - sd["fc2.weight"].T).abs().max())
        print(json.dumps({"fc2_weight_vs_sd": res["fc2_weight_vs_sd"]}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
