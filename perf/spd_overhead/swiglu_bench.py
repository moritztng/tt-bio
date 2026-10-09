"""Transition (swiglu) silu placement: accuracy against a float64 reference and ms per module call.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/swiglu_bench.py --out OUT.json [--levers silu_f32]

Arms: `fused` (silu in fc1's epilogue, shipped) and `unfused` (standalone silu on bf16 fc1, held off
on accuracy). `--levers` sets Protenix's lever set for every arm (`silu_f32` runs the fused silu on
kernels/silu_f32); run once with and once without and compare `fused`. Each arm runs the
real `Transition` module on the whole pair tensor, row chunking included, with Protenix's compute
config, so the time is what a fold pays per call. Error is quoted against float64 math on the same
bf16 inputs and weights.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 2, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="fused,unfused")
    ap.add_argument("--levers", default="")
    ap.add_argument("--shapes", default="736x256x1024,736x128x512")
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    T._LEVERS = T.parse_levers(a.levers)  # the whole run, as a fold holds it
    dev = T.get_device()
    arch = "wormhole" if T.is_wormhole() else "blackhole"
    ckc = (ttnn.types.BlackholeComputeKernelConfig if arch == "blackhole"
           else ttnn.WormholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": arch,
           "grid": list(T.COMPUTE_GRID_MAIN),
           "levers": a.levers, "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT"), "loadavg": open("/proc/loadavg").read().split()[:3],
           "rows": []}

    def sync():
        ttnn.synchronize_device(dev)

    for shp in a.shapes.split(","):
        S, C, HID = map(int, shp.split("x"))
        g = torch.Generator().manual_seed(0)
        bf = lambda t: t.to(torch.bfloat16).to(torch.float64)
        sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
              "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
              "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
              "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
              "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
        zt = bf(torch.randn(1, S, S, C, generator=g))
        xn = torch.nn.functional.layer_norm(zt, (C,), sd["norm.weight"], sd["norm.bias"], 1e-5)
        ref = (torch.nn.functional.silu(xn @ sd["fc1.weight"].T) * (xn @ sd["fc2.weight"].T)) @ sd["fc3.weight"].T
        tr = T.Transition({k: v.float() for k, v in sd.items()}, ckc)
        z = ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for arm in a.arms.split(","):
            T._UNFUSED_SILU = arm == "unfused"
            row = {"shape": shp, "arm": arm}
            try:
                o = tr(z)
                out = ttnn.to_torch(o).to(torch.float64)
                ttnn.deallocate(o)
                d = out - ref
                row["rel_rms"] = float(d.norm() / ref.norm())
                row["max_abs"] = float(d.abs().max())
                row["mean_abs"] = float(d.abs().mean())
                i = int(d.abs().argmax())
                row["argmax"] = [int(v) for v in torch.unravel_index(torch.tensor(i), d.shape)]
                row["at"] = [float(out.flatten()[i]), float(ref.flatten()[i])]
                # The same call again, warm: a cold-only error is a different bug from a steady one.
                o = tr(z)
                d2 = ttnn.to_torch(o).to(torch.float64) - ref
                ttnn.deallocate(o)
                row["max_abs_warm"] = float(d2.abs().max())
                row["n_over_0.1"] = [int((d.abs() > 0.1).sum()), int((d2.abs() > 0.1).sum())]
                for _ in range(WARM):
                    ttnn.deallocate(tr(z))
                sync()
                ts = []
                for _ in range(REPS):
                    sync(); t0 = time.perf_counter()
                    o = tr(z); sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o)
                row["ms"] = round(st.median(ts), 3)
                row["ms_spread"] = round(max(ts) - min(ts), 3)
                row["h_chunk"] = [k[2] for k in T.TRANSITION_H_CHUNK_SHAPES if k[0].startswith(f"{S}x{S}x{C}")]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
        ttnn.deallocate(z)
    T._UNFUSED_SILU = False
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
