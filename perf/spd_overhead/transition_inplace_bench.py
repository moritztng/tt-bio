"""Pair Transition `z + t(z)`: eager chunk/concat/add_ against row blocks written back in place.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/transition_inplace_bench.py --out OUT.json

Runs the real `Transition` module with add_to_input=True, the call the Pairformer makes, once with
TRANSITION_ROWS_INPLACE off and once on, on the same bf16 input. Reports ms per call and whether the
two outputs are torch.equal.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 2, 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="736x256x1024,512x256x1024,256x256x1024,736x128x512")
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    dev = T.get_device()
    arch = "wormhole" if T.is_wormhole() else "blackhole"
    ckc = (ttnn.types.BlackholeComputeKernelConfig if arch == "blackhole"
           else ttnn.WormholeComputeKernelConfig)(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": arch,
           "grid": list(T.COMPUTE_GRID_MAIN), "loadavg": open("/proc/loadavg").read().split()[:3],
           "rows": []}
    sync = lambda: ttnn.synchronize_device(dev)

    for shp in a.shapes.split(","):
        S, C, HID = map(int, shp.split("x"))
        g = torch.Generator().manual_seed(0)
        sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g),
              "norm.bias": 0.1 * torch.randn(C, generator=g),
              "fc1.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
              "fc2.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
              "fc3.weight": torch.randn(C, HID, generator=g) / HID ** 0.5}
        zt = torch.randn(1, S, S, C, generator=g).to(torch.bfloat16)
        tr = T.Transition(sd, ckc)
        up = lambda: ttnn.from_torch(zt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        outs = {}
        for arm in ("eager", "inplace"):
            T.TRANSITION_ROWS_INPLACE = arm == "inplace"
            row = {"shape": shp, "arm": arm}
            try:
                n0 = T.PAIR_INPLACE_STATS[1]
                o = tr(up(), add_to_input=True)
                row["blocks_written"] = T.PAIR_INPLACE_STATS[1] - n0
                outs[arm] = ttnn.to_torch(o)
                ttnn.deallocate(o)
                for _ in range(WARM):
                    ttnn.deallocate(tr(up(), add_to_input=True))
                ts = []
                for _ in range(REPS):
                    z = up(); sync(); t0 = time.perf_counter()
                    o = tr(z, add_to_input=True); sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o)
                row["ms"] = round(st.median(ts), 3)
                row["ms_spread"] = round(max(ts) - min(ts), 3)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
        if len(outs) == 2:
            eq = {"shape": shp, "equal": bool(torch.equal(outs["eager"], outs["inplace"]))}
            print(json.dumps(eq), flush=True)
            res["rows"].append(eq)
    T.TRANSITION_ROWS_INPLACE = False
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
