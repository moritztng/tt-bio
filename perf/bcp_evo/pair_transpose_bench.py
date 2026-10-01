#!/usr/bin/env python3
"""`pair_transpose` against the round's route (`tenstorrent._pair_transpose_impl` to DRAM) and the
tiled `ttnn.permute`. torch.equal against the torch permutation, input intact, then timed synced
with the arms alternating every rep. lead_sum_bench.py is the template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="288x288x128,288x288x32,256x256x128,512x512x128,64x96x64")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/pair_transpose_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import pair_transpose as PT, tenstorrent as TN
    from tt_bio.tenstorrent import get_device
    PT.PAIR_TRANSPOSE_FUSED = True
    dev = get_device()
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).float()  # noqa: E731
    D = ttnn.DRAM_MEMORY_CONFIG
    res = {}
    for spec in a.shapes.split(","):
        for dt, tdt in ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)):
            shape = [int(v) for v in spec.split("x")]
            torch.manual_seed(0)
            x = torch.randn(shape).to(tdt)
            xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev, memory_config=D)
            ref = x.float().permute(1, 0, 2)
            assert PT.eligible(xd)
            fns = {"round_route": lambda: TN._pair_transpose_impl(xd, D),
                   "permute_tiled": lambda: ttnn.permute(xd, (1, 0, 2), memory_config=D),
                   "pair_transpose": lambda: PT.pair_transpose(xd)}
            out = {}
            for arm, fn in fns.items():
                out[arm] = {"equal": bool(torch.equal(down(fn()), ref)), "t": []}
            out["input_intact"] = bool(torch.equal(down(xd), x.float()))
            for rep in range(a.reps):
                for arm in (list(fns) if rep % 2 == 0 else list(fns)[::-1]):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    r = fns[arm]()
                    ttnn.synchronize_device(dev)
                    out[arm]["t"].append(time.perf_counter() - t0)
                    del r
            for arm in fns:
                ts = sorted(out[arm].pop("t"))
                out[arm].update(median_ms=1e3 * ts[len(ts) // 2], min_ms=1e3 * ts[0])
            key = f"{spec} {'bf16' if dt == ttnn.bfloat16 else 'f32'}"
            out["window_utc"] = time.time()
            res[key] = out
            print(json.dumps({key: out}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
