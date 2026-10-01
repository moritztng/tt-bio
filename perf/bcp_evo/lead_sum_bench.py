#!/usr/bin/env python3
"""triatt_bw's dbias reduction: `lead_sum` against `ttnn.sum(dim=0)`, graded against float64.

Float32 partials at the round's [27, 4, 288, 288] and a few others; rel L2 and max abs against the
float64 sum of the same float32 values, inputs read back intact, then timed synced with the arms
alternating every rep. gate_bw_bench.py is the template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="27x4x288x288,13x4x288x288,27x4x512x512,5x3x50x70,2x1x32x32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/lead_sum_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import lead_sum as LS
    from tt_bio.tenstorrent import get_device
    LS.LEAD_SUM_FUSED = True
    dev = get_device()
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {}
    for spec in a.shapes.split(","):
        shape = [int(v) for v in spec.split("x")]
        torch.manual_seed(0)
        x = torch.randn(shape) * 1e-2
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref = x.double().sum(0, keepdim=True)
        fns = {"ttnn_sum": lambda: ttnn.sum(xd, dim=0, keepdim=True),
               "lead_sum": lambda: (LS.eligible(xd) or (_ for _ in ()).throw(AssertionError()))
               and LS.lead_sum(xd)}
        out = {}
        for arm, fn in fns.items():
            y = down(fn()).reshape(ref.shape)
            out[arm] = {"rel_l2": float((y - ref).norm() / ref.norm()),
                        "max_abs": float((y - ref).abs().max()), "t": []}
        out["inputs_intact"] = bool(torch.equal(down(xd), x.double()))
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
        out["window_utc"] = time.time()
        res[spec] = out
        print(json.dumps({spec: out}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
