#!/usr/bin/env python3
"""The fan-in's last add: `rne_add.round_add(acc, last)` against `typecast(widen_add(acc, last))`.

acc float32 (or bf16 for a two-way fan-in), last bf16, at the round's [1, 288, 288, 128] and two
others. Equality arm to arm, rel L2 of each against the float64 sum, then timed synced with the arms
alternating every rep. lead_sum_bench.py is the template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1x288x288x128,1x288x288x32,1x512x512x128")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/fanin_cast_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import rne_add as R
    from tt_bio.tenstorrent import get_device
    R.WIDEN_ADD = True
    dev = get_device()
    up = lambda t, dt: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=dt, device=dev,  # noqa: E731
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).double()  # noqa: E731
    res = {}
    for spec in a.shapes.split(","):
        for acc_dt in (ttnn.float32, ttnn.bfloat16):
            shape = [int(v) for v in spec.split("x")]
            torch.manual_seed(0)
            acc, last = up(torch.randn(shape), acc_dt), up(torch.randn(shape) * 0.3, ttnn.bfloat16)
            assert R.widen_eligible(acc, last)
            ref = down(acc) + down(last)
            fns = {"widen_cast": lambda: ttnn.typecast(R.widen_add(acc, last), ttnn.bfloat16),
                   "round_add": lambda: R.round_add(acc, last)}
            out, ys = {}, {}
            for arm, fn in fns.items():
                ys[arm] = down(fn())
                out[arm] = {"rel_l2": float((ys[arm] - ref).norm() / ref.norm()), "t": []}
            rec = {"equal": bool(torch.equal(ys["widen_cast"], ys["round_add"])),
                   "differ_frac": float((ys["widen_cast"] != ys["round_add"]).double().mean())}
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
            rec.update(out, window_utc=time.time())
            key = f"{spec} acc={'f32' if acc_dt == ttnn.float32 else 'bf16'}"
            res[key] = rec
            print(json.dumps({key: rec}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
