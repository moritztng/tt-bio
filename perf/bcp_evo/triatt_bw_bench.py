#!/usr/bin/env python3
"""Raw `triatt_bw.run`, one arm per value of a module attribute (`--lever`), on fixed random q, k,
v, bias and cotangent at the triangle attention's shape [lead, H, N, 32].

Reports dq, dk, dv and dbias torch.equal and max |diff| against the first arm, then each arm timed
synced, alternating every rep. With no `--lever` it is one arm: the driver for a device-profiler
run with TT_BIO_TRIATT_BW_ZONES=1 (per-phase zones, `--lead 27 --reps 0` so each core logs one row).
lead_sum_bench.py is the template.
"""
import argparse, json, pathlib, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="288,128,256")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lead", type=int, default=0, help="leading axis, default N")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--lever", default="", help="triatt_bw attribute to arm off/on")
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_evo/out/triatt_bw_bench.json"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import triatt_bw as TB
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    up = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,  # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = lambda t: torch.Tensor(ttnn.to_torch(t)).float()  # noqa: E731
    arms = ("off", "on") if a.lever else ("on",)
    res = {}
    for n in (int(s) for s in a.ns.split(",")):
        torch.manual_seed(0)
        q, k, v, g = (up(torch.randn(a.lead or n, a.heads, n, 32)) for _ in range(4))
        bias = up(torch.randn(1, a.heads, n, n))

        def call(arm):
            if a.lever:
                setattr(TB, a.lever, arm == "on")
            return TB.run(dev, q, k, v, bias, g, 32 ** -0.5, (ttnn.MathFidelity.HiFi4,))
        out = {"lever": a.lever}
        ref, t = None, {arm: [] for arm in arms}
        for arm in arms:
            r = [down(x) for x in call(arm)]
            if ref is None:
                ref = r
            else:
                out["equal"] = [bool(torch.equal(x, y)) for x, y in zip(ref, r)]
                out["max_abs_diff"] = [float((x - y).abs().max()) for x, y in zip(ref, r)]
        for rep in range(a.reps):
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                r = call(arm)
                ttnn.synchronize_device(dev)
                t[arm].append(time.perf_counter() - t0)
                for x in r:
                    ttnn.deallocate(x)
        for arm in arms:
            ts = sorted(t[arm])
            out[arm] = {"median_ms": 1e3 * ts[len(ts) // 2], "min_ms": 1e3 * ts[0]}
        out["window_utc"] = time.time()
        res[n] = out
        print(json.dumps({n: out}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
