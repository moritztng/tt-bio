#!/usr/bin/env python3
"""Triangle-attention SDPA at [512, H, 512, 32] bf16: shipped k256 (two k chunks, online rescale)
against k512 (one k chunk, no rescale), offered by `_SDPA_WIDE_K_UP` now that the unused ping-pong
CBs are not allocated at one k chunk. Arms interleaved in one process; each arm's output graded
against a float64 reference on the first `--ref-rows` batch rows, and the route each arm served read
back from `SDPA_CHUNK_PICKS` so a silent fall-back cannot pass as a result.
"""
import argparse, json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, D = 512, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--ref-rows", type=int, default=8)
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/widek_512.json")
    args = ap.parse_args()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    H = args.heads

    def dram(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    qh, kh, vh = (torch.randn(S, H, S, D).to(torch.bfloat16) for _ in range(3))
    bh = torch.randn(1, H, S, S).to(torch.bfloat16)
    q, k, v, bias = dram(qh), dram(kh), dram(vh), dram(bh)
    scale = D ** -0.5
    r = args.ref_rows
    # The fused path scales the additive mask with the scores, softmax((q k^T + mask) * scale), as
    # perf/spd_ditsdpa/opbench.py and perf/lpx_sdpa/bench.py grade it.
    refs = {"f64": torch.softmax((qh[:r].double() @ kh[:r].double().transpose(-1, -2) + bh.double()) * scale, -1)
            @ vh[:r].double()}
    outs = {}

    arms = {"k256": False, "k512": True}
    res = {"meta": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "shape": [S, H, S, D],
                    "grid": list(T.COMPUTE_GRID_MAIN), "loadavg": os.getloadavg(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, "arms": {}}

    def run(wide):
        T._SDPA_WIDE_K_UP = wide
        o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        assert o is not None
        return o

    for name, wide in arms.items():
        o = run(wide)
        out = ttnn.to_torch(o)[:r].double()
        ttnn.deallocate(o)
        outs[name] = out
        res["arms"][name] = {"pick": list(T.SDPA_CHUNK_PICKS.get((S, S), [])), "ms": [], **{
            f"rel_rms_{n}": ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            for n, ref in refs.items()}}
    d = outs["k512"] - outs["k256"]
    res["k512_vs_k256"] = {"max_abs": d.abs().max().item(),
                           "rel_rms": (d.pow(2).mean().sqrt() / outs["k256"].pow(2).mean().sqrt()).item()}
    for _ in range(args.reps):
        for name, wide in arms.items():
            run(wide)  # warm the program cache for this arm before its timed call
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = run(wide)
            ttnn.synchronize_device(dev)
            res["arms"][name]["ms"].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    for a in res["arms"].values():
        a["med_ms"] = statistics.median(a["ms"])
    res["speedup"] = res["arms"]["k256"]["med_ms"] / res["arms"]["k512"]["med_ms"]
    res["meta"]["aiclk_end"] = sorted({p.read_text().strip() for p in pathlib.Path("/sys/class/tenstorrent").glob("*/tt_aiclk")})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    for n, a in res["arms"].items():
        print(n, a["pick"], f"{a['med_ms']:.3f} ms", {k: round(x, 5) for k, x in a.items() if k.startswith("rel")})
    print("k512 vs k256", res["k512_vs_k256"], "speedup", round(res["speedup"], 4))


if __name__ == "__main__":
    main()
