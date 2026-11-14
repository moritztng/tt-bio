#!/usr/bin/env python3
"""Ceiling for a masked streaming triangle-attention SDPA at [512, H, 512, 32] bf16.

Stock ttnn SDPA takes the streaming compute path (`compute_streaming.hpp` v2, no row buffers) only
without a mask and without fp32_dest_acc. Triangle attention always has a mask, so it never gets it.
This times the stock op in four arms per (q_chunk, k_chunk): mask / no mask x fp32_dest_acc off / on.
fp32_dest_acc on forces the non-streaming path, so

    streaming gain ~= (nomask_acc0 / nomask_acc1) / (mask_acc0 / mask_acc1)

separates what streaming buys from what dropping the mask buys. The shipped fused path
(`_tri_att_sdpa_at`) is timed alongside as the bar a masked streaming kernel must beat.
Arms interleaved in one process, warm call before each timed call.
"""
import argparse, json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, D = 512, 32
CHUNKS = ((256, 256), (512, 256), (512, 512))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/stream_512.json")
    args = ap.parse_args()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    H = args.heads
    grid = T.COMPUTE_GRID_MAIN

    def dram(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    q, k, v = (dram(torch.randn(S, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def stock(qc, kc, mask, acc):
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                               fp32_dest_acc_en=acc, packer_l1_acc=False)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                                    q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
        return lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=bias if mask else None, scale=scale,
            program_config=pc, compute_kernel_config=ckc)

    arms = {"shipped": lambda: T._tri_att_sdpa_at(q, k, v, bias, scale)}
    for qc, kc in CHUNKS:
        for mask in (True, False):
            for acc in (False, True):
                arms[f"q{qc}k{kc}_{'mask' if mask else 'nomask'}_acc{int(acc)}"] = stock(qc, kc, mask, acc)

    res = {"meta": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "shape": [S, H, S, D], "grid": list(grid),
                    "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
           "arms": {n: {"ms": []} for n in arms}}
    live = dict(arms)
    for name, fn in arms.items():  # first call: compile, and drop arms the op refuses (L1)
        try:
            ttnn.deallocate(fn())
        except Exception as exc:  # noqa: BLE001 -- recorded, the arm is dropped
            res["arms"][name]["error"] = str(exc).splitlines()[0][:300]
            del live[name]
    for _ in range(args.reps):
        for name, fn in live.items():
            ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            res["arms"][name]["ms"].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    med = {n: statistics.median(a["ms"]) for n, a in res["arms"].items() if a["ms"]}
    for n, m in med.items():
        res["arms"][n]["med_ms"] = m
    res["streaming_gain"] = {}
    for qc, kc in CHUNKS:
        p = f"q{qc}k{kc}_"
        try:
            res["streaming_gain"][p[:-1]] = ((med[p + "nomask_acc1"] / med[p + "nomask_acc0"])
                                             / (med[p + "mask_acc1"] / med[p + "mask_acc0"]))
        except KeyError:
            pass
    res["meta"]["aiclk_end"] = sorted({p.read_text().strip() for p in pathlib.Path("/sys/class/tenstorrent").glob("*/tt_aiclk")})
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    for n, a in res["arms"].items():
        print(n, f"{a['med_ms']:.3f} ms" if "med_ms" in a else a.get("error"))
    print("streaming gain", {k: round(x, 3) for k, x in res["streaming_gain"].items()})


if __name__ == "__main__":
    main()
