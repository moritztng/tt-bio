"""OpenDDE's bf16 token-DiT linears at the partner short bucket: ttnn.linear (shipped) vs k1_linear vs minimal_matmul blocks.

dfid (.107 dfid/out.json, 1000 MHz) showed these linears are not math-bound: HiFi4 -> HiFi2 saves ~5 % (qkv
[2880,768]@[768,3072] 489 -> 462 us), so ttnn.linear's schedule is the limit (~55 GB/s). dit_step_mm fixes that for
Protenix-v2's fp32 / bfp8 DiT inside DitLowp.linear, which a bf16 DiT without dit_b8 (OpenDDE, both modes) never
reaches. This sweeps the candidates at OpenDDE's shapes, HiFi4 + fp32 dest, bf16 in and out.
K block stays 1: fp32 dest accumulation of more than one K tile is wrong on ~1e-7 of Wormhole outputs (k1_linear).

Per arm: median us per call (NCALL back-to-back after one sync, REPS rounds, arms interleaved), float64 rel_rms, and
max |out - float64| over the 99.9th-percentile error of the shipped arm (a wrong element reads >> 1).

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/dit_mm_bench.py OUT.json [--rows 2880]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rows", type=int, default=5 * 576)
ap.add_argument("--ncall", type=int, default=20); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
# name: (K, N, bias) -- qkv, the s-term / transition-in linb, transition out, attention gate/out
SHAPES = {"qkv": (768, 3072, True), "linb": (768, 1536, True), "trans_out": (1536, 768, False),
          "gate_out": (768, 768, False)}
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
gx, gy = T.COMPUTE_GRID_MAIN


def mm_arm(m, n, sh, sw):
    cfg = ttnn.MinimalMatmulConfig(M_block_size=m, K_block_size=1, N_block_size=n, subblock_h=sh, subblock_w=sw,
                                   compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))
    return lambda x, w, b: ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, compute_kernel_config=CKC,
                                                            dtype=ttnn.bfloat16, config=cfg)


ARMS = {"ship": lambda x, w, b: ttnn.linear(x, w, bias=b, compute_kernel_config=CKC, core_grid=T.CORE_GRID_MAIN),
        "k1": lambda x, w, b: T.k1_linear(x, w, b, compute_kernel_config=CKC, dtype=ttnn.bfloat16)}
for m in (2, 4, 8):
    for n in (4, 8, 16):
        for sh, sw in ((1, 4), (2, 2), (1, 2), (2, 1), (4, 1)):
            if m % sh == 0 and n % sw == 0:
                ARMS[f"mm{m}x{n}s{sh}{sw}"] = mm_arm(m, n, sh, sw)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = {"rows": a.rows, "shapes": {}}
for name, (K, N, has_b) in SHAPES.items():
    torch.manual_seed(K + N)
    x, w = up(torch.randn(1, a.rows, K)), up(torch.randn(K, N) * K ** -0.5)
    b = up((torch.randn(N) * 0.1).reshape(1, N)) if has_b else None
    ref = ttnn.to_torch(x)[0].double() @ ttnn.to_torch(w).double()
    if b is not None:
        ref = ref + ttnn.to_torch(b).double()[0]
    acc, live = {}, {}
    for arm, f in ARMS.items():
        try:
            d = (ttnn.to_torch(f(x, w, b))[0].double() - ref).abs()
        except Exception as e:  # a block the op refuses at this shape
            acc[arm] = {"error": str(e).splitlines()[0][:160]}
            continue
        acc[arm] = {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), "max": float(d.max()),
                    "p999": float(d.flatten().kthvalue(int(d.numel() * 0.999)).values)}
        live[arm] = f
    for arm in live:
        acc[arm]["max_over_ship_p999"] = acc[arm]["max"] / acc["ship"]["p999"]
    t = {arm: [] for arm in live}
    for _ in range(a.reps):
        for arm, f in live.items():
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            for _ in range(a.ncall):
                ttnn.deallocate(f(x, w, b))
            ttnn.synchronize_device(dev); t[arm].append((time.perf_counter() - t0) / a.ncall * 1e6)
    for arm in live:
        acc[arm]["us"] = statistics.median(t[arm])
    best = min(live, key=lambda k: acc[k]["us"])
    res["shapes"][name] = {"K": K, "N": N, "best": best, "arms": acc}
    print(name, "ship", round(acc["ship"]["us"], 1), "best", best, round(acc[best]["us"], 1), acc[best], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
