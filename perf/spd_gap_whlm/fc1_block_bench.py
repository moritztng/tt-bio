"""Kill gate for a row-blocked fc12g on the ESMC 300M FFN: ESMC's fc1 (both halves, [3808, 960] x [960, 5120], bf16)
as one minimal_matmul over every row (today's program) against ttnn.linear over 1024-row blocks (the stock matmul a
fused fc12g kernel would sit on). If the blocked stock path is more than ~15 % slower, fusing the gate into it is
net zero. Median device ms of each, summed over blocks.

    python3 perf/spd_gap_whlm/fc1_block_bench.py --out fc1_block.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed

ROWS, D, N, BLOCK = 3808, 960, 5120, 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    g = torch.Generator().manual_seed(0)
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    x = up(torch.randn(1, ROWS, D, generator=g))
    w = up(torch.randn(D, N, generator=g) * D ** -0.5)
    blocks = [ttnn.slice(x, (0, r, 0), (1, min(r + BLOCK, ROWS), D)) for r in range(0, ROWS, BLOCK)]

    def mm():
        return [ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                 **esmc._mm_config(x, w))]

    def blocked():
        return [ttnn.linear(b, w, compute_kernel_config=ck) for b in blocks]

    rec = {"rows": ROWS, "k": D, "n": N, "block": BLOCK}
    ms = {"minimal_matmul": [], "linear_blocked": []}
    for _ in range(args.repeat):
        for arm, fn in (("minimal_matmul", mm), ("linear_blocked", blocked)):
            ms[arm].append(timed(lambda fn=fn: [ttnn.deallocate(t) for t in fn()], dev, 1, args.inner))
    for arm, ts in ms.items():
        rec[arm + "_ms"] = sorted(ts)[len(ts) // 2]
    rec["tflops_mm"] = round(2 * ROWS * D * N / rec["minimal_matmul_ms"] / 1e9, 1)
    rec["blocked_over_mm"] = round(rec["linear_blocked_ms"] / rec["minimal_matmul_ms"], 3)
    print(json.dumps(rec), flush=True)
    open(args.out, "w").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
