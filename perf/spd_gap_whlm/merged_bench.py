"""The LM attention output layout: shsdpa into [B, H, S, D] then nlp_concat_heads (old) against shsdpa writing
[B, 1, S, H * D] itself (esmc._LM_SDPA_MERGED). Per shape: whether the two outputs are bit-identical and the median
device ms of each arm (arms alternate per repeat, so host drift lands on both).

    python3 perf/spd_gap_whlm/merged_bench.py --out merged.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed

SHAPES = {"300m": (7, 15), "600m": (7, 18), "6b": (1, 40), "saprot650m": (7, 20)}   # batch, heads; head dim 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=544)
    ap.add_argument("--valid", type=int, default=514)
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import esmc, shsdpa
    dev = T.get_device()
    S, D = args.seq, 64
    out = open(args.out, "w")
    for name, (B, H) in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        q, k, v = (up(torch.randn(B, H, S, D, generator=g)) for _ in range(3))
        pad = esmc.KeyPad(*shsdpa.key_pad([args.valid - 3 * b for b in range(B)], S, device=dev))

        def run(merged):
            esmc._LM_SDPA_MERGED = merged
            return esmc.lm_sdpa(q, k, v, pad, D ** -0.5, merge=True)

        a, b = run(False), run(True)
        rec = {"shape": name, "bhsd": [B, H, S, D], "out_shape": list(b.shape),
               "bit_identical": bool(torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)))}
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ms = {"concat": [], "merged": []}
        for _ in range(args.repeat):
            for arm, m in (("concat", False), ("merged", True)):
                ms[arm].append(timed(lambda m=m: ttnn.deallocate(run(m)), dev, 1, args.inner))
        for arm, ts in ms.items():
            rec[arm + "_ms"] = sorted(ts)[len(ts) // 2]
        rec["saved_ms"] = round(rec["concat_ms"] - rec["merged_ms"], 4)
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
