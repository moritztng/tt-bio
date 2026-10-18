"""The protein LMs' residual add where no matmul absorbs it (ESMC-6B at its 544-row page shape, SaProt): ttnn.add on
binary_ng (today) against the legacy binary op. Per shape: whether the legacy output is bit-identical
to today's, and each arm's median device ms.

    python3 perf/spd_gap_whlm/add_bench.py --out add.jsonl
"""
import argparse
import json

import torch
import ttnn

from qkfuse_bench import timed

SHAPES = {"6b": (1, 544, 2560), "saprot650m": (7, 544, 1280)}   # SaProt 1.3B has the 650M width


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=9)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    out = open(args.out, "w")
    for name, shape in SHAPES.items():
        g = torch.Generator().manual_seed(0)
        up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        a, b = up(4 * torch.randn(*shape, generator=g)), up(torch.randn(*shape, generator=g))
        arms = {
            "ng": lambda: ttnn.add(a, b),
            "legacy": lambda: ttnn.add(a, b, use_legacy=True),
        }
        ref = ttnn.to_torch(arms["ng"]())
        rec = {"shape": name, "dims": list(shape)}
        for arm, fn in arms.items():
            try:
                rec[arm + "_exact"] = bool(torch.equal(ttnn.to_torch(fn()), ref))
            except Exception as exc:  # noqa: BLE001
                rec[arm + "_error"] = str(exc)[:300]
        ms = {k: [] for k in arms}
        for _ in range(args.repeat):
            for arm, fn in arms.items():
                if arm + "_error" not in rec:
                    ms[arm].append(timed(lambda fn=fn: ttnn.deallocate(fn()), dev, 1, args.inner))
        for arm, ts in ms.items():
            if ts:
                rec[arm + "_ms"] = sorted(ts)[len(ts) // 2]
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
