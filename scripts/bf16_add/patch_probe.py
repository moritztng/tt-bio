"""Does the repo-wide patch actually make ``ttnn.add``/``ttnn.add_`` agree with torch, bit for bit?

Three arms on the same operands: stock ttnn, the patched op, and torch. Reports the element
mismatch against torch for each, for both the out-of-place and the in-place call, and for the
in-place call also checks the MUTATED operand and not just the return value -- the patch has to
fix both or it fires silently at call sites that ignore the return.

    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/bf16_add/patch_probe.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rne_add  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402

SHAPES = ((1, 128, 128, 128), (1, 256, 256, 128), (1, 1, 512, 384))
RATIOS = (1.0, 0.1, 0.01)


def main() -> None:
    device = get_device()
    rows = []
    for shape in SHAPES:
        for ratio in RATIOS:
            g = torch.Generator().manual_seed(0)
            a = torch.randn(*shape, generator=g).to(torch.bfloat16)
            b = (ratio * torch.randn(*shape, generator=g)).to(torch.bfloat16)
            want = a + b

            def up(t):
                return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device,
                                       dtype=ttnn.bfloat16)

            def miss(got):
                got = torch.Tensor(ttnn.to_torch(got)).to(torch.bfloat16)
                return int((got != want).sum())

            n = want.numel()
            row = {"shape": list(shape), "ratio": ratio, "elements": n}
            rne_add.uninstall()
            row["stock_add"] = miss(ttnn.add(up(a), up(b)))
            row["stock_add_"] = miss(ttnn.add_(up(a), up(b)))
            rne_add.install()
            row["rne_add"] = miss(ttnn.add(up(a), up(b)))
            at = up(a)
            ret = ttnn.add_(at, up(b))
            row["rne_add_"] = miss(ret)
            row["rne_add__mutated_operand"] = miss(at)
            rows.append(row)
            print("%-22s ratio %-5s stock add %8.4f%% add_ %8.4f%% | rne add %d add_ %d operand %d of %d"
                  % (shape, ratio, 100 * row["stock_add"] / n, 100 * row["stock_add_"] / n,
                     row["rne_add"], row["rne_add_"], row["rne_add__mutated_operand"], n),
                  flush=True)
    total = sum(r["elements"] for r in rows)
    bad = sum(r["rne_add"] + r["rne_add_"] + r["rne_add__mutated_operand"] for r in rows)
    stock = sum(r["stock_add"] + r["stock_add_"] for r in rows)
    print("\nstock disagrees with torch on %d/%d = %.4f%%" % (stock, 2 * total, 100 * stock / (2 * total)))
    print("patched disagrees with torch on %d/%d" % (bad, 3 * total))
    print(rne_add.report())
    out = os.environ.get("PATCH_PROBE_JSON")
    if out:
        with open(out, "w") as fh:
            json.dump({"rows": rows, "stock_mismatch": stock, "patched_mismatch": bad,
                       "elements_per_arm": total}, fh, indent=2)
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
