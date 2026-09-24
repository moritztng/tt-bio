#!/usr/bin/env python3
"""bcx-heads: the two head-verb backwards the unified fix has to choose between, head to head.

`nlp_concat_heads`' backward is a head split, ``[B, 1, L, H*dh] -> [B, H, L, dh]``:

  transpose  wk/of3t-cropwall: transpose(-2,-1), reshape [B,H,dh,L], transpose(-2,-1). 3 ops.
  slice      wk/bcx-stack: reshape [B,L,H*dh], H tile-aligned last-axis slices, one concat. H+1.
  main       main's reshape [B,L,H,dh] + permute, for scale only.

and `nlp_create_qkv_heads`' backward builds its rows with `nlp_concat_heads`, which wk/bcx-stack
pins to DRAM_MEMORY_CONFIG and wk/of3t-cropwall does not (`rows_pin` / `rows_nopin`).

Every form is checked bit for bit against the torch permutation of the same bf16 tensor before it
is timed. One process, one card, forms interleaved rep by rep, AICLK from the card's own sysfs
node sampled inside each timed window, loadavg at start and end of each shape.
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock, sysfs_node  # noqa: E402

# (label, B, L, H, dh): AF2's triangle attention at 128/256, OF3's at its SL arm's 384 and its
# 576 wall, OF3's 16-head attention-pair-bias single track, and 8/16 heads on the pair track to
# show how each form scales with H.
SHAPES = [("af2_tri_n128", 128, 128, 4, 32), ("af2_tri_n256", 256, 256, 4, 32),
          ("of3_tri_n384", 384, 384, 4, 32), ("of3_tri_n576", 576, 576, 4, 32),
          ("of3_apb_n384_h16", 1, 384, 16, 32), ("of3_apb_n576_h16", 1, 576, 16, 32),
          ("af2_msa_row_s128_h8", 128, 256, 8, 32),
          ("pair_n256_h8", 256, 256, 8, 32), ("pair_n256_h16", 256, 256, 16, 32),
          ("pair_n256_h4_dh16", 256, 256, 4, 16),
          # correctness only: widths and lengths that are not whole tiles
          ("odd_L100_h4_dh24", 3, 100, 4, 24), ("odd_L77_h2_dh32", 2, 77, 2, 32)]


def run(ttnn, dev, fn, reps_inner):
    t0 = time.perf_counter()
    for _ in range(reps_inner):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps_inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-time", dest="time", action="store_false",
                    help="bit-exactness only, no timed reps")
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    dev = tt.get_device()
    clock = Clock()
    TILE, DRAM = ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG

    def slice_form(x, H):              # wk/bcx-stack's `_split_heads_v`, verbatim
        B, S, C = (int(d) for d in x.shape)
        d = C // H
        if d % 32:
            return ttnn.permute(ttnn.reshape(x, [B, S, H, d]), (0, 2, 1, 3))
        return ttnn.concat([ttnn.reshape(ttnn.slice(x, [0, 0, h * d], [B, S, (h + 1) * d]),
                                         [B, 1, S, d]) for h in range(H)], dim=1)

    out = {"host": os.uname().nodename, "pci": sysfs_node()[1], "aiclk_node": clock.path,
           "visible": os.environ.get("TT_VISIBLE_DEVICES"), "shapes": {}}
    for label, B, L, H, dh in SHAPES:
        if args.only and label not in args.only.split(","):
            continue
        torch.manual_seed(0)
        gt = torch.randn(B, 1, L, H * dh).to(torch.bfloat16)
        ref = gt.reshape(B, L, H, dh).permute(0, 2, 1, 3).contiguous()   # [B, H, L, dh]
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=DRAM)
        h = ttnn.from_torch(ref, dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=DRAM)
        forms = {
            "transpose": lambda: ttnn.transpose(
                ttnn.reshape(ttnn.transpose(g, -2, -1), [B, H, dh, L]), -2, -1),
            "slice": lambda: slice_form(ttnn.reshape(g, [B, L, H * dh]), H),
            "shipped": lambda: ag.split_heads_value(g, H),
            "rows_shipped": lambda: ag.merge_heads_value(h),
            "main": lambda: ttnn.permute(ttnn.reshape(g, [B, L, H, dh]), (0, 2, 1, 3)),
            "rows_nopin": lambda: ttnn.experimental.nlp_concat_heads(h),
            "rows_pin": lambda: ttnn.experimental.nlp_concat_heads(h, memory_config=DRAM),
        }
        want = {"transpose": ref, "slice": ref, "main": ref, "shipped": ref,
                "rows_nopin": gt, "rows_pin": gt, "rows_shipped": gt}
        load0 = os.getloadavg()[0]
        res = {"B": B, "L": L, "H": H, "dh": dh, "forms": {}}
        live = {}
        for name, fn in forms.items():
            print(" ", label, name, flush=True)
            try:
                y = fn()
                mc = str(y.memory_config().buffer_type)
                yt = ttnn.to_torch(y).reshape(want[name].shape)
                res["forms"][name] = {"bitexact": bool(torch.equal(yt, want[name])),
                                      "out_buffer": mc}
                for _ in range(2):
                    fn()
                ttnn.synchronize_device(dev)
                live[name] = fn
            except Exception as e:  # a form that cannot run at a shape is a result, not a crash
                res["forms"][name] = {"error": str(e).splitlines()[0][:300]}
        if not args.time:
            live = {}
        inner = {}
        for name, fn in live.items():
            t = run(ttnn, dev, fn, 3)
            inner[name] = max(3, int(0.05 / max(t, 1e-6)))
        ts = {n: [] for n in live}
        spans = []
        names = list(live)
        for r in range(args.reps):
            for name in (names if r % 2 == 0 else names[::-1]):
                a = time.time()
                ts[name].append(run(ttnn, dev, live[name], inner[name]) * 1e6)
                spans.append((a, time.time()))
        for name in live:
            xs = sorted(ts[name])
            res["forms"][name].update(us_median=statistics.median(xs), us_p10=xs[len(xs) // 10],
                                      us_p90=xs[(9 * len(xs)) // 10], inner=inner[name],
                                      reps=len(xs))
        res["aiclk"] = clock.window(spans)
        res["load1"] = [load0, os.getloadavg()[0]]
        out["shapes"][label] = res
        f = res["forms"]
        print(label, {k: (round(v["us_median"], 1) if "us_median" in v else v.get("error"),
                          v.get("bitexact")) for k, v in f.items()},
              res["aiclk"], res["load1"], flush=True)
        for t in (g, h):
            ttnn.deallocate(t)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))   # a hang keeps the rest
    clock.stop()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
