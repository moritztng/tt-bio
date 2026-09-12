#!/usr/bin/env python3
"""Price the AdaLN s-norm fold on one settled diffusion step, both arms in ONE process.

The fold is a LOAD-TIME weight change, so the two arms cannot be separated by an env var across
two processes without comparing across two model builds. They can be separated inside one build:
the folded and unfolded projection weights are both derivable from the same host checkpoint, so
this harness holds both on device and rebinds three attributes per AdaLN between arms. Nothing
else about the model, the fixture or the grabbed call changes.

  base   `s_terms` = layer_norm(s, weight=s_norm.weight) @ W        (what main ships)
  fold   `s_terms` = s_normed @ (diag(s_norm.weight) W), with `s_normed` computed once by
         DiffusionTransformer for the whole 24-layer stack

Arms are interleaved block by block with a base at both ends, per the campaign's rule 3.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()

    # The model must be BUILT with the fold on, so every AdaLN loads folded weights; the harness
    # rebuilds the unfolded pair from the same host tensors for the base arm.
    os.environ["BOLTZ2_ADALN_S_NORM_FOLD"] = "1"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import step_probe as SP
    import tt_baseline as B

    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "size": a.size,
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()}
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    fence = SP.make_fence(ttnn, dev)
    diff = g["obj"]

    # ---- collect the AdaLNs of the token stack and build both weight sets --------------
    adalns = []
    for layer in diff.token_transformer.layers:
        adalns += [layer.adaln, layer.transition.adaln]
    arms_w = []
    for ad in adalns:
        w = ad.weights["s_norm.weight"].float()
        plain = {k: ttnn.from_torch(ad.weights[f"{k}.weight"].t(), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.bfloat16)
                 for k in ("s_scale", "s_bias")}
        arms_w.append({"ad": ad, "w": w,
                       "base": (plain["s_scale"], plain["s_bias"]),
                       "fold": (ad.s_scale_weight, ad.s_bias_weight)})
    SP.OUT["n_adaln_token"] = len(adalns)

    def set_arm(name):
        T._B2_ADALN_S_NORM_FOLD = (name == "fold")
        for e in arms_w:
            e["ad"].s_norm_folded = (name == "fold")
            e["ad"].s_scale_weight, e["ad"].s_bias_weight = e[name]

    # ---- correctness first: the fold must change the answer only at bf16 level ---------
    set_arm("base")
    ref = ttnn.to_torch(diff(*g["args"], **g["kwargs"])).float()
    set_arm("base")
    ref2 = ttnn.to_torch(diff(*g["args"], **g["kwargs"])).float()
    set_arm("fold")
    got = ttnn.to_torch(diff(*g["args"], **g["kwargs"])).float()
    den = ref.abs().mean().item()
    SP.OUT["step_output"] = {
        "base_self_bit_exact": bool(torch.equal(ref, ref2)),
        "fold_bit_exact_vs_base": bool(torch.equal(ref, got)),
        "max_abs": float((ref - got).abs().max()),
        "rel_mean": float((ref - got).abs().mean() / den),
        "scale_mean_abs": den}
    print("step_output", json.dumps(SP.OUT["step_output"]), flush=True)
    SP.dump()

    # ---- program counts, one graph capture per arm -------------------------------------
    from itemize import top_level_spans
    counts = {}
    for name in ("base", "fold"):
        set_arm(name)
        diff(*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        diff(*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
        ops, _ = top_level_spans(ttnn.graph.end_graph_capture())
        counts[name] = len(ops)
    SP.OUT["ttnn_top_level_calls"] = counts
    print("ttnn calls", counts, flush=True)
    SP.dump()

    # ---- interleaved timing, base at both ends -----------------------------------------
    order = ["base", "fold", "base"]
    walls = {k: [] for k in ("base", "fold")}
    for name in order:
        set_arm(name)
        for _ in range(3):
            diff(*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
    for blk in range(a.blocks):
        for name in order:
            set_arm(name)
            fence()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                diff(*g["args"], **g["kwargs"])
            ttnn.synchronize_device(dev)
            walls[name].append((time.perf_counter() - t0) / a.reps * 1e3)
        print(f"  block {blk}: base {walls['base'][-2]:.4f}/{walls['base'][-1]:.4f} "
              f"fold {walls['fold'][-1]:.4f} ms", flush=True)
        SP.OUT["step_ms"] = {k: {"median": round(st.median(v), 4),
                                 "spread_pct": round(100 * (max(v) - min(v)) / st.median(v), 3),
                                 "all": [round(x, 4) for x in v]} for k, v in walls.items()}
        b = walls["base"]
        SP.OUT["ratio"] = round(st.median(b) / st.median(walls["fold"]), 5)
        SP.OUT["aa_floor"] = round(st.median(b[0::2]) / st.median(b[1::2]), 5)
        SP.dump()
    print("RATIO", SP.OUT["ratio"], "A/A floor", SP.OUT["aa_floor"], flush=True)
    print("step_ms", json.dumps(SP.OUT["step_ms"]), flush=True)
    SP.dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
