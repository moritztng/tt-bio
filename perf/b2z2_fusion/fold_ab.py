#!/usr/bin/env python3
"""The 512 aa Boltz-2 fold A/B for the two Transition levers, on the cell.

Chunk ratios do not survive to the fold on their own -- three wave-2 levers were real, bit-exact and
smaller than the noise. So both levers get measured where the claim is made, with the block wall
recorded next to the fold wall so a null on the fold can be told apart from a lever that never fired.

  base    main's defaults: silu fused into ttnn.linear, no fused SwiGLU kernel.
  swiglu  tt_bio/transition_swiglu.py ON. Same arithmetic in one kernel, PCC 0.9999983 on the chunk.
  usilu   TT_BIO_UNFUSED_SILU's path: ttnn.linear with no activation, then ttnn.silu. Already in
          main, default off, release-gated. PCC 0.9999949 on the chunk.

Every arm sets BOTH flags explicitly, so no arm can inherit the previous one's state. `base` is run
at two positions in every rep, first and last, so a drift across the rep is visible rather than
folded into the ratio. The cold fold is discarded.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
STATE = {"dev": None}
SERVED = [0, 0]
BLOCK_KEY = "block:PairformerLayer"


def timed_call(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="base,swiglu,usilu,base")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--grid", default="11x8", help="core grid for the fused kernel, or 'main'")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.transition_swiglu as TS
    import tt_baseline as B
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    import fold_ab_multi as FAM

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        FAM.patch_boltz2_cfg()

    TS.GRID = None if a.grid == "main" else tuple(int(v) for v in a.grid.split("x"))

    # Count what the kernel actually served. A lever reporting 0 served is UNTESTED, not null.
    orig_fs = TS.fused_swiglu

    def counted(*args, **kw):
        out = orig_fs(*args, **kw)
        SERVED[0 if out is not None else 1] += 1
        return out

    TS.fused_swiglu = counted

    installed = []
    for cls, key in ((getattr(T, "PairformerLayer", None), BLOCK_KEY),
                     (getattr(T, "Transition", None), "body:Transition")):
        if cls is None:
            continue
        f = cls.__call__
        cls.__call__ = (lambda g, k: lambda self, *x, **kw: timed_call(k, g, self, *x, **kw))(f, key)
        installed.append(key)

    def set_arm(name):
        """Both flags on every arm. Never bool(dict.get(name)) -- that inherits."""
        T._UNFUSED_SILU = (name == "usilu")
        TS.set_enabled(name == "swiglu")
        TS.REJECTS.clear()
        SERVED[0] = SERVED[1] = 0

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    set_arm("base")
    one_fold, meta, state = B.build_fold(a.model, ROOT / (".msa_b2z2fr_%d" % a.size), tgt, a3m)
    STATE["dev"] = T.get_device()
    g = STATE["dev"].compute_with_storage_grid_size()
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "grid": [g.x, g.y], "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
           "fused_grid": list(TS.GRID) if TS.GRID else "main",
           "block_config": {str(k): list(v) for k, v in TS.BLOCK_KEYS.items()},
           "mul_mode": TS.MUL_MODE, "mul_batch": TS.MUL_BATCH, "round": TS.ROUND,
           "timers_installed": installed, "runs": []}

    print("=== %s %d aa rec=%s steps=%s: cold ==="
          % (a.model, a.size, B.RECYCLING_STEPS, B.SAMPLING_STEPS), flush=True)
    cold_s, cold_m = one_fold()
    print("  cold %.2fs n_tokens=%s plddt=%s"
          % (cold_s, cold_m.get("n_tokens"), cold_m.get("plddt")), flush=True)
    res["cold"] = {"fold_s": round(cold_s, 3), "plddt": cold_m.get("plddt"),
                   "n_tokens": cold_m.get("n_tokens")}

    for rep in range(a.reps):
        for pos, arm in enumerate(a.arms.split(",")):
            set_arm(arm)
            WALL.clear()
            fold_s, m = one_fold()
            blk = WALL.get(BLOCK_KEY, {"n": 0, "s": 0.0})
            rec = {"rep": rep, "pos": pos, "arm": arm, "fold_s": round(fold_s, 3),
                   "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
                   "cif_sha256": sha_dir(struct_dir),
                   "swiglu": {"enabled": TS.ENABLED, "served": SERVED[0], "declined": SERVED[1],
                              "rejects": {"%s:%s" % (r, sh): n
                                          for (r, sh), n in TS.REJECTS.items()}},
                   "unfused_silu": T._UNFUSED_SILU,
                   "walls": {k: {"n": v["n"], "s": round(v["s"], 4)} for k, v in WALL.items()}}
            res["runs"].append(rec)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
            print("  rep%d %-7s fold %7.3fs  block %7.3fs/%-4d  plddt %s  swiglu served %d "
                  "declined %d" % (rep, arm, fold_s, blk["s"], blk["n"], m.get("plddt"),
                                   SERVED[0], SERVED[1]), flush=True)

    # ---- ratios, base taken as the median of BOTH its positions ----
    by = defaultdict(list)
    blk = defaultdict(list)
    for r in res["runs"]:
        by[r["arm"]].append(r["fold_s"])
        b = r["walls"].get(BLOCK_KEY)
        if b:
            blk[r["arm"]].append(b["s"])
    base = st.median(by["base"])
    res["summary"] = {}
    for arm, v in sorted(by.items()):
        row = {"n": len(v), "median_fold_s": round(st.median(v), 3),
               "fold_ratio_vs_base": round(base / st.median(v), 5),
               "spread_pct": round(100 * (max(v) - min(v)) / st.median(v), 2)}
        if blk.get(arm) and blk.get("base"):
            row["median_block_s"] = round(st.median(blk[arm]), 3)
            row["block_ratio_vs_base"] = round(st.median(blk["base"]) / st.median(blk[arm]), 5)
        res["summary"][arm] = row
        print("%-8s n=%d fold %.3fs %.5fx  block %ss %sx  spread %.2f%%"
              % (arm, row["n"], row["median_fold_s"], row["fold_ratio_vs_base"],
                 row.get("median_block_s", "-"), row.get("block_ratio_vs_base", "-"),
                 row["spread_pct"]), flush=True)

    # the A/A floor: base at position 0 against base at the last position
    p0 = [r["fold_s"] for r in res["runs"] if r["arm"] == "base" and r["pos"] == 0]
    pn = [r["fold_s"] for r in res["runs"] if r["arm"] == "base" and r["pos"] != 0]
    if p0 and pn:
        res["summary"]["aa_floor"] = round(st.median(p0) / st.median(pn), 5)
        print("A/A floor (base pos0 / base pos-last): %.5fx" % res["summary"]["aa_floor"],
              flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
