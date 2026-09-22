#!/usr/bin/env python3
"""Interleaved fold A/B for the derived fused `_MM_BLOCK` entry, with an A/A floor.

The `off` arm makes `_mm_fused_block` return None, which is exactly what `origin/main` does with an
absent key, so `off` IS main and `on` IS what merging delivers. Arms alternate inside one process,
one device open, so compile and warmup bias cannot land on one arm
(`op-ab-must-interleave-arms-compile-warmup-bias`). Same-arm legs give the A/A floor the ratio is
scored against, and the CIF digest is taken every fold because bit-exactness is the cheapest
regression signal there is.

  python3 perf/allm_gates/fused_key_ab.py --model opendde --arms off,on,off,on,off,on \\
      --out perf/allm_gates/ab_opendde_512.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(ROOT / "perf" / "pvx_eligibility"))

import firing_census as FC  # noqa: E402
sys.path.insert(0, str(ROOT / "perf" / "allm_gates"))
import gate_census as GC  # noqa: E402  the same counter snapshot the census took


def digest(struct_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(struct_dir.glob("**/*")):
        if f.is_file() and f.suffix in (".cif", ".pdb"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    SHIPPED = T._mm_fused_block

    # The baseline arm is origin/main's TABLE, not "no derivation at all". `_MM_BLOCK` carried six
    # fused literals before this change, and (2, 9) is one of them -- opendde presents it 320 times
    # a fold. An arm that returns None for every key therefore measures the derivation PLUS the
    # removal of a config main already had, which overstates what merging delivers. The first run
    # of this harness did exactly that and read 1.01975x; these are the values read out of
    # `git show origin/main:tt_bio/tenstorrent.py`, so `main` serves what main serves and nothing
    # else. `off` is kept for the separate question of what the whole fused table is worth.
    MAIN_FUSED = {(4, 16): (4, 4, 1, 4, 1), (4, 17): (4, 4, 1, 4, 1),
                  (8, 32): (4, 8, 1, 4, 1), (8, 33): (4, 8, 1, 4, 1),
                  (2, 8): (4, 2, 1, 4, 1), (2, 9): (4, 2, 1, 4, 1)}

    def _count(kt, nt, blk):
        if blk is None:
            T._MM_FUSED_STATS[1] += 1
            return None
        T._MM_FUSED_STATS[0] += 1
        k = f"kt={kt},nt={nt}"
        T._MM_FUSED_DERIVED[k] = T._MM_FUSED_DERIVED.get(k, 0) + 1
        return blk

    def main_arm(kt, nt):
        return _count(kt, nt, MAIN_FUSED.get((kt, nt)))

    def off_arm(kt, nt):
        return _count(kt, nt, None)

    ARM_FN = {"on": SHIPPED, "main": main_arm, "off": off_arm}

    def set_arm(name):
        T._mm_fused_block = ARM_FN[name]
        T._MM_FUSED_STATS[0] = T._MM_FUSED_STATS[1] = 0
        T._MM_FUSED_DERIVED.clear()

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arms": a.arms,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "legs": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_allmgates_{a.model}_{a.size}",
                                         tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res["grid"] = list(T.COMPUTE_GRID_MAIN)

    set_arm("on")
    print(f"=== {a.model} {a.size}: cold fold (arm on) ===", flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        before = GC.snapshot()
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        after = GC.snapshot()
        d = FC.delta(before, after)
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0),
               "digest": digest(struct_dir),
               "fused_stats": list(T._MM_FUSED_STATS),
               "fused_derived": dict(T._MM_FUSED_DERIVED),
               "qkvg": d.get("triatt_qkv.QKVG_STATS"), "qkvgb": d.get("triatt_qkv.QKVGB_STATS"),
               "qkvg_rejects": d.get("triatt_qkv.QKVG_REJECTS")}
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  leg {i} {arm:3s}: {fold_s:8.3f}s  qkvg={leg['qkvg']} "
              f"fused={leg['fused_stats']} {leg['digest'][:12]}  {leg['clock_line']}", flush=True)

    # A leg whose counters do not move with its arm makes the A/B vacuous, so say it here.
    by = {}
    for leg in res["legs"]:
        by.setdefault(leg["arm"], []).append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    res["aa_floor"] = {k: (max(v) - min(v)) for k, v in by.items()}
    res["digests"] = {k: sorted({leg["digest"] for leg in res["legs"] if leg["arm"] == k})
                      for k in by}
    base = "main" if "main" in res["medians"] else "off"
    if base in res["medians"] and "on" in res["medians"]:
        res["baseline_arm"] = base
        res["ratio"] = round(res["medians"][base] / res["medians"]["on"], 5)
        res["delta_s"] = round(res["medians"][base] - res["medians"]["on"], 3)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("medians", "aa_floor", "digests", "ratio", "delta_s")
                      if k in res}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
