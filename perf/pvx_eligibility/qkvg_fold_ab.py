#!/usr/bin/env python3
"""Interleaved single-process fold A/B for the fused qkv+gate block entries at c_z=256.

The arm is two keys in `_MM_BLOCK`, and `_mm_block_for` reads that dict on every call with no
cache in front of it, so an arm is a dict mutation between folds: no reload, no second device
open, and both arms see the same weights and the same MSA cache. That is what makes this an
interleaved A/B rather than the two-process screen it replaces.

`off` is forced explicitly on every arm, so an `off` leg provably runs without the entries rather
than inheriting the previous leg's dict. QKVG_STATS is read per fold: an `on` leg that reports 0
served is an A/A pair wearing an A/B label, and only the counter says which happened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))

# The arm, as a `--keys` name -> the entries it adds. `c256` is the fused qkv+gate pair at
# protenix-v2's trunk width; `c64` is the same fusion plus the plain qkv at its template width,
# which shares no key with openfold3's 12-tile qkv.
KEYSETS = {
    "c256": {(8, 32): (4, 8, 1, 4, 1), (8, 33): (4, 8, 1, 4, 1)},
    "c64": {(2, 6): (4, 2, 1, 4, 1), (2, 8): (4, 2, 1, 4, 1), (2, 9): (4, 2, 1, 4, 1)},
    # Every key this branch adds. `off` is then exactly `origin/main` and `on` is exactly what
    # merging delivers, which is the arm the merge decision actually needs.
    "all": {(8, 32): (4, 8, 1, 4, 1), (8, 33): (4, 8, 1, 4, 1),
            (2, 6): (4, 2, 1, 4, 1), (2, 8): (4, 2, 1, 4, 1), (2, 9): (4, 2, 1, 4, 1)},
}
KEYS = KEYSETS["c256"]


def main() -> int:
    global KEYS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keys", default="c256", choices=sorted(KEYSETS))
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT)
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_qkv as HM
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

    KEYS = KEYSETS[a.keys]
    for k in KEYS:
        assert k in T._MM_BLOCK, f"{k} missing -- this tree does not carry the arm under test"

    def set_arm(name):
        for k, v in KEYS.items():
            if name == "on":
                T._MM_BLOCK[k] = v
            else:
                T._MM_BLOCK.pop(k, None)
        return {str(k): list(T._MM_BLOCK[k]) if k in T._MM_BLOCK else None for k in KEYS}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "board": "p150a",
           "arms_order": a.arms, "keyset": a.keys, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "runs": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)

    set_arm("off")
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxel_{a.model}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    a.out.write_text(json.dumps(res, indent=1))

    print("=== cold fold (off) ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s", flush=True)
    res["cold_s"] = round(cold_s, 3)

    for i, arm in enumerate(a.arms.split(",")):
        table = set_arm(arm)
        HM.STATS[0] = HM.STATS[1] = 0
        HM.QKVG_STATS[0] = HM.QKVG_STATS[1] = 0
        HM.QKVGB_STATS[0] = HM.QKVGB_STATS[1] = 0
        HM.QKVG_REJECTS.clear()
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(struct_dir.glob("*")) if p.is_file()}
        rec = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "mm_block": table, "qkvg_stats": list(HM.QKVG_STATS),
               "qkvgb_stats": list(HM.QKVGB_STATS),
               "qkvg_rejects": {f"{k[0]}:{k[1]}": v for k, v in HM.QKVG_REJECTS.items()},
               "aiclk": clk.summary(), "clock_line": clk.line(0),
               "cif_sha256": digest, "loadavg": os.getloadavg()}
        # Vacuity is decided by the arms DIFFERING, not by an absolute: with the c256 entries
        # already in the tree a `c64` off-leg still serves 1048 calls, and an absolute test would
        # call that vacuous. What must hold is that flipping the keyset moves the served count.
        rec["qkv_stats"] = list(HM.STATS)
        prev = [r for r in res["runs"] if r["arm"] != arm]
        rec["vacuous"] = bool(prev) and prev[-1]["qkvg_stats"][0] == HM.QKVG_STATS[0] \
            and prev[-1]["qkv_stats"][0] == HM.STATS[0]
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}: {fold_s:.2f}s  qkvg served/declined {HM.QKVG_STATS}  "
              f"{'VACUOUS' if rec['vacuous'] else ''}  {clk.line(0)}", flush=True)

    by = {}
    for r in res["runs"]:
        by.setdefault(r["arm"], []).append(r["fold_s"])
    summ = {k: {"n": len(v), "median": round(st.median(v), 3),
                "spread": round(max(v) - min(v), 3), "folds": v} for k, v in by.items()}
    if "on" in summ and "off" in summ:
        summ["ratio_off_over_on"] = round(summ["off"]["median"] / summ["on"]["median"], 4)
        summ["delta_s"] = round(summ["off"]["median"] - summ["on"]["median"], 3)
        summ["aa_floor_s"] = max(summ["on"]["spread"], summ["off"]["spread"])
    shas = {r["arm"]: r["cif_sha256"] for r in res["runs"]}
    summ["digest_identical"] = len({json.dumps(s, sort_keys=True) for s in
                                    (r["cif_sha256"] for r in res["runs"])}) == 1
    res["summary"] = summ
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
