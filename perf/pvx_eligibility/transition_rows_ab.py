#!/usr/bin/env python3
"""Interleaved fold A/B for the Blackhole Transition row raise at Protenix-v2's channel.

`_BH_TRANSITION_L1_ROWS_MAX_C` is read at call time inside `Transition.__call__`, so an arm is a
module-global write between folds: one process, one device open, same weights, same MSA cache.

The raise changes the row-block height, which changes the fc1/fc2 matmul's M, which the tree
already records as NOT bit-exact for a different height constant (LEDGER K9). So every arm's CIFs
are kept rather than only their digests: if the digest moves, the structure has to be scored in
Angstrom against the 0.60 A kill bar with the 1.84 A seed floor beside it, and that needs the
files.

`TRANSITION_H_CHUNK_STATS` is read per fold. An `on` leg that does not move the served count is an
A/A pair wearing an A/B label, and only the counter says which happened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--on-c", type=int, default=256, help="_BH_TRANSITION_L1_ROWS_MAX_C in the on arm")
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT)
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

    SHIPPED = T._BH_TRANSITION_L1_ROWS_MAX_C
    assert SHIPPED == 128, f"this tree ships {SHIPPED}, not 128 -- the off arm is not main"

    def set_arm(name):
        T._BH_TRANSITION_L1_ROWS_MAX_C = a.on_c if name == "on" else SHIPPED
        return T._BH_TRANSITION_L1_ROWS_MAX_C

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "board": "p150a",
           "shipped_max_c": SHIPPED, "on_max_c": a.on_c, "arms_order": a.arms,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "runs": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cifroot = a.out.parent / f"{a.out.stem}_cifs"

    set_arm("off")
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxel_{a.model}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    a.out.write_text(json.dumps(res, indent=1))

    print("=== cold fold (off) ===", flush=True)
    cold_s, _cm = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        maxc = set_arm(arm)
        T.TRANSITION_H_CHUNK_STATS[0] = T.TRANSITION_H_CHUNK_STATS[1] = 0
        T.TRANSITION_H_CHUNK_REJECTS.clear()
        T.TRANSITION_H_CHUNK_SHAPES.clear()
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        err = None
        try:
            with clocksample.during(period=1.0) as clk:
                fold_s, m = one_fold()
        except Exception as e:                                                  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:500]
            fold_s, m = float("nan"), {}
        keep = cifroot / f"{i}_{arm}"
        keep.mkdir(parents=True, exist_ok=True)
        digest = {}
        for p in sorted(struct_dir.glob("*")):
            if p.is_file():
                digest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
                shutil.copy2(p, keep / p.name)
        rec = {"i": i, "arm": arm, "max_c": maxc, "error": err,
               "fold_s": None if err else round(fold_s, 3), "plddt": m.get("plddt"),
               "h_chunk_served": T.TRANSITION_H_CHUNK_STATS[0],
               "h_chunk_declined": T.TRANSITION_H_CHUNK_STATS[1],
               "h_chunk_rejects": {f"{k[0]}:{k[1]}": v for k, v in T.TRANSITION_H_CHUNK_REJECTS.items()},
               "h_chunk_shapes": {f"{k[0]}|hid{k[1]}|h{k[2]}": v
                                  for k, v in T.TRANSITION_H_CHUNK_SHAPES.items()},
               "aiclk": None if err else clk.summary(),
               "clock_line": None if err else clk.line(0),
               "cif_sha256": digest, "cif_dir": str(keep), "loadavg": os.getloadavg()}
        prev = [r for r in res["runs"] if r["arm"] != arm]
        rec["vacuous"] = bool(prev) and prev[-1]["h_chunk_served"] == rec["h_chunk_served"]
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm} (max_c={maxc}): "
              f"{'ERROR ' + err if err else f'{fold_s:.2f}s'}  "
              f"h_chunk served/declined {rec['h_chunk_served']}/{rec['h_chunk_declined']}  "
              f"{'VACUOUS' if rec['vacuous'] else ''}  {rec['clock_line'] or ''}", flush=True)
        if rec["h_chunk_shapes"]:
            for k, v in sorted(rec["h_chunk_shapes"].items()):
                print(f"      {v:5d}  {k}", flush=True)

    ok = [r for r in res["runs"] if r["fold_s"] is not None]
    by = {}
    for r in ok:
        by.setdefault(r["arm"], []).append(r["fold_s"])
    summ = {k: {"n": len(v), "median": round(st.median(v), 3),
                "spread": round(max(v) - min(v), 3), "folds": v} for k, v in by.items()}
    if "on" in summ and "off" in summ:
        summ["ratio_off_over_on"] = round(summ["off"]["median"] / summ["on"]["median"], 4)
        summ["delta_s"] = round(summ["off"]["median"] - summ["on"]["median"], 3)
        summ["aa_floor_s"] = max(summ["on"]["spread"], summ["off"]["spread"])
    summ["digest_identical"] = len({json.dumps(r["cif_sha256"], sort_keys=True)
                                    for r in ok}) == 1
    summ["digest_by_arm"] = {r["arm"]: sorted(r["cif_sha256"].values())[:1] for r in ok}
    res["summary"] = summ
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
