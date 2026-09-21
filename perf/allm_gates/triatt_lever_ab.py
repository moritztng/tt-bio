#!/usr/bin/env python3
"""Interleaved fold A/B for the two shipped-but-OFF triangle-attention levers (ledger M38).

`TT_BIO_TRIATT_FUSE_QKV` folds the qkv projection into the SDPA kernel; `TT_BIO_TRIATT_GATE_EPILOGUE`
folds `o * sigmoid(g)` into its pack stage. Both default False, so the `off` arm IS `origin/main`.

Arms alternate inside one process on one device open, so compile and warmup bias cannot land on one
arm; a cold fold is discarded first. Every leg counts the lever's own firing -- GATE_STATS,
triatt_sdpa.STATS and FUSE_REJECTS -- because a lever that does not fire looks exactly like a lever
that does not work, and a leg whose counters do not move with its arm makes the A/B vacuous.

  python3 perf/allm_gates/triatt_lever_ab.py --model boltz2 --arms off,qkv,off,qkv,off,qkv \
      --out perf/allm_gates/ab_triatt_boltz2_512.json
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
for p in (ROOT, ROOT / "scripts" / "gpu_vs_tt", ROOT / "perf", ROOT / "perf" / "pvx_eligibility"):
    sys.path.insert(0, str(p))


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
    ap.add_argument("--arms", default="off,qkv,off,qkv,off,qkv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.triatt_sdpa as TS
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

    # Both flags default False on main, so `off` is main exactly -- no reconstruction needed.
    assert TS._FUSE_QKV is False and TS._GATE_EPILOGUE is False, (
        f"a lever is already on from the environment: fuse_qkv={TS._FUSE_QKV} "
        f"gate={TS._GATE_EPILOGUE}; the off arm would not be main")
    ARMS = {"off": (False, False), "qkv": (True, False), "gate": (False, True),
            "both": (True, True)}

    def set_arm(name):
        TS._FUSE_QKV, TS._GATE_EPILOGUE = ARMS[name]
        TS.GATE_STATS[0] = TS.GATE_STATS[1] = 0
        TS.STATS[0] = TS.STATS[1] = 0
        TS.GATE_REJECTS.clear()
        TS.FUSE_REJECTS.clear()

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

    set_arm(a.arms.split(",")[1] if "," in a.arms else "off")
    print(f"=== {a.model} {a.size}: cold fold ===", flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0),
               "digest": digest(struct_dir),
               "gate_stats": list(TS.GATE_STATS), "sdpa_stats": list(TS.STATS),
               "gate_rejects": {f"{k[0]}{list(k[1])}": v for k, v in TS.GATE_REJECTS.items()},
               "fuse_rejects": dict(TS.FUSE_REJECTS)}
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  leg {i} {arm:4s}: {fold_s:8.3f}s  gate={leg['gate_stats']} "
              f"sdpa={leg['sdpa_stats']} fuse_rej={leg['fuse_rejects']} "
              f"{leg['digest'][:12]}  {leg['clock_line']}", flush=True)

    by = {}
    for leg in res["legs"]:
        by.setdefault(leg["arm"], []).append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    res["aa_floor"] = {k: (max(v) - min(v)) for k, v in by.items()}
    res["digests"] = {k: sorted({leg["digest"] for leg in res["legs"] if leg["arm"] == k})
                      for k in by}
    for k in by:
        if k != "off" and "off" in res["medians"]:
            res[f"ratio_{k}"] = round(res["medians"]["off"] / res["medians"][k], 5)
            res[f"delta_s_{k}"] = round(res["medians"]["off"] - res["medians"][k], 3)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k.startswith(("ratio", "delta")) or k in
                      ("medians", "aa_floor", "digests")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
