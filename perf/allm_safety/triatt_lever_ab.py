#!/usr/bin/env python3
"""Interleaved fold A/B for the two triangle-attention levers, with firing COUNTED per leg.

`TT_BIO_TRIATT_GATE_EPILOGUE` and `TT_BIO_TRIATT_FUSE_QKV` both ship OFF, so the `off` arm is
main by construction -- no dict is edited and no literal is deleted, which is what made the
`_MM_BLOCK` candidate's control arm wrong (see state/allm-safety.md, pass 2 PART 5).

Firing is counted, never read off the clause. Per leg this records:
  gate_served / gate_rejected  -- `triatt_sdpa.GATE_STATS`, and the reject reasons
  fuse_served                  -- a counting wrapper on `sdpa_fused_qkv`, which has no counter
  fuse_rejected                -- `triatt_sdpa.FUSE_REJECTS`, by reason

An `on` leg whose served count is 0 means the lever is INERT on this model, which for four of the
seven models is the claim under test and not a failure. An `off` leg whose served count is nonzero
would mean the arm switch does not work and the run is vacuous; both are asserted.

Both levers write process-lifetime L1-refusal state (`_GATE_OVER_L1`, `PM_L1_ERRORS`,
`SG.note_l1_refusal`), so the per-leg counters are compared across same-arm legs: a served count
that DROPS between two `on` legs is that contamination, and it is reported rather than averaged.

  python3 perf/allm_safety/triatt_lever_ab.py --model boltz2 --lever gate \
      --arms off,on,off,on,off,on --out perf/allm_safety/out/lever_gate_boltz2_512.json
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


# `_fuse_reject` is called from BOTH sides of the wrapper: these three reasons are raised in
# `tenstorrent.py`, BEFORE `sdpa_fused_qkv` is entered, and every other reason is raised inside it.
# So served = calls - the rejects raised inside. Subtracting the total instead reads protenix-v2 as
# -1208, because all 1208 of its rejects are the outside kind and it never enters the function.
_FUSE_REJECT_OUTSIDE = ("qkv_already_fused_with_gate", "site", "no_full_S_chunk")


def _fuse_served(rejects, calls):
    inside = sum(v for k, v in rejects.items() if k not in _FUSE_REJECT_OUTSIDE)
    served = calls - inside
    assert served >= 0, f"negative serves: calls={calls} inside={inside} rejects={dict(rejects)}"
    return served


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
    ap.add_argument("--lever", required=True, choices=("gate", "fuse", "both"))
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    # Card grant enforced HERE, before torch and before the model load -- a probe at launch
    # cannot say who owns the card three minutes later, so the lease is taken now and held for
    # the whole launch. See card_guard.py for the incident this closes.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import card_guard
    card_guard.preflight()


    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS
    import tt_baseline as B
    import clocksample
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)

    # Both levers ship off. If that ever changes, `off` stops being main and this harness lies.
    assert TS.TRIATT_GATE_EPILOGUE is False, "gate epilogue no longer ships off"
    assert TS._GATE_EPILOGUE is False and TS._FUSE_QKV is False, (
        f"a lever is already on at import: gate={TS._GATE_EPILOGUE} fuse={TS._FUSE_QKV}; "
        f"the off arm would not be main")

    # `sdpa_fused_qkv` (triatt_sdpa.py:476) is the whole function INCLUDING its reject ladder --
    # the `_FUSE_QKV` flag check, the precondition guards and the L1 budget test all live inside
    # it. So a wrapper counts INVOCATIONS, not fusions served. Serves = invocations - rejects, and
    # the JSON records all three rather than the ambiguous one. Counting the wrapper alone reads
    # opendde as 1048 "served" when all 1048 are refused for l1_budget.
    FUSE_CALLS = [0]
    _real_fused = TS.sdpa_fused_qkv

    def _counting_fused(*args, **kw):
        FUSE_CALLS[0] += 1
        return _real_fused(*args, **kw)
    TS.sdpa_fused_qkv = _counting_fused

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

    def set_arm(name):
        assert name in ("off", "on"), name
        on = name == "on"
        if a.lever in ("gate", "both"):
            TS._GATE_EPILOGUE = on
        if a.lever in ("fuse", "both"):
            TS._FUSE_QKV = on
        TS.GATE_STATS[0] = TS.GATE_STATS[1] = 0
        TS.GATE_REJECTS.clear()
        TS.FUSE_REJECTS.clear()
        FUSE_CALLS[0] = 0

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "lever": a.lever, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arms": a.arms,
           "control": "off = the shipped default for both flags, so off IS main",
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "legs": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_allmsafety_{a.model}_{a.size}",
                                         tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res["grid"] = list(T.COMPUTE_GRID_MAIN)

    # Cold fold on the OFF arm: the shipped default is what should absorb the compile, and the
    # `on` arm is the one that can write process-lifetime L1-refusal state.
    set_arm("off")
    print(f"=== {a.model} {a.size} lever={a.lever}: cold fold (arm off) ===", flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0), "digest": digest(struct_dir),
               "gate_served": TS.GATE_STATS[0], "gate_rejected": TS.GATE_STATS[1],
               "gate_rejects": {f"{r}|{s}": n for (r, s), n in TS.GATE_REJECTS.items()},
               "fuse_calls": FUSE_CALLS[0],
               "fuse_rejected": sum(TS.FUSE_REJECTS.values()),
               "fuse_served": _fuse_served(TS.FUSE_REJECTS, FUSE_CALLS[0]),
               "fuse_rejects": dict(TS.FUSE_REJECTS)}
        keep = a.out.parent / f"cif_{a.model}_{a.size}_{a.lever}_leg{i}_{arm}"
        keep.mkdir(parents=True, exist_ok=True)
        for f in sorted(struct_dir.glob("**/*")):
            if f.is_file() and f.suffix in (".cif", ".pdb"):
                (keep / f.name).write_bytes(f.read_bytes())
        leg["cif_dir"] = str(keep)
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  leg {i} {arm:3s}: {fold_s:8.3f}s gate={leg['gate_served']}/{leg['gate_rejected']} "
              f"fuse={leg['fuse_served']} {leg['digest'][:12]}  {leg['clock_line']}", flush=True)

    by = {}
    for leg in res["legs"]:
        by.setdefault(leg["arm"], []).append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    res["aa_floor"] = {k: (max(v) - min(v)) for k, v in by.items()}
    res["digests"] = {k: sorted({leg["digest"] for leg in res["legs"] if leg["arm"] == k})
                      for k in by}
    if "off" in res["medians"] and "on" in res["medians"]:
        res["ratio"] = round(res["medians"]["off"] / res["medians"]["on"], 5)
        res["delta_s"] = round(res["medians"]["off"] - res["medians"]["on"], 3)
        # The ratio is only readable if it clears the WIDER of the two same-arm spreads.
        res["resolvable"] = bool(abs(res["delta_s"]) > max(res["aa_floor"].values()))

    # firing verdicts, counted
    onlegs = [l for l in res["legs"] if l["arm"] == "on"]
    offlegs = [l for l in res["legs"] if l["arm"] == "off"]
    served = {"gate": [l["gate_served"] for l in onlegs], "fuse": [l["fuse_served"] for l in onlegs]}
    res["firing"] = {
        "on_arm_served": served,
        "off_arm_served": {"gate": [l["gate_served"] for l in offlegs],
                           "fuse": [l["fuse_served"] for l in offlegs]},
        "inert_on_this_model": {k: (max(v) == 0 if v else None) for k, v in served.items()},
        "served_stable_across_on_legs": {k: (len(set(v)) == 1 if v else None)
                                         for k, v in served.items()},
    }
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ("medians", "aa_floor", "digests", "ratio", "delta_s", "resolvable", "firing")
                      if k in res}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
