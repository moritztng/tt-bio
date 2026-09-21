#!/usr/bin/env python3
"""Interleaved fold A/B for the derived fused `_MM_BLOCK` entry, with a control arm that is main.

This is a corrected fork of `perf/allm_gates/fused_key_ab.py`. That harness sets
`_mm_fused_block` to a None stub for its `off` arm and calls that arm `origin/main`. It is not:
the candidate (ce7467175) DELETED six fused literals from `_MM_BLOCK`, so under the stub those six
resolve to None as well -- configs main serves today. Counted on the pass-1 census, `off` wrongly
declines 1120 calls a fold on boltz2, 3264 on boltzgen, 2416 on protenix-v2, 848 on openfold3, 320
on opendde and 40 on rf3. Only esmfold2 is clean. Scoring `on` against that arm credits the
derivation with restoring configs main already has.

The `main` arm here restores the six literals into `_MM_BLOCK` AND disables derivation, so the
dict is main's fifteen keys and the fallback is dead -- that is main, by construction. The `cand`
arm pops them and restores the shipped `_mm_fused_block`, which is the candidate as written.

Arms alternate inside one process on one device open, so compile and warmup bias cannot land on
one arm. Same-arm legs give the A/A floor. The CIF digest is taken every fold.

  python3 perf/allm_safety/fused_key_ab_fixed.py --model opendde --arms main,cand,main,cand \
      --out perf/allm_safety/out/ab_opendde_512.json
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

# The six fused literals ce7467175 deleted, with the values they carried on main. The `main` arm
# puts these back; without them the control is not main.
DELETED_LITERALS = {
    (4, 16): (4, 4, 1, 4, 1), (4, 17): (4, 4, 1, 4, 1),
    (8, 32): (4, 8, 1, 4, 1), (8, 33): (4, 8, 1, 4, 1),
    (2, 8): (4, 2, 1, 4, 1), (2, 9): (4, 2, 1, 4, 1),
}


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
    ap.add_argument("--arms", default="main,cand,main,cand")
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
    import tt_baseline as B
    import clocksample
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)

    # The candidate must actually be in the tree, or both arms are main and the A/B is vacuous.
    assert hasattr(T, "_mm_fused_block"), (
        "tree has no _mm_fused_block -- check out the candidate before running this")
    SHIPPED = T._mm_fused_block

    # Derivation must reproduce every deleted literal, or the `main` arm and the `cand` arm differ
    # on keys the candidate never meant to touch. Proven offline; asserted here in-process too.
    for k, v in DELETED_LITERALS.items():
        T._MM_BLOCK.pop(k, None)
    for k, v in DELETED_LITERALS.items():
        got = SHIPPED(*k)
        assert got == v, f"derivation for {k} gives {got}, main's literal was {v}"
    T._MM_FUSED_STATS[0] = T._MM_FUSED_STATS[1] = 0
    T._MM_FUSED_DERIVED.clear()

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

    def no_derive(kt, nt):
        T._MM_FUSED_STATS[1] += 1
        return None

    def set_arm(name):
        assert name in ("main", "cand"), name
        if name == "main":
            T._MM_BLOCK.update(DELETED_LITERALS)   # main's fifteen keys ...
            T._mm_fused_block = no_derive          # ... and no derivation. That is main.
        else:
            for k in DELETED_LITERALS:
                T._MM_BLOCK.pop(k, None)
            T._mm_fused_block = SHIPPED
        T._MM_FUSED_STATS[0] = T._MM_FUSED_STATS[1] = 0
        T._MM_FUSED_DERIVED.clear()

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arms": a.arms,
           "control": "main = six deleted literals restored + derivation disabled",
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

    # Cold on the FIRST arm of the list, not always `cand`. `--arms main,main,main` must be a
    # process that never runs the candidate at all: the candidate's cold fold throws L1 and latches
    # `_L1_OUT_REFUSED` for the whole process, so a cand-cold contaminates every later main leg.
    cold_arm = a.arms.split(",")[0]
    set_arm(cold_arm)
    print(f"=== {a.model} {a.size}: cold fold (arm {cold_arm}) ===", flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        before = FC.snapshot() if hasattr(FC, "snapshot") else None
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        d = FC.delta(before, FC.snapshot()) if before is not None else {}
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0),
               "digest": digest(struct_dir),
               "fused_stats": list(T._MM_FUSED_STATS),
               "fused_derived": dict(T._MM_FUSED_DERIVED),
               "qkvg": d.get("triatt_qkv.QKVG_STATS"), "qkvgb": d.get("triatt_qkv.QKVGB_STATS")}
        # keep the structure of every leg, so accuracy can be scored off-line per arm
        keep = a.out.parent / f"cif_{a.model}_{a.size}_leg{i}_{arm}"
        keep.mkdir(parents=True, exist_ok=True)
        for f in sorted(struct_dir.glob("**/*")):
            if f.is_file() and f.suffix in (".cif", ".pdb"):
                (keep / f.name).write_bytes(f.read_bytes())
        leg["cif_dir"] = str(keep)
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  leg {i} {arm:4s}: {fold_s:8.3f}s  fused={leg['fused_stats']} "
              f"derived={leg['fused_derived']} {leg['digest'][:12]}  {leg['clock_line']}",
              flush=True)

    by = {}
    for leg in res["legs"]:
        by.setdefault(leg["arm"], []).append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    res["aa_floor"] = {k: (max(v) - min(v)) for k, v in by.items()}
    res["digests"] = {k: sorted({leg["digest"] for leg in res["legs"] if leg["arm"] == k})
                      for k in by}
    if "main" in res["medians"] and "cand" in res["medians"]:
        res["ratio"] = round(res["medians"]["main"] / res["medians"]["cand"], 5)
        res["delta_s"] = round(res["medians"]["main"] - res["medians"]["cand"], 3)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("medians", "aa_floor", "digests", "ratio", "delta_s")
                      if k in res}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
