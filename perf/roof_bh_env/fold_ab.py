#!/usr/bin/env python3
"""Fold-level cost of the fp32-softmax tuned rectangle on the Galaxy chip. Paired, ABBA.

Arm `off` pins the fitted `_FP32_SOFTMAX_L1_GRID = (8, 8)`, arm `on` is this branch's default,
the live 8x9 grid. Both arms run in ONE process with the plan caches cleared between them, so
neither inherits the other's block.

`BOLTZ2_FP32_SOFTMAX=1` is required and is the PATH, not the arm: Boltz-2 ships with the fused
SDPA instead, so this fold prices the constant on a model whose weights are on the box rather
than on OpenFold3 or AF2-IG, which reach `_fp32_softmax_attention` by default and whose weights
are not. Both arms carry the flag, so the only thing that differs between them is the rectangle.

The change only moves a memory config, so the acceptance is a matching CIF sha256 across every
fold of both arms. ABBA inside a rep so a drifting box cancels rather than accumulating into
whichever arm goes second.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, statistics as st
import sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)

FIX = REPO / "perf" / "size512" / "fixtures"
ORDER = ["off", "on", "on", "off"]
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    args = ap.parse_args()
    OUT_PATH = args.out
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert "TT_BIO_FP32_SOFTMAX_L1_LIVE_GRID" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"
    assert TT._FP32_SOFTMAX, "BOLTZ2_FP32_SOFTMAX=1 is the path under test, not the arm"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="roof-bhenv-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-bh-envelopes-on-wh", cfg)

    live = (dev.compute_with_storage_grid_size().y, dev.compute_with_storage_grid_size().x)
    OUT["grids"] = {"off": [8, 8], "on": list(live)}

    def fold(arm: str) -> dict:
        TT._FP32_SOFTMAX_L1_GRID = live if arm == "on" else (8, 8)
        TT._FP32_SOFTMAX_L1_ROW_CAP.clear()
        TT._FP32_SOFTMAX_L1_FREE_ROW_CAP.clear()
        TT._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
        TT._fp32_softmax_l1_plan.cache_clear()
        TT._fp32_softmax_core_grid.cache_clear()
        for key in TT.FP32_SOFTMAX_STATS:
            TT.FP32_SOFTMAX_STATS[key] = 0
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        return {"arm": arm, "fold_s": round(wall, 3),
                "grid": list(TT._FP32_SOFTMAX_L1_GRID),
                "gated_calls": [TT.FP32_SOFTMAX_STATS["l1_blocks"],
                                TT.FP32_SOFTMAX_STATS["blocks"]],
                "fp32_softmax": dict(TT.FP32_SOFTMAX_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s {r['grid']} l1blk={r['gated_calls']} "
              f"cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in ORDER:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r['fold_s']:7.3f}s gated={r['gated_calls']} "
                  f"plddt={r['plddt']} load={r['loadavg1']} cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in ("off", "on")}
    offs = [r["fold_s"] for r in timed if r["arm"] == "off"]
    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["n_per_arm"] = {a: sum(r["arm"] == a for r in timed) for a in ("off", "on")}
    OUT["median_fold_s"] = med
    OUT["ratio"] = med["off"] / med["on"]
    OUT["aa_floor"] = (min(offs) / max(offs)) if len(offs) > 1 else None
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["paired_deltas_s"] = [round(o - n, 3) for o, n in
                              zip([r["fold_s"] for r in timed if r["arm"] == "off"],
                                  [r["fold_s"] for r in timed if r["arm"] == "on"])]
    OUT["gated_calls"] = {a: [list(c) for c in
                              sorted({tuple(r["gated_calls"]) for r in timed if r["arm"] == a})]
                          for a in ("off", "on")}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    dump()
    keys = ("n_per_arm", "median_fold_s", "ratio", "aa_floor", "plddt", "paired_deltas_s",
            "gated_calls", "bit_exact", "cif_sha256")
    print(json.dumps({k: OUT[k] for k in keys if k in OUT}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
