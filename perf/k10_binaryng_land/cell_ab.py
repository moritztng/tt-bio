#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B for the two BinaryNg L1 placements, paired and interleaved ABBA.

Arm `off` pins `tenstorrent._TRIMUL_MASK_L1` and `._RESIDUAL_L1` to False, which is what main
does today: the trimul's broadcast mask operand is read from DRAM once per channel block, and the
starting triangle attention and the pair transition write their residual update to DRAM for the
very next op to read back. Arm `on` is this branch's default. `mask` and `resid` run one site at a
time, so the union can be compared against its parts rather than assumed to be their product.

Both flags are read at call time, so an arm is a flag flip between folds: one process, one device
context, one loaded model. A memory config decides which banks a tile lands in and not what is in
it, so the bar is a matching CIF sha256 and an unchanged plDDT, not an RMSD.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, the fold the rest of
this campaign times. 200 sampling steps, 3 recycles, full MSA: no work is removed.
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
PINS = ("TT_BIO_TRIMUL_MASK_L1", "TT_BIO_RESIDUAL_L1")
ARMS = {"off": (False, False), "on": (True, True),
        "mask": (True, False), "resid": (False, True)}
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
    ap.add_argument("--order", default="off,on,on,off",
                    help="one rep's arm order; ABBA cancels box drift inside the rep")
    args = ap.parse_args()
    OUT_PATH = args.out
    order = args.order.split(",")
    assert all(a in ARMS for a in order), order
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
    for p in PINS:
        assert p not in os.environ, f"{p} is pinned; the arms are set in-process"
    assert TT._TRIMUL_MASK_L1 and TT._RESIDUAL_L1, "this tree does not default both levers on"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture, "order": order,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "l1_total_bytes": TT._l1_total_bytes(dev),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="k10-binaryng-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("k10-binaryng-l1-land", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str) -> dict:
        TT._TRIMUL_MASK_L1, TT._RESIDUAL_L1 = ARMS[arm]
        for s in (TT.TRIMUL_MASK_L1_STATS, TT.RESIDUAL_L1_STATS):
            s[0] = s[1] = 0
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
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
                "mask_l1": list(TT.TRIMUL_MASK_L1_STATS),
                "resid_l1": list(TT.RESIDUAL_L1_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in sorted(set(order)):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:5s} {r['fold_s']:7.3f}s mask={r['mask_l1']} resid={r['resid_l1']} "
              f"cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in order:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:5s} {r['fold_s']:7.3f}s mask={r['mask_l1']} "
                  f"resid={r['resid_l1']} plddt={r['plddt']} load={r['loadavg1']} "
                  f"cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    arms = sorted(set(order))
    med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in arms}
    rep: dict = {}
    for r in timed:
        rep.setdefault(r["rep"], []).append(r)
    # Pair each non-baseline arm against the `off` folds that bracket it inside the same rep.
    paired: dict = {a: [] for a in arms if a != "off"}
    aa = []
    for i in sorted(rep):
        g = rep[i]
        offs = [x["fold_s"] for x in g if x["arm"] == "off"]
        if len(offs) > 1:
            aa.append(round(max(offs) / min(offs), 5))
        base = st.median(offs) if offs else None
        for a in paired:
            xs = [x["fold_s"] for x in g if x["arm"] == a]
            if len(xs) > 1:
                aa.append(round(max(xs) / min(xs), 5))
            if base:
                paired[a] += [round(base / x, 5) for x in xs]
    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in arms}
    OUT["median_fold_s"] = med
    OUT["ratio_vs_off"] = {a: round(med["off"] / med[a], 5) for a in arms if a != "off"
                           and "off" in med}
    OUT["ab_paired_ratios"] = paired
    OUT["ab_paired_median"] = {a: round(st.median(v), 5) for a, v in paired.items() if v}
    OUT["ab_paired_all_positive"] = {a: all(x > 1 for x in v) for a, v in paired.items() if v}
    OUT["aa_floor_max"] = max(aa) if aa else None
    OUT["aa_ratios"] = aa
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in arms}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = all(len(shas[a]) == 1 for a in arms) and len({
        shas[a][0] for a in arms}) == 1
    OUT["served"] = {a: {k: [sum(x[k][j] for x in timed if x["arm"] == a) for j in (0, 1)]
                         for k in ("mask_l1", "resid_l1")} for a in arms}
    dump()
    keys = ("median_fold_s", "ratio_vs_off", "ab_paired_median", "ab_paired_ratios",
            "ab_paired_all_positive", "aa_floor_max", "plddt", "bit_exact", "cif_sha256",
            "served")
    print(json.dumps({k: OUT[k] for k in keys if k in OUT}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
