#!/usr/bin/env python3
"""Boltz-2 512 aa cell A/B of the three trunk byte levers, paired and interleaved ABBA.

Arm `off` pins `triatt_qkv._QKVG_ENABLED`, `triatt_qkv._QKVGB_ENABLED` and
`tenstorrent._TRIMUL_FUSED_GOUT` to False, which is what main did before this branch. Arm `on` is
this branch's default. All three are read at CALL time, and the concatenated weights are built at
construction regardless of the flag, so both arms run out of one process against one loaded model
and the only thing that changes between them is which matmul the call site asks for.

All three are byte-identical transforms, so the bar is a matching CIF sha256 and an unchanged
complex plDDT, not an RMSD. The ratio is the median of the paired ABBA deltas; the A/A floor comes
out of the same run, off-first against off-last.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, which is the fold the
rest of this campaign times. 200 sampling steps, 3 recycles, full MSA: no work is removed.
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
ORDER = ["off", "on", "on", "off"]          # ABBA, so box drift cancels inside a rep
PINS = ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB", "TT_BIO_TRIMUL_FUSED_GOUT")
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
    import tt_bio.triatt_qkv as TQ
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    for p in PINS:
        assert p not in os.environ, f"{p} is pinned; the arms are set in-process"
    assert TQ.TRIATT_FUSED_QKVG and TQ.TRIATT_FUSED_QKVGB and TT.TRIMUL_FUSED_GOUT, \
        "this tree does not default all three levers on"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-trunkship-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-trunk-byte-round2-ship", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str) -> dict:
        want = arm == "on"
        TQ._QKVG_ENABLED = TQ._QKVGB_ENABLED = want
        TT._TRIMUL_FUSED_GOUT = want
        for s in (TQ.QKVG_STATS, TQ.QKVGB_STATS, TT.TRIMUL_GOUT_STATS):
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
                "qkvg": list(TQ.QKVG_STATS), "qkvgb": list(TQ.QKVGB_STATS),
                "gout": list(TT.TRIMUL_GOUT_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s qkvg={r['qkvg']} qkvgb={r['qkvgb']} "
              f"gout={r['gout']} cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in ORDER:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r['fold_s']:7.3f}s qkvg={r['qkvg']} qkvgb={r['qkvgb']} "
                  f"gout={r['gout']} plddt={r['plddt']} load={r['loadavg1']} "
                  f"cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in ("off", "on")}
    rep: dict = {}
    for r in timed:
        rep.setdefault(r["rep"], []).append(r)
    ab, aa = [], []
    for i in sorted(rep):
        g = rep[i]
        assert [x["arm"] for x in g] == ORDER, [x["arm"] for x in g]
        off, on = [g[0]["fold_s"], g[3]["fold_s"]], [g[1]["fold_s"], g[2]["fold_s"]]
        ab += [round(o / n, 5) for o, n in zip(off, on)]
        aa += [round(max(off) / min(off), 5), round(max(on) / min(on), 5)]
    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["median_fold_s"] = med
    OUT["ratio_median"] = round(med["off"] / med["on"], 5)
    OUT["ab_paired_ratios"] = ab
    OUT["ab_paired_median"] = round(st.median(ab), 5)
    OUT["ab_paired_all_positive"] = all(x > 1 for x in ab)
    OUT["aa_floor_max"] = max(aa) if aa else None
    OUT["aa_ratios"] = aa
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a})
                    for a in ("off", "on")}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    OUT["served"] = {
        "off": {k: [sum(x[k][j] for x in timed if x["arm"] == "off") for j in (0, 1)]
                for k in ("qkvg", "qkvgb", "gout")},
        "on": {k: [sum(x[k][j] for x in timed if x["arm"] == "on") for j in (0, 1)]
               for k in ("qkvg", "qkvgb", "gout")}}
    dump()
    keys = ("median_fold_s", "ratio_median", "ab_paired_median", "ab_paired_ratios",
            "ab_paired_all_positive", "aa_floor_max", "plddt", "bit_exact", "cif_sha256",
            "served")
    print(json.dumps({k: OUT[k] for k in keys if k in OUT}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
