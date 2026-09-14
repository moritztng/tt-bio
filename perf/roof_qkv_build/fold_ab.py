#!/usr/bin/env python3
"""The fold-level reading: the qkv->SDPA fold on and off, paired, same seed, same process.

The op-level ratio is in `pair.py`. This is the number the campaign's 0.60 A bar is defined
against, plus the wall-clock the fold actually gets. Every CIF is kept under its own name, so the
same run gives the A/A floor (two folds of one arm) and the A/B deviation -- on a card with a known
matmul nondeterminism (`pc-card0-512aa-fold-nondeterminism`) that floor is the only thing that makes
the arm number readable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH is not None:
        OUT_PATH.write_text(json.dumps(OUT, indent=2) + "\n")


def patch_cfg():
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()


def run_size(T, B, TS, size, pairs, cifdir):
    B.RECYCLING_STEPS = B._resolve_recycling_steps(None, "boltz2")
    patch_cfg()
    T.get_device()
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, _state = B.build_fold(
        "boltz2", HERE / f".fq_{size}", fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")
    rec = {"hardware": meta["hardware"], "grid": meta.get("grid"),
           "card_type": meta.get("card_type"), "recycling_steps": meta["recycling_steps"],
           "struct_dir": meta["struct_dir"], "folds": []}
    OUT.setdefault("sizes", {})[str(size)] = rec
    cifdir.mkdir(parents=True, exist_ok=True)

    def fold(arm, tag):
        TS._FUSE_QKV = (arm == "1")
        TS.FUSE_REJECTS.clear()
        served0 = TS.STATS[0]
        t, m = one_fold()
        cifs = {}
        for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
            cifs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            shutil.copy2(f, cifdir / f"{size}_{tag}_{arm}_{f.name}")
        r = {"arm": arm, "tag": tag, "s": round(t, 3), "plddt": m.get("plddt"),
             "cif_sha256": cifs, "fuse_rejects": dict(TS.FUSE_REJECTS),
             "stock_fused_sdpa_calls": TS.STATS[0] - served0}
        rec["folds"].append(r)
        pl = r["plddt"]
        print(f"  {size} {tag:8s} fuse={arm}  {t:7.3f} s  plddt "
              f"{pl if pl is None else round(float(pl), 4)}  rejects {r['fuse_rejects']} "
              f"stock_sdpa {r['stock_fused_sdpa_calls']}", flush=True)
        dump()
        return t

    try:
        rec["cold_s"] = {"off": round(fold("0", "cold"), 3), "on": round(fold("1", "cold"), 3)}
        arms = {"0": [], "1": []}
        for i in range(pairs):
            for arm in (("0", "1") if i % 2 == 0 else ("1", "0")):
                arms[arm].append(fold(arm, f"warm{i}"))
        rec["warm_s"] = {k: [round(x, 3) for x in v] for k, v in arms.items()}
        if pairs:
            rec["warm_median_s"] = {k: round(st.median(v), 3) for k, v in arms.items()}
            rec["speedup"] = round(rec["warm_median_s"]["0"] / rec["warm_median_s"]["1"], 4)
            print(f"  {size} MEDIAN off {rec['warm_median_s']['0']:.3f} s  on "
                  f"{rec['warm_median_s']['1']:.3f} s  ->  {rec['speedup']:.4f}x", flush=True)
    finally:
        TS._FUSE_QKV = False
    dump()


def main():
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,298")
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "fold_ab.json"))
    ap.add_argument("--cifdir", default=str(HERE / "cif"))
    a = ap.parse_args()
    OUT_PATH = Path(a.out)

    import tt_bio.tenstorrent as T
    from tt_bio import triatt_sdpa as TS
    import importlib
    B = importlib.import_module("tt_baseline")
    OUT["host"] = __import__("os").uname().nodename
    for size in [int(s) for s in a.sizes.split(",")]:
        run_size(T, B, TS, size, a.pairs, Path(a.cifdir))
    dump()
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
