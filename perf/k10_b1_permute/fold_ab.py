#!/usr/bin/env python3
"""Boltz-2 fold A/B of B1, the fused in-projection's column permutation. Paired, ABBA.

Arm `off` pins the shipped role-major order `[g_a, g_b, p_a, p_b]`, arm `on` is this branch's
default `[p_a, g_a, p_b, g_b]`. Both arms run in ONE process and the weight caches are keyed on
the order, so each arm builds its own layout instead of inheriting the other's.

The lever is a permutation, not arithmetic, so the acceptance is a matching CIF sha256 across
every fold of both arms. An RMSD here would be answering a question that is not being asked.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, so this is the same
fold every other ship row on this ladder times. ABBA inside a rep so a drifting box cancels
rather than accumulating into whichever arm goes second.
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
    import tt_bio.reblock_permute as RB
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert "TT_BIO_TRIMUL_GP_BANK_SPLIT" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="k10-b1-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("k10-b1-permute-land", cfg)

    def fold(arm: str) -> dict:
        TT.set_trimul_gp_bank_split(arm == "on")
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        RB.REJECTS.clear()
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
                "roles": list(TT.gp_roles()), "gated_calls": list(RB.STATS_GATED),
                "gated_rejects": {str(k): v for k, v in RB.REJECTS.items()},
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s {r['roles']} gated={r['gated_calls']} "
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
