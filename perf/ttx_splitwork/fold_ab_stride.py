#!/usr/bin/env python3
"""Boltz-2 fold A/B of the strided group assignment in the channel move. Paired, ABBA.

Arm `off` is main's contiguous block, arm `on` is this branch's default: core i takes groups
i, i+num_cores, i+2*num_cores, ... Both arms run in ONE process, and the assignment is part of
all three `reblock_permute` descriptor cache keys, so each arm compiles its own program instead
of inheriting the other's.

The lever moves which core does which group, never a value, so the acceptance is a matching CIF
sha256 across every fold of both arms. The per-op ratios and the mechanism are in
perf/ttx_splitwork/; this answers the only question those cannot, which is what a whole fold gets.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, the same fold every
other ship row on this ladder times. ABBA inside a rep so a drifting box cancels.
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
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    args = ap.parse_args()
    OUT_PATH = args.out
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.reblock_permute as RB
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert "TT_BIO_REBLOCK_STRIDE" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"
    assert RB.STRIDED, "this branch ships strided on; arm `on` must be the default"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="ttx-stride-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-reblock-cores-ship", cfg)

    def fold(arm: str) -> dict:
        RB.STRIDED = (arm == "on")
        for s in (RB.STATS, RB.STATS_BACK, RB.STATS_GATED):
            s[0] = s[1] = 0
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
        return {"arm": arm, "fold_s": round(wall, 3), "strided": RB.STRIDED,
                "calls": {"fwd": list(RB.STATS), "back": list(RB.STATS_BACK),
                          "gated": list(RB.STATS_GATED)},
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s calls={r['calls']} "
              f"cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in ORDER:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r['fold_s']:7.3f}s plddt={r['plddt']} "
                  f"load={r['loadavg1']} cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    per = {a: [r["fold_s"] for r in timed if r["arm"] == a] for a in ("off", "on")}
    med = {a: st.median(v) for a, v in per.items()}
    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    deltas = [round(o - n, 3) for o, n in zip(per["off"], per["on"])]
    OUT["n_per_arm"] = {a: len(v) for a, v in per.items()}
    OUT["median_fold_s"] = med
    OUT["ratio"] = med["off"] / med["on"]
    # The A/A floor of THIS session, from the control arm alone: the spread the box gives you for
    # free. A ratio inside it is not a win.
    OUT["aa_floor"] = min(per["off"]) / max(per["off"])
    OUT["paired_deltas_s"] = deltas
    OUT["paired_delta_median_s"] = st.median(deltas)
    OUT["paired_wins"] = sum(d > 0 for d in deltas)
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["calls"] = {a: [r["calls"] for r in timed if r["arm"] == a][0] for a in ("off", "on")}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    dump()
    keys = ("n_per_arm", "median_fold_s", "ratio", "aa_floor", "paired_deltas_s",
            "paired_delta_median_s", "paired_wins", "plddt", "calls", "bit_exact", "cif_sha256")
    print(json.dumps({k: OUT[k] for k in keys if k in OUT}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
