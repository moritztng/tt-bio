#!/usr/bin/env python3
"""Boltz-2 512 aa cell A/B of TT_BIO_DEVICE_ZINIT, paired and interleaved ABBA.

Arm `on` is this tree`s default: the trunk`s pair input is assembled on the device where the
resident trunk consumes it. Arm `off` sets the flag to 0, which is what main does -- six
[1, n, n, 128] fp32 terms in torch and a 134 MB upload. The flag is read at call time, so both
arms run out of one process against one loaded model and the only thing that changes between
them is where z_init is built.

The transform is not bit-exact (bf16 device math, and the six terms are summed in a different
order), so the CIF sha is recorded as a fact and not as the bar; the bar is the Angstrom reading
from `control298.py` + `score298.py` / `domain_split.py`.

`PairAssemblyDevice.__call__` is counted, because a lever that can decline needs an instrument
that says it fired: an `on` arm with a zero count is a silent fallback, not a measurement.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, which is the fold
the rest of this campaign times. 200 sampling steps, 3 recycles, full MSA: no work is removed.
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
FLAG = "TT_BIO_DEVICE_ZINIT"
OUT: dict = {}
OUT_PATH: Path | None = None
CALLS = [0]


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path)
    ap.add_argument("--reps", type=int, default=8)
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
    import tt_bio.boltz2 as B2
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert FLAG not in os.environ, f"{FLAG} is pinned; the arms are set in-process"
    assert B2._device_zinit(), "this tree does not default the lever on"

    _call = TT.PairAssemblyDevice.__call__

    def counted(self, *a, **k):
        CALLS[0] += 1
        return _call(self, *a, **k)

    TT.PairAssemblyDevice.__call__ = counted

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "arch": TT.arch_name(), "flag": FLAG,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-zinitship-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-zinit-ship", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str, keep: str | None = None) -> dict:
        os.environ[FLAG] = "1" if arm == "on" else "0"
        CALLS[0] = 0
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
        if keep and args.cifdir:
            dest = args.cifdir / keep
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(struct_dir, dest)
        return {"arm": arm, "fold_s": round(wall, 3), "assembly_calls": CALLS[0],
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r[fold_s]:7.3f}s calls={r[assembly_calls]} "
              f"cif {r[cif_sha256][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for j, arm in enumerate(ORDER):
            keep = f"512_{arm}" if (i == 0 and j < 2) else None
            r = fold(arm, keep); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r[fold_s]:7.3f}s calls={r[assembly_calls]} "
                  f"plddt={r[plddt]} load={r[loadavg1]} cif {r[cif_sha256][:16]}",
                  flush=True)
            OUT["runs"] = runs; dump()
    os.environ.pop(FLAG, None)

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
    OUT["ab_paired_mean_delta_s"] = round(st.mean(
        [o - n for i in sorted(rep) for o, n in
         zip([rep[i][0]["fold_s"], rep[i][3]["fold_s"]],
             [rep[i][1]["fold_s"], rep[i][2]["fold_s"]])]), 4)
    OUT["aa_floor_max"] = max(aa) if aa else None
    OUT["aa_ratios"] = aa
    OUT["assembly_calls"] = {a: sorted({r["assembly_calls"] for r in timed if r["arm"] == a})
                             for a in ("off", "on")}
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a})
                    for a in ("off", "on")}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    dump()
    keys = ("median_fold_s", "ratio_median", "ab_paired_median", "ab_paired_ratios",
            "ab_paired_all_positive", "ab_paired_mean_delta_s", "aa_floor_max",
            "assembly_calls", "plddt", "bit_exact", "cif_sha256")
    print(json.dumps({k: OUT[k] for k in keys if k in OUT}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
