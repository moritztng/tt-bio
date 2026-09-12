#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B of the slice+concat atom key gather, paired and interleaved.

Arm `off` pins `_ATOM_SHIFT_GATHER_OFF`, so the gather stays the one-hot matmul with its four
arithmetic-free programs. Arm `on` is the shipped default of this branch. The arms are set in
process and the diffusion static cache is reset between them, because the decision is taken once
per fold where `keys_indexing` is still a torch tensor.

Both arms must write the same CIF byte for byte: the gather is a pure index permutation, so the
bar is a matching sha256, not an RMSD.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py so this is the same
fold the campaign's other rows time.
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
ORDER = ["off", "on", "off", "on"]
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
    ap.add_argument("--reps", type=int, default=2)
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
    assert "TT_BIO_ATOM_SHIFT_GATHER" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "atom_key_shift": TT.ATOM_KEY_SHIFT,
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-elision-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-layout-op-elision", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str) -> dict:
        TT._ATOM_SHIFT_GATHER_OFF = (arm == "off")
        TT.ATOM_SHIFT_GATHER_STATS[0] = TT.ATOM_SHIFT_GATHER_STATS[1] = 0
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
        sp = LEV_Splitter(ttnn, dev)
        state.pfn = sp
        state.model.progress_fn = sp
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        stages = sp.close()
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        ps = None
        if len(sp.loop) > 2:
            ps = round(1e3 * (sp.loop[-1][1] - sp.loop[0][1])
                       / (sp.loop[-1][0] - sp.loop[0][0]), 4)
        return {"arm": arm, "fold_s": round(wall, 3), "stages_s": stages,
                "sampler_ms_per_step": ps,
                "gather_stats": list(TT.ATOM_SHIFT_GATHER_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s gather={r['gather_stats']} "
              f"cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in ORDER:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r['fold_s']:7.3f}s "
                  f"sampler {r['stages_s'].get('sampler')} "
                  f"{r['sampler_ms_per_step']} ms/step gather={r['gather_stats']} "
                  f"cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in ("off", "on")}
    smed = {a: st.median([r["sampler_ms_per_step"] for r in timed
                          if r["arm"] == a and r["sampler_ms_per_step"]])
            for a in ("off", "on")}
    offs = [r["fold_s"] for r in timed if r["arm"] == "off"]
    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["median_fold_s"] = med
    OUT["ratio"] = med["off"] / med["on"]
    OUT["aa_floor"] = (min(offs) / max(offs)) if len(offs) > 1 else None
    OUT["median_sampler_ms_per_step"] = smed
    OUT["sampler_ratio"] = smed["off"] / smed["on"]
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    dump()
    print(json.dumps({k: OUT[k] for k in ("median_fold_s", "ratio", "aa_floor",
                                          "median_sampler_ms_per_step", "sampler_ratio",
                                          "bit_exact", "cif_sha256")}, indent=1), flush=True)
    return 0


def LEV_Splitter(ttnn, dev):
    """The flag-lever harness's stage splitter, rebound to this process's device."""
    import time as _t

    class Splitter:
        def __init__(self):
            self.marks, self.loop, self.n_diff = [], [], 0

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, _t.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "diffusion":
                self.n_diff += 1
                if self.n_diff == 1:
                    self._mark("diffusion_conditioning")
                elif self.n_diff == 2:
                    self._mark("sampler")
                    self.loop.append((step, self.marks[-1][1]))
                else:
                    self.loop.append((step, _t.perf_counter()))
            elif stage == "confidence":
                self._mark("confidence")

        def close(self):
            self._mark("end")
            return {lab: round(t1 - t0, 4)
                    for (lab, t0), (_l, t1) in zip(self.marks, self.marks[1:])}
    return Splitter()


if __name__ == "__main__":
    sys.exit(main())
