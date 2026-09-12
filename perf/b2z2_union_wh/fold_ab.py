#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B of the composed atom-branch union, paired and interleaved on WH.

The step ratio is not a fold ratio and this wave has been burned carrying one to the other, so
the union is folded end to end: 200 sampling steps, 3 recycles, the fixture's fixed 35-row a3m,
one sample, seed 0. Arms are set as module globals and the diffusion static cache is reset
between them, because the gather decision is taken once per fold while `keys_indexing` is still
a torch tensor.

Both arms must write the same CIF byte for byte if the union is bit-exact end to end, so the bar
here is a matching sha256 and not an RMSD.

Protocol, fixtures and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, so this is the same
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

#        arm  -> (KEY_WINDOW, KV_PREPROJ, ATOM_L1, SHIFT_GATHER)
ARMS = {
    "base":  (False, False, False, False),
    "G":     (True,  False, False, False),
    "S":     (False, False, False, True),
    "L1":    (False, False, True,  False),
    "GK":    (True,  True,  False, False),
    "SK":    (False, True,  False, True),
    "UNION": (True,  True,  True,  False),
    "SUNION": (False, True,  True,  True),
}
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
    ap.add_argument("--arms", default="base,SUNION")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    args = ap.parse_args()
    OUT_PATH = args.out
    arms = args.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm
    # rule 0 is not negotiable, and the driver asserts it rather than the author
    assert args.steps == 200 and args.recycles == 3, "200 sampling steps, 3 recycles"
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
    for k in ("TT_BIO_ATOM_KEY_WINDOW", "TT_BIO_ATOM_KV_PREPROJ", "TT_BIO_ATOM_L1",
              "TT_BIO_ATOM_SHIFT_GATHER"):
        assert k not in os.environ, f"{k} is pinned in the env; the arms are set in process"

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
        "arch": TT.arch_name(), "grid": str(TT.CORE_GRID_MAIN),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "arms": arms, "reps": args.reps, "atom_key_shift": TT.ATOM_KEY_SHIFT,
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-union-wh-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-sampler-union-wh", cfg)
    diff_mod = state.model.structure_module.score_model

    def fold(arm: str) -> dict:
        (TT._ATOM_KEY_WINDOW, TT._ATOM_KV_PREPROJ,
         TT._ATOM_L1, TT._ATOM_SHIFT_GATHER) = ARMS[arm]
        TT.ATOM_SHIFT_GATHER_STATS[0] = TT.ATOM_SHIFT_GATHER_STATS[1] = 0
        TT.ATOM_L1_STATS["l1"] = TT.ATOM_L1_STATS["dram"] = 0
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
        sp = splitter(ttnn, dev)
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
        assert sp.n_steps in (0, args.steps), f"{sp.n_steps} sampler steps, not {args.steps}"
        return {"arm": arm, "fold_s": round(wall, 3), "stages_s": stages,
                "sampler_ms_per_step": ps, "step_n": sp.n_steps,
                "gather_stats": list(TT.ATOM_SHIFT_GATHER_STATS),
                "atom_l1_stats": dict(TT.ATOM_L1_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    for arm in arms:                                        # one cold fold per arm, discarded
        r = fold(arm); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:6s} {r['fold_s']:7.3f}s gather={r['gather_stats']} "
              f"l1={r['atom_l1_stats']} cif {r['cif_sha256'][:16]}", flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:6s} {r['fold_s']:7.3f}s "
                  f"sampler {r['stages_s'].get('sampler')} "
                  f"{r['sampler_ms_per_step']} ms/step n={r['step_n']} "
                  f"cif {r['cif_sha256'][:16]} load {r['loadavg1']}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]

    def med(a, key="fold_s"):
        v = [r[key] for r in timed if r["arm"] == a and r[key] is not None]
        return st.median(v) if v else None

    OUT["median_fold_s"] = {a: round(med(a), 4) for a in arms}
    OUT["median_sampler_ms_per_step"] = {a: round(med(a, "sampler_ms_per_step"), 4) for a in arms}
    OUT["ratio_vs_base"] = {a: round(med("base") / med(a), 5) for a in arms}
    OUT["sampler_ratio_vs_base"] = {
        a: round(med("base", "sampler_ms_per_step") / med(a, "sampler_ms_per_step"), 5)
        for a in arms}
    # paired: every rep folds every arm back to back, so a per-rep ratio cancels slow reps
    per_rep = {}
    for a in arms:
        rs = []
        for i in range(args.reps):
            b = [r["fold_s"] for r in timed if r["rep"] == i and r["arm"] == "base"]
            x = [r["fold_s"] for r in timed if r["rep"] == i and r["arm"] == a]
            if b and x:
                rs.append(b[0] / x[0])
        per_rep[a] = rs
    OUT["paired_ratio"] = {a: round(st.median(v), 5) for a, v in per_rep.items() if v}
    OUT["paired_ratio_all"] = {a: [round(x, 5) for x in v] for a, v in per_rep.items()}
    bases = sorted(r["fold_s"] for r in timed if r["arm"] == "base")
    OUT["aa_floor"] = round(bases[-1] / bases[0], 5) if len(bases) > 1 else None
    OUT["cif_sha256"] = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a})
                         for a in arms}
    base_sha = OUT["cif_sha256"]["base"]
    OUT["bit_exact_vs_base"] = {a: (len(v) == 1 and v == base_sha)
                                for a, v in OUT["cif_sha256"].items()}
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in arms}
    OUT["env"]["loadavg_end"] = os.getloadavg()
    dump()
    print(json.dumps({k: OUT[k] for k in (
        "median_fold_s", "ratio_vs_base", "paired_ratio", "aa_floor",
        "median_sampler_ms_per_step", "sampler_ratio_vs_base",
        "bit_exact_vs_base", "cif_sha256", "plddt")}, indent=1), flush=True)
    return 0


def splitter(ttnn, dev):
    """The flag-lever harness's stage splitter, rebound to this process's device."""
    import time as _t

    class Splitter:
        def __init__(self):
            self.marks, self.loop, self.n_diff, self.n_steps = [], [], 0, 0

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
                self.n_steps = max(self.n_steps, int(total or 0))
            elif stage == "confidence":
                self._mark("confidence")

        def close(self):
            self._mark("end")
            return {lab: round(t1 - t0, 4)
                    for (lab, t0), (_l, t1) in zip(self.marks, self.marks[1:])}
    return Splitter()


if __name__ == "__main__":
    sys.exit(main())
