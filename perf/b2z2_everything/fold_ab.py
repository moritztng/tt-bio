#!/usr/bin/env python3
"""Everything wave 2 built, on one chip, in one fold: base vs the bit-exact subset vs the union.

Eleven named levers in four stages -- sampler, trunk, MSA, host -- each measured against its own
base on its own branch. This folds them together. 200 sampling steps, 3 recycles, full MSA depth,
the fixture's fixed 35-row a3m, one sample, seed 0.

Arms are lever SETS, applied as module globals and env vars between folds, so one device open
times every arm against one base. Two things the arms are built to prove rather than assert:

  * the bit-exact subset must write the base CIF byte for byte. If it does not, one member is not
    bit-exact and the per-arm legs say which.
  * the union must NOT write the base CIF, because three of its members are host levers that move
    atoms. An identical hash on both arms of a non-bit-exact A/B means the toggle never reached
    the code -- this wave has already produced one clean null that way.

Per-fold CIFs are kept so the parity scorer can read them without re-folding.
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

# The population, by the name this row's doc uses. TT_BIO_ATOM_KEY_WINDOW is deliberately absent:
# it computes the wrong gather and TT_BIO_ATOM_SHIFT_GATHER is the same optimization done right.
LEVERS = ("SHG", "KVP", "L1", "SDPAQ", "QKVG", "PWA", "COND", "ZINIT", "CONF")
BITEXACT = ("SHG", "KVP", "L1", "SDPAQ", "QKVG", "PWA")
HOST = ("COND", "ZINIT", "CONF")

ARMS = {
    "base": (),
    "BX": BITEXACT,                      # everything that must leave the CIF byte-identical
    "ALL": BITEXACT + HOST,              # the whole union
    "HOSTONLY": HOST,
    "SAMP": ("SHG", "KVP", "L1", "SDPAQ"),
    "TRUNK": ("QKVG",),
    "MSA": ("PWA",),
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
    ap.add_argument("--cifs", type=Path, default=None, help="keep one CIF per arm+rep here")
    ap.add_argument("--arms", default="base,BX,ALL")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_PATH = args.out
    arms = args.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm
    assert arms[0] == "base", "base must lead so the cold-fold order matches the timed order"
    # rule 0 is not negotiable, and the driver asserts it rather than the author
    assert args.steps == 200 and args.recycles == 3, "200 sampling steps, 3 recycles"
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_qkv as QK
    import tt_bio.boltz2 as B2
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    # The arms are module globals and env vars set in process; a pin outside would silently make
    # two arms the same arm, which is exactly how this wave produced a clean null over four folds.
    for k in ("TT_BIO_ATOM_KEY_WINDOW", "TT_BIO_ATOM_KV_PREPROJ", "TT_BIO_ATOM_L1",
              "TT_BIO_ATOM_SHIFT_GATHER", "TT_BIO_SDPA_GRID_Q_CHUNK", "TT_BIO_TRIATT_FUSED_QKVG",
              "TT_BIO_PWA_RESIDENCY", "TT_BIO_PWA_BATCH_HEAD_WEIGHTS", "TT_BIO_PWA_L1_NORM_M",
              "TT_BIO_PWA_L1_ROWS", "TT_BIO_DEVICE_CONDITIONING", "TT_BIO_DEVICE_ZINIT",
              "TT_BIO_DEVICE_CONFIDENCE"):
        assert k not in os.environ, f"{k} is pinned in the env; the arms are set in process"
    assert "TT_METAL_DEVICE_PROFILER" not in os.environ, (
        "TT_METAL_DEVICE_PROFILER must be ABSENT, not 0 -- setting it at all trips TT_FATAL "
        "rtoptions.cpp:708 on a non-Tracy ttnn, and a watcher-armed run is not a ratio")

    def apply(arm: str) -> dict:
        on = set(ARMS[arm])
        TT._ATOM_KEY_WINDOW = False                      # never taken: it gathers the wrong atoms
        TT._ATOM_SHIFT_GATHER = "SHG" in on
        TT._ATOM_KV_PREPROJ = "KVP" in on
        TT._ATOM_L1 = "L1" in on
        TT._SDPA_GRID_Q_CHUNK = "SDPAQ" in on
        TT._sdpa_program_config_for_lengths.cache_clear()
        TT._grid_q_chunk.cache_clear()
        QK._QKVG_ENABLED = "QKVG" in on
        TT._PWA_BATCH_HEAD_WEIGHTS = "PWA" in on
        TT._PWA_L1_NORM_M = "PWA" in on
        TT._PWA_L1_ROWS = 0 if "PWA" in on else -1
        for name, env in (("COND", "TT_BIO_DEVICE_CONDITIONING"),
                          ("ZINIT", "TT_BIO_DEVICE_ZINIT"),
                          ("CONF", "TT_BIO_DEVICE_CONFIDENCE")):
            os.environ[env] = "1" if name in on else "0"
        # read the arm back off the code rather than off this function
        return {"atom_key_window": TT._ATOM_KEY_WINDOW, "shift_gather": TT._ATOM_SHIFT_GATHER,
                "kv_preproj": TT._ATOM_KV_PREPROJ, "atom_l1": TT._ATOM_L1,
                "sdpa_grid_q": TT._SDPA_GRID_Q_CHUNK, "qkvg": QK._QKVG_ENABLED,
                "pwa_heads": TT._PWA_BATCH_HEAD_WEIGHTS, "pwa_l1_norm_m": TT._PWA_L1_NORM_M,
                "pwa_l1_rows": TT._PWA_L1_ROWS,
                "conditioning": B2._device_conditioning(), "zinit": B2._device_zinit(),
                "confidence": B2._device_confidence()}

    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture, "seed": args.seed,
        "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
        "arch": TT.arch_name(), "grid": str(TT.CORE_GRID_MAIN),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "arms": arms, "reps": args.reps, "levers": {a: list(ARMS[a]) for a in arms},
    }
    OUT["arm_flags"] = {a: apply(a) for a in ARMS}
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-everything-wh-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    if args.seed:
        cfg.seed = args.seed
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-everything-union-wh", cfg)
    diff_mod = state.model.structure_module.score_model
    if args.cifs:
        args.cifs.mkdir(parents=True, exist_ok=True)

    def fold(arm: str, tag: str) -> dict:
        flags = apply(arm)
        TT.ATOM_SHIFT_GATHER_STATS[0] = TT.ATOM_SHIFT_GATHER_STATS[1] = 0
        TT.ATOM_L1_STATS["l1"] = TT.ATOM_L1_STATS["dram"] = 0
        QK.QKVG_STATS[0] = QK.QKVG_STATS[1] = 0
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
        if args.cifs:
            shutil.copy(cifs[0], args.cifs / f"{args.fixture}_{arm}_{tag}.cif")
        ps = None
        if len(sp.loop) > 2:
            ps = round(1e3 * (sp.loop[-1][1] - sp.loop[0][1])
                       / (sp.loop[-1][0] - sp.loop[0][0]), 4)
        assert sp.n_steps in (0, args.steps), f"{sp.n_steps} sampler steps, not {args.steps}"
        return {"arm": arm, "fold_s": round(wall, 3), "stages_s": stages,
                "sampler_ms_per_step": ps, "step_n": sp.n_steps, "flags": flags,
                "gather_stats": list(TT.ATOM_SHIFT_GATHER_STATS),
                "atom_l1_stats": dict(TT.ATOM_L1_STATS), "qkvg_stats": list(QK.QKVG_STATS),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2)}

    runs = []
    # One cold fold PER ARM, discarded: an arm that adds device programs compiles them on its
    # first fold, and with a single cold fold that compile lands inside the first timed A.
    for arm in arms:
        r = fold(arm, "cold"); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:8s} {r['fold_s']:7.3f}s gather={r['gather_stats']} "
              f"l1={r['atom_l1_stats']} qkvg={r['qkvg_stats']} cif {r['cif_sha256'][:16]}",
              flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            r = fold(arm, f"r{i}"); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:8s} {r['fold_s']:7.3f}s stages {r['stages_s']} "
                  f"{r['sampler_ms_per_step']} ms/step n={r['step_n']} "
                  f"cif {r['cif_sha256'][:16]} plddt {r['plddt']} load {r['loadavg1']}",
                  flush=True)
            OUT["runs"] = runs; dump()
            summarise(OUT, arms, args.reps); dump()

    summarise(OUT, arms, args.reps)
    OUT["env"]["loadavg_end"] = os.getloadavg()
    OUT["env"]["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    print(json.dumps({k: OUT[k] for k in (
        "median_fold_s", "paired_ratio", "aa_floor_paired", "aa_floor_spread",
        "median_sampler_ms_per_step", "sampler_ratio_vs_base", "median_stage_s",
        "stage_ratio_vs_base", "bit_exact_vs_base", "cif_sha256", "plddt")}, indent=1), flush=True)
    return 0


def summarise(OUT, arms, reps):
    timed = [r for r in OUT["runs"] if not r["warmup"]]
    if not timed:
        return

    def med(a, key="fold_s"):
        v = [r[key] for r in timed if r["arm"] == a and r.get(key) is not None]
        return st.median(v) if v else None

    OUT["median_fold_s"] = {a: round(med(a), 4) for a in arms if med(a)}
    OUT["median_sampler_ms_per_step"] = {
        a: round(med(a, "sampler_ms_per_step"), 4) for a in arms
        if med(a, "sampler_ms_per_step")}
    b = med("base")
    OUT["ratio_vs_base"] = {a: round(b / med(a), 5) for a in arms if med(a)}
    bs = med("base", "sampler_ms_per_step")
    if bs:
        OUT["sampler_ratio_vs_base"] = {
            a: round(bs / med(a, "sampler_ms_per_step"), 5) for a in arms
            if med(a, "sampler_ms_per_step")}
    # per-stage walls: trunk, conditioning, sampler, confidence
    keys = sorted({k for r in timed for k in r["stages_s"]})
    OUT["median_stage_s"] = {a: {k: round(st.median(v), 4) for k in keys
                                 if (v := [r["stages_s"][k] for r in timed
                                           if r["arm"] == a and k in r["stages_s"]])}
                             for a in arms}
    base_st = OUT["median_stage_s"].get("base", {})
    OUT["stage_ratio_vs_base"] = {
        a: {k: round(base_st[k] / v, 5) for k, v in s.items() if base_st.get(k) and v}
        for a, s in OUT["median_stage_s"].items() if a != "base"}
    # paired: every rep folds every arm back to back, so a per-rep ratio cancels slow reps
    per_rep = {}
    for a in arms:
        rs = []
        for i in range(reps):
            bb = [r["fold_s"] for r in timed if r.get("rep") == i and r["arm"] == "base"]
            xx = [r["fold_s"] for r in timed if r.get("rep") == i and r["arm"] == a]
            if bb and xx:
                rs.append(bb[0] / xx[0])
        if rs:
            per_rep[a] = rs
    OUT["paired_ratio"] = {a: round(st.median(v), 5) for a, v in per_rep.items()}
    OUT["paired_ratio_all"] = {a: [round(x, 5) for x in v] for a, v in per_rep.items()}
    # The A/A floor of the SAME estimator: base folds split into two halves by rep parity and
    # paired against each other, which is the null the headline median is read against.
    bases = [r["fold_s"] for r in sorted(
        (r for r in timed if r["arm"] == "base"), key=lambda r: r.get("rep", 0))]
    if len(bases) > 1:
        OUT["aa_floor_spread"] = round(max(bases) / min(bases), 5)
        pairs = [bases[i] / bases[i + 1] for i in range(len(bases) - 1)]
        OUT["aa_floor_paired"] = round(max(st.median([max(p, 1 / p) for p in pairs]), 1.0), 5)
    OUT["cif_sha256"] = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a})
                         for a in arms}
    base_sha = OUT["cif_sha256"].get("base")
    OUT["bit_exact_vs_base"] = {a: (len(v) == 1 and v == base_sha)
                                for a, v in OUT["cif_sha256"].items() if v}
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in arms}


def splitter(ttnn, dev):
    """Stage walls with an explicit sync at each boundary: trunk, conditioning, sampler, head."""
    import time as _t

    class Splitter:
        def __init__(self):
            self.marks, self.loop, self.n_diff, self.n_steps = [], [], 0, 0
            self.n_trunk = 0

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, _t.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "trunk":
                self.n_trunk += 1
                if self.n_trunk == 1:
                    # opens the trunk wall: every recycle, MSA module included, up to the
                    # first diffusion mark. Feature prep before this is outside every stage
                    # and shows up as fold_s minus the sum, which is what it is.
                    self._mark("trunk")
            elif stage == "diffusion":
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
