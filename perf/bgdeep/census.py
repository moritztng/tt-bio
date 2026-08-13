#!/usr/bin/env python3
"""BoltzGen design-step decomposition + cross-model lever census, one process, card-pinned.

Design, not fold, so `tt_baseline.build_fold` cannot be used. `tt-bio design --model boltzgen`
runs the BoltzGen CLI IN-PROCESS for a single `--device_ids` (`tt_bio.main._run_boltzgen_cli` ->
`tt_bio.boltzgen.cli.boltzgen.main`, dispatch at cli/boltzgen.py:727 `len(devices) > 1`), so the
hooks installed here see the real calls. Modelled on perf/y_permute_crossmodel/boltzgen_ab.py.

Two things per run:

1. DECOMPOSITION of the design forward into trunk / diffusion device / diffusion host, with the
   per-step device call timed by the call itself (`DiffusionModule.forward` ends in `_to_torch`,
   which is a blocking device read, so the region is synced by construction -- no added sync).
   Host residual = sample() wall minus the summed device calls; that is the AtomDiffusion loop's
   own torch work (center, augmentation einsum, weighted_rigid_align SVD, randn) and the
   host<->device staging, and it cannot overlap the device because the loop is sequential.

2. LEVER CENSUS with counters, so "gated off with a reason" is distinguishable from "fired and
   did nothing". A lever reporting served 0 / declined 0 is UNTESTED, not inactive.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltzgen-deep-perf \\
        python3 perf/bgdeep/census.py --num-designs 2
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
SHAPES = Counter()
DIFF_SHAPE = {}


def _t(key, fn, *a, **k):
    t0 = time.perf_counter()
    try:
        return fn(*a, **k)
    finally:
        d = time.perf_counter() - t0
        WALL[key]["n"] += 1
        WALL[key]["s"] += d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="design spec yaml (default: the fixed-length copy)")
    ap.add_argument("--protocol", default="protein-anything")
    ap.add_argument("--num-designs", type=int, default=2)
    ap.add_argument("--steps", default="design")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    spec = Path(a.spec) if a.spec else here / "binder_fixed100.yaml"
    out_json = Path(a.out) if a.out else here / "census.json"

    import tt_bio.tenstorrent as T
    import tt_bio.triatt_qkv as HM        # K1 / K1b
    import tt_bio.triatt_sdpa as PM       # K2
    import tt_bio.reblock_permute as RB   # E6 + the two channel moves

    # ---- counters -------------------------------------------------------------------------
    ORIG_QKVMM = T._qkv_mm_config

    def qkvmm(x, w, *args, **kw):
        cfg = ORIG_QKVMM(x, w, *args, **kw)
        kt, nt = (int(w.shape[-2]) + 31) // 32, (int(w.shape[-1]) + 31) // 32
        DEC[f"qkv_mm_config|kt={kt},nt={nt}"]["config" if cfg is not None else "None"] += 1
        return cfg

    T._qkv_mm_config = qkvmm

    ORIG_TMC = T._transpose_memory_config

    def tmc(*args, **kw):
        mc = ORIG_TMC(*args, **kw)
        DEC["transpose_memory_config"]["L1" if mc is not None else "DRAM"] += 1
        return mc

    T._transpose_memory_config = tmc

    # ---- timers ---------------------------------------------------------------------------
    def patch(mod, name, key, meth="__call__"):
        cls = getattr(mod, name, None)
        if cls is None:
            return False
        f = getattr(cls, meth)
        setattr(cls, meth, (lambda g: lambda self, *x, **k: _t(key, g, self, *x, **k))(f))
        return True

    installed = {}
    for nm, key in (("PairformerModule", "stage:PairformerModule"),
                    ("Pairformer", "stage:Pairformer"),
                    ("PairformerLayer", "block:PairformerLayer"),
                    ("TriangleMultiplication", "body:TriangleMultiplication"),
                    ("TriangleAttention", "body:TriangleAttention"),
                    ("PairWeightedAveraging", "body:PairWeightedAveraging")):
        installed[key] = patch(T, nm, key)

    # The per-step device call. `DiffusionModule.forward` returns a host torch tensor, so the
    # region ends on a blocking read: synced by construction.
    _orig_dm_fwd = T.DiffusionModule.forward

    def dm_fwd(self, r, times, s_inputs, s_trunk, *rest, **kw):
        if not DIFF_SHAPE:
            DIFF_SHAPE.update(N_real=int(r.shape[1]), seq_len=int(s_inputs.shape[1]),
                              batch=int(r.shape[0]))
        return _t("diffusion:device_step", _orig_dm_fwd, self, r, times, s_inputs, s_trunk,
                  *rest, **kw)

    T.DiffusionModule.forward = dm_fwd
    installed["diffusion:device_step"] = True

    from tt_bio.boltzgen.model.modules.diffusion import AtomDiffusion
    _orig_sample = AtomDiffusion.sample

    def sample(self, *x, **k):
        return _t("diffusion:sample_loop", _orig_sample, self, *x, **k)

    AtomDiffusion.sample = sample
    installed["diffusion:sample_loop"] = True

    from tt_bio.boltzgen.model.modules import diffusion_conditioning as DC
    for nm in ("DiffusionConditioning",):
        installed[f"cond:{nm}"] = patch(DC, nm, f"cond:{nm}", meth="forward")

    from tt_bio.boltzgen.model.models import boltz as BZ
    _orig_boltz_fwd = BZ.Boltz.forward

    # E1d: the planning pass left 7.63 s/design of `Boltz.forward` un-attributed (17.1 %, larger
    # than tri-att + trimul combined). Wrap the named submodules `forward` actually calls, so the
    # residual is decomposed instead of estimated. These are torch nn.Modules, so replacing the
    # bound `forward` on the INSTANCE is enough -- `obj(...)` routes through it.
    E1D_SUBMODULES = (
        "input_embedder", "inverse_folding_encoder", "s_init", "rel_pos", "token_bonds",
        "token_bonds_type", "contact_conditioning", "token_distance_module", "template_module",
        "msa_module", "pairformer_module", "distogram_module", "bfactor_module",
        "confidence_module", "masker",
    )

    # Per-design snapshots. The planning pass averaged a COLD design 1 (which JIT-compiles the
    # whole ~1275-program diffusion graph) together with a warm one, so every row of its table is
    # inflated. Snapshot the timers at each design boundary and report the designs separately.
    PER_DESIGN = []

    def _snapshot():
        return {k: (v["n"], v["s"]) for k, v in WALL.items()}

    def boltz_fwd(self, *x, **k):
        if not getattr(self, "_e1d_wrapped", False):
            self._e1d_wrapped = True
            for nm in E1D_SUBMODULES:
                sub = getattr(self, nm, None)
                if sub is None or not callable(getattr(sub, "forward", None)):
                    continue
                key = f"e1d:{nm}"
                installed[key] = True
                setattr(sub, "forward",
                        (lambda g, kk: lambda *a2, **k2: _t(kk, g, *a2, **k2))(sub.forward, key))
        before = _snapshot()
        try:
            return _t("stage:Boltz.forward", _orig_boltz_fwd, self, *x, **k)
        finally:
            after = _snapshot()
            PER_DESIGN.append({
                k: {"calls": after[k][0] - before.get(k, (0, 0.0))[0],
                    "s": round(after[k][1] - before.get(k, (0, 0.0))[1], 4)}
                for k in after
                if after[k][0] - before.get(k, (0, 0.0))[0] > 0
            })

    BZ.Boltz.forward = boltz_fwd
    installed["stage:Boltz.forward"] = True

    # ---- run ------------------------------------------------------------------------------
    workdir = Path.home() / "bgdeep_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", str(spec), "--output", str(workdir),
            "--num_designs", str(a.num_designs), "--protocol", a.protocol,
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    if a.steps:
        argv += ["--steps", *a.steps.split(",")]
    from tt_bio.main import _run_boltzgen_cli
    t0 = time.perf_counter()
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    wall = time.perf_counter() - t0

    import importlib.metadata as md
    R = {
        "wheel": md.version("ttnn"), "host": "qb2",
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "spec": str(spec),
        "protocol": a.protocol, "num_designs": a.num_designs, "steps": a.steps,
        "wall_s": round(wall, 3),
        "s_per_design": round(wall / max(1, a.num_designs), 3),
        "designs_per_hour": round(3600.0 * a.num_designs / wall, 2),
        "timers_installed": installed,
        "shapes": {"MAX_ATOMS_PER_TOKEN": T.MAX_ATOMS_PER_TOKEN,
                   "PAIRFORMER_PAD_MULTIPLE": T.PAIRFORMER_PAD_MULTIPLE,
                   "ATOM_WINDOW": T.ATOM_WINDOW, **DIFF_SHAPE},
        "wall_by_component": {k: {"calls": v["n"], "s": round(v["s"], 4)}
                              for k, v in sorted(WALL.items())},
        "per_design": PER_DESIGN,
        "levers": {
            "K1_head_major_qkv": {"served": HM.STATS[0], "declined": HM.STATS[1],
                                  "rejects": {f"{k}": v for k, v
                                              in getattr(HM, "REJECTS", {}).items()}},
            "K1b_head_major_tail": {"served": HM.TAIL_STATS[0], "declined": HM.TAIL_STATS[1]},
            "K2_persistent_sdpa_bias": {"served": PM.STATS[0], "declined": PM.STATS[1]},
            "E6_fused_gated_channel_move": {"served": RB.STATS_GATED[0],
                                            "declined": RB.STATS_GATED[1]},
            "channel_move_fwd": {"served": RB.STATS[0], "declined": RB.STATS[1]},
            "channel_move_back": {"served": RB.STATS_BACK[0], "declined": RB.STATS_BACK[1]},
            "reblock_rejects": {f"{k}": v for k, v in getattr(RB, "REJECTS", {}).items()},
        },
        "counters": {k: dict(v) for k, v in sorted(DEC.items())},
    }
    out_json.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("wall_s", "s_per_design", "designs_per_hour",
                                        "shapes", "wall_by_component", "levers", "counters")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
