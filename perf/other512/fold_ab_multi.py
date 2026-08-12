#!/usr/bin/env python3
"""Phase 0 for boltz2 / openfold3 / opendde / esmfold2: does the landed protenix lever set fire?

`perf/size512/fold_ab512.py` answers this for protenix-v2 and only for protenix-v2: it hard-codes
the model, the arm vocabulary is protenix's, and it folds at `tt_baseline`'s RECYCLING_STEPS=10,
which is protenix's own spec and not boltz2's or openfold3's (both ship 3, `main._resolve_recycling_steps`).
Folding boltz2 at 10 recycles and calling the result boltz2's wall would inflate its trunk by 3.3x.

So this harness is that one generalised on exactly three axes and nothing else:

  * `--model` reaches `tt_baseline.build_fold`, and each model folds at ITS OWN shipped
    recycling/sampling default, resolved from `tt_bio.main` and recorded per row.
  * the timer class list is probed, not assumed: esmfold2 has no shared `TriangleAttention`
    (`esmfold2.py:28-32` imports only `TriangleMultiplication`), so patching it blind would crash.
    Every row records which timers were actually installed.
  * the arm vocabulary is the six landed levers, each set EXPLICITLY on every arm so no arm can
    inherit the previous one's state:
      on     main's production defaults
      e6     `reblock_permute_gated` ON (main ships it False -- the only lever off by default)
      nok1   head-major qkv + tail OFF
      nok2   persistent SDPA bias OFF
      tr125  `_TRANSPOSE_L1_HEADROOM` 2.5 -> 1.25
      nomm   `_MM_BLOCK[8]` reverted to (2,8,1,2,1) and `_PAIR_PROJ_MM` off (the two engine-wide ones)
      nofp32 every `fp32_softmax=True` attention switched to the fused flash-SDPA route. openfold3
             is the only model that sets it (`openfold3_trunk.py:130`); this arm is the SCREEN for a
             fused fp32-softmax kernel, and it is an upper bound on that kernel by construction --
             it buys the whole traffic saving and pays none of the fp32 reduction cost, so a kernel
             that keeps the reduction in fp32 lands at or below it. It is NOT bit-exact and is not a
             shipping proposal; its plDDT against the `on` arm is the accuracy number it exists for.

A lever that reports `served 0, declined 0` is UNTESTED, not inactive -- the counters distinguish
"gated off with a reason" from "fired and did nothing", which is the whole point of Phase 0.
"""
import argparse, hashlib, json, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
STATE = {"dev": None, "model": None}
FP32_OWNERS: list = []


def _collect_fp32(root, seen=None, depth=0):
    """Every live attention module whose `fp32_softmax` is True, found by walking the built model.

    A class-level flip would not work: `fp32_softmax` is an instance attribute set by the caller
    (`openfold3_trunk.py:130`), so the arm has to reach the instances. Returns the list so a run can
    record how many it found -- an arm that flips zero modules is an A/A pair mislabelled as an A/B.
    """
    if root is None or depth > 12:
        return []
    seen = seen if seen is not None else set()
    if id(root) in seen:
        return []
    seen.add(id(root))
    out = []
    if getattr(root, "fp32_softmax", False) is True:
        out.append(root)
    for v in list(getattr(root, "__dict__", {}).values()):
        if isinstance(v, (list, tuple)):
            for it in v:
                out += _collect_fp32(it, seen, depth + 1)
        elif hasattr(v, "__dict__"):
            out += _collect_fp32(v, seen, depth + 1)
    return out

ARMS = ("on", "e6", "nok1", "nok2", "tr125", "nomm", "nofp32", "nofp32hifi",
        "nonewmm", "oldkey", "g12", "mm12", "all")

# opendde's three levers, and the integrated arm that is the only number allowed to ship:
#   g12   `_TRIMUL_INPROJ_GROUP` 12 + the divisor search -- the 12-pair channel loop in one pass
#   mm12  the two `_MM_BLOCK` widths opendde's own sweep selected, which also unlock K1/K1b there
#   all   every one of them, measured together in one arm rather than added up
_MM_BLOCK_ODDE = {(12, 36): (4, 12, 1, 2, 1), (12, 12): (8, 12, 1, 2, 1)}
GROUPS = Counter()

# The fused SDPA at full precision: fp32 accumulation in DST, exact exp, HiFi4 matmuls. MEASURED
# off-fold at openfold3's own tri-att shape (perf/other512/s2_sdpa_precision.json): 1.7097 ms against
# 1.4048 ms for the shipped fused config and 62.5789 ms for _fp32_softmax_attention -- so the whole
# precision ladder costs 0.305 ms/call and still runs 36.6x faster than the path it replaces.
CKC_HIFI = None            # bound in main() once ttnn is imported


# `oldkey` is the control for the (kt, nt) re-key: main's lookup, verbatim. main keys `_MM_BLOCK`
# on nt ALONE and then guards on `kt % blk[1]`, so it serves any kt that is a multiple of 8 at
# nt in {24, 8}. `nonewmm` only pops the four widths the re-key added and still looks up on
# (kt, nt), so it cannot see a projection main served at some kt != 8 -- which is exactly the
# regression this task exists to rule out. OLDKEY_HITS proves the arm actually ran.
_MM_BLOCK_OLD = {24: (4, 8, 1, 4, 1), 8: (4, 8, 1, 4, 1)}
OLDKEY_HITS = [0, 0]


def _mm_block_old(w):
    blk = _MM_BLOCK_OLD.get((int(w.shape[-1]) + 31) // 32)
    OLDKEY_HITS[0 if blk is not None else 1] += 1
    return blk


def _walk_trimuls(root, seen=None, depth=0):
    """Every live TriangleMultiplication in the built model, so `gated_move` is counted and not
    assumed. Same walk as `_collect_fp32`, same reason: the flag is an instance attribute."""
    import tt_bio.tenstorrent as T
    if root is None or depth > 14:
        return []
    seen = seen if seen is not None else set()
    if id(root) in seen:
        return []
    seen.add(id(root))
    out = [root] if isinstance(root, T.TriangleMultiplication) else []
    for v in list(getattr(root, "__dict__", {}).values()):
        if isinstance(v, (list, tuple)):
            for it in v:
                out += _walk_trimuls(it, seen, depth + 1)
        elif hasattr(v, "__dict__"):
            out += _walk_trimuls(v, seen, depth + 1)
    return out


def timed_call(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def patch_boltz2_cfg():
    """`build_fold`'s cfg carries no Boltz-2 hyperparameters, so `_WorkerState.load_model` raises
    KeyError('conf_kwargs'). Inject exactly what `tt_bio.main` builds. Process-local."""
    from tt_bio import worker as _W
    import tt_baseline as B

    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                  "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                  "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                  "alignment_reverse_diff": True, "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": True, "num_subsampled_msa": 1024, "use_paired_feature": True,
            "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
            "pairwise_head_width": 32, "pairwise_num_heads": 4,
            "activation_checkpointing": True}
    _steering = {"fk_steering": False, "physical_guidance_update": False,
                 "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                 "fk_resampling_interval": 3, "num_gd_steps": 20}
    _conf = dict(predict_args={"recycling_steps": B.RECYCLING_STEPS,
                               "sampling_steps": B.SAMPLING_STEPS,
                               "diffusion_samples": B.DIFFUSION_SAMPLES,
                               "max_parallel_samples": None},
                 diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
                 steering_args=_steering, use_kernels=True, use_tenstorrent=True, trace=False,
                 diffusion_trace=False)
    _aff = dict(predict_args={"recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 5,
                              "max_parallel_samples": 1},
                diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
                steering_args=dict(_steering, contact_guidance_update=False),
                affinity_mw_correction=False, use_tenstorrent=True, trace=False,
                diffusion_trace=False)
    _orig = _W._WorkerState.load_model

    def _load(self, cfg):
        cfg.setdefault("conf_kwargs", _conf)
        cfg.setdefault("aff_kwargs", _aff)
        cfg.setdefault("use_potentials", False)
        return _orig(self, cfg)

    _W._WorkerState.load_model = _load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--arms", default="on,on")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-cif", type=Path, default=None,
                    help="copy each arm's structures to <dir>/<size>_<arm>/, so two arms that "
                         "differ on purpose can be compared by RMSD and not just by sha256")
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    global CKC_HIFI
    CKC_HIFI = (ttnn.MathFidelity.HiFi4, False, True, False)
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    # qb2 is two dual-chip p300 boards; a bare single-chip open fails without the mesh descriptor.
    import os
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    # ---- each model at ITS OWN shipped defaults, not protenix's ----------------------------
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        patch_boltz2_cfg()

    # ---- decision counters: read the branch taken, never infer it from the shape -------------
    ORIG_TMC, ORIG_LN, ORIG_PPC = T._transpose_memory_config, T._l1_layer_norm, T._pair_proj_config

    def tmc(t):
        mc = ORIG_TMC(t)
        DEC[f"transpose|{'x'.join(str(int(d)) for d in t.shape)}"][
            "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"] += 1
        return mc

    def ln(x, headroom, **kw):
        out, in_l1 = ORIG_LN(x, headroom, **kw)
        DEC[f"l1_layer_norm|h={headroom}|{'x'.join(str(int(d)) for d in x.shape)}"][
            "L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    def ppc(x, w, bw_cap=-1, out_l1=False):
        cfg = ORIG_PPC(x, w, bw_cap=bw_cap, out_l1=out_l1)
        if out_l1:
            DEC[f"pair_proj_out_l1|{'x'.join(str(int(d)) for d in x.shape)}"
                f"@{int(list(w.shape)[-1])}"]["L1" if cfg is not None else "DRAM"] += 1
        return cfg

    T._transpose_memory_config = tmc
    T._l1_layer_norm = ln
    T._pair_proj_config = ppc
    try:
        import tt_bio.protenix as P
        P._l1_layer_norm = ln       # imported by name there, so both namespaces need it
    except Exception:               # noqa: BLE001
        pass

    # `_qkv_mm_config` is the dict lookup that decides whether K1 can serve at all. Census the
    # (kt, nt) pairs each model presents so a missing `_MM_BLOCK` entry is a measured fact.
    ORIG_QKVMM = T._qkv_mm_config

    def qkvmm(x, w, *args, **kw):
        cfg = ORIG_QKVMM(x, w, *args, **kw)
        kt, nt = int(x.shape[-1]) // 32, int(w.shape[-1]) // 32
        DEC[f"qkv_mm_config|kt={kt},nt={nt}"]["config" if cfg is not None else "None"] += 1
        return cfg

    T._qkv_mm_config = qkvmm

    # The group is the whole point of the g12 arm, so read it back rather than infer it: an arm
    # that flips the cap and still returns 4 is an A/A pair wearing an A/B's label.
    ORIG_GROUP = T._trimul_inproj_group          # this branch: largest divisor at or below the cap

    def group_halving(seq_len, chunk, batch, n_pairs):
        """main's search, verbatim. Restoring the cap to 8 is NOT enough to restore main: at
        n_pairs = 12 main halves 8 -> 4 while a divisor search at cap 8 returns 6, so an `on` arm
        that only moved the constant would be a third behaviour and not a baseline."""
        fused = 4 * chunk * seq_len * seq_len * batch * 2
        g = 8
        while g > 1 and (n_pairs % g or g * fused > T._TRIMUL_INPROJ_FUSED_BYTES):
            g //= 2
        return g

    GROUP_SEARCH = [ORIG_GROUP]

    def group_census(seq_len, chunk, batch, n_pairs):
        g = GROUP_SEARCH[0](seq_len, chunk, batch, n_pairs)
        GROUPS[f"S={seq_len},n_pairs={n_pairs}->g={g}"] += 1
        return g

    T._trimul_inproj_group = group_census

    ORIG_MM_BLOCK_FOR = T._mm_block_for

    ORIG_TAS = T._tri_att_sdpa

    def tas(qq, kk, vv, bias, scale):
        ql, kl = int(qq.shape[2]), int(kk.shape[2])
        fits = [c for c in T._tri_att_q_chunks(ql, kl)
                if (ql, kl, c) not in T._SDPA_Q_CHUNK_OVER_L1]
        DEC[f"tri_att_sdpa|q{ql}k{kl}"][f"q_chunk={fits[0] if fits else None}"] += 1
        return ORIG_TAS(qq, kk, vv, bias, scale)

    T._tri_att_sdpa = tas

    # ---- timers, probed rather than assumed --------------------------------------------------
    installed = []
    import tt_bio.esmfold2 as E2

    def patch(mod, name, key):
        cls = getattr(mod, name, None)
        if cls is None or not hasattr(cls, "__call__"):
            return
        f = cls.__call__
        cls.__call__ = (lambda g: lambda self, *x, **k: timed_call(key, g, self, *x, **k))(f)
        installed.append(key)

    patch(T, "Pairformer", "stage:Pairformer")
    patch(T, "PairformerLayer", "block:PairformerLayer")
    for nm in ("TriangleMultiplication", "TriangleAttention", "AttentionPairBias",
               "PairWeightedAveraging"):
        patch(T, nm, f"body:{nm}")
    if a.model.startswith("esmfold2"):
        patch(E2, "PairUpdateBlock", "block:PairUpdateBlock")
        patch(E2, "FoldingTrunkModel", "stage:FoldingTrunk")

    def set_arm(name):
        """Every lever is written on every arm, so an arm provably runs its own state."""
        assert name in ARMS, f"unknown arm {name}"
        import tt_bio.reblock_permute as RB
        import tt_bio.triatt_qkv as HM
        import tt_bio.triatt_sdpa as PM

        RB.set_enabled(True)                         # main ships the forward move ON
        RB.set_enabled_back(True)                    # and the back move ON
        RB.set_enabled_gated(name in ("e6", "all"))   # the master switch; models opt in per instance
        RB.STATS[0] = RB.STATS[1] = 0
        RB.STATS_BACK[0] = RB.STATS_BACK[1] = 0
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        RB.REJECTS.clear()

        HM._ENABLED = name != "nok1"
        HM._TAIL_ENABLED = name != "nok1"
        HM._TAIL_OVER_L1 = True
        HM.STATS[0] = HM.STATS[1] = 0
        HM.TAIL_STATS[0] = HM.TAIL_STATS[1] = 0
        HM.REJECTS.clear()
        HM.TAIL_REJECTS.clear()

        PM._ENABLED = name != "nok2"
        PM.STATS[0] = PM.STATS[1] = 0
        PM.REJECTS.clear()

        GROUP_SEARCH[0] = ORIG_GROUP if name in ("g12", "all") else group_halving
        T._TRIMUL_INPROJ_GROUP = 12 if name in ("g12", "all") else 8
        for k, v in _MM_BLOCK_ODDE.items():
            if name in ("mm12", "all"):
                T._MM_BLOCK[k] = v
            else:
                T._MM_BLOCK.pop(k, None)
        GROUPS.clear()
        T._TRANSPOSE_L1_HEADROOM = 1.25 if name == "tr125" else 2.5
        T._PAIR_PROJ_MM = name != "nomm"
        T._mm_block_for = _mm_block_old if name == "oldkey" else ORIG_MM_BLOCK_FOR
        OLDKEY_HITS[0] = OLDKEY_HITS[1] = 0
        T._MM_BLOCK[(8, 8)] = (2, 8, 1, 2, 1) if name == "nomm" else (4, 8, 1, 4, 1)
        # `nonewmm` reverts ONLY the four widths this task added, so the A/B isolates
        # the re-key from the two engine-wide levers `nomm` also moves.
        for k in ((4, 12), (4, 4), (2, 12), (2, 2)):
            if name == "nonewmm":
                T._MM_BLOCK.pop(k, None)
            else:
                T._MM_BLOCK[k] = (4, k[0], 1, 4, 1)

        # capacity gates stay at production defaults on every arm: they are not under test here
        for mod in FP32_OWNERS:
            mod.fp32_softmax = name not in ("nofp32", "nofp32hifi")
        PM._CKC_OVERRIDE = CKC_HIFI if name == "nofp32hifi" else None

        T._PAIR_PROJ_L1_OUT = T._PAIR_BIAS_L1_NORM = True
        T._PWA_L1_NORM = T._TEMPLATE_L1_NORM = True
        T._pair_proj_program_config.cache_clear()
        T._tri_att_q_chunks.cache_clear()
        T._L1_OUT_REFUSED.clear()

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "timers_installed": installed, "runs": []}

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        set_arm("on")
        one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{size}", tgt, a3m)
        STATE["dev"] = T.get_device()
        STATE["model"] = getattr(state, "model", None)
        FP32_OWNERS[:] = _collect_fp32(STATE["model"])
        res["fp32_softmax_modules"] = len(FP32_OWNERS)
        print(f"  fp32_softmax attention modules: {len(FP32_OWNERS)}", flush=True)
        g = STATE["dev"].compute_with_storage_grid_size()
        res["grid"] = [g.x, g.y]
        res["compute_grid_main"] = list(T.COMPUTE_GRID_MAIN)
        struct_dir = Path(meta["struct_dir"])
        print(f"=== {a.model} size {size} rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS}: cold ===",
              flush=True)
        try:
            cold_s, cold_m = one_fold()
            print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
                  f"plddt={cold_m.get('plddt')}", flush=True)
        except Exception as e:                                                  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["runs"].append({"size": size, "arm": "cold",
                                "error": f"{type(e).__name__}: {e}"[:600]})
            a.out.write_text(json.dumps(res, indent=1))
            continue

        for arm in a.arms.split(","):
            set_arm(arm)
            WALL.clear(); DEC.clear()
            try:
                fold_s, m = one_fold()
            except Exception as e:                                              # noqa: BLE001
                res["runs"].append({"size": size, "arm": arm,
                                    "error": f"{type(e).__name__}: {e}"[:600]})
                a.out.write_text(json.dumps(res, indent=1))
                print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            RB = __import__("tt_bio.reblock_permute", fromlist=["x"])
            HM = __import__("tt_bio.triatt_qkv", fromlist=["x"])
            PM = __import__("tt_bio.triatt_sdpa", fromlist=["x"])
            rec = {"size": size, "arm": arm, "fold_s": round(fold_s, 3),
                   "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                   "cif_sha256": sha_dir(struct_dir),
                   "reblock_fwd": [RB._ENABLED, list(RB.STATS)],
                   "reblock_back": [RB._ENABLED_BACK, list(RB.STATS_BACK)],
                   "gated_kernel": [RB._ENABLED_GATED, list(RB.STATS_GATED)],
                   "head_major_qkv": {"enabled": HM._ENABLED, "served": HM.STATS[0],
                                      "declined": HM.STATS[1],
                                      "rejects": {f"{r}:{sh}": n for (r, sh), n in HM.REJECTS.items()},
                                      "tail_enabled": HM._TAIL_ENABLED,
                                      "tail_served": HM.TAIL_STATS[0],
                                      "tail_declined": HM.TAIL_STATS[1],
                                      "tail_rejects": {f"{r}:{sh}": n
                                                       for (r, sh), n in HM.TAIL_REJECTS.items()}},
                   "persistent_mask": {"enabled": PM._ENABLED, "served": PM.STATS[0],
                                       "declined": PM.STATS[1],
                                       "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}},
                   "transpose_l1_headroom": T._TRANSPOSE_L1_HEADROOM,
                   "trimul_inproj_group_cap": T._TRIMUL_INPROJ_GROUP,
                   "group_search": "divisor" if GROUP_SEARCH[0] is ORIG_GROUP else "halving",
                   "trimul_inproj_groups": dict(GROUPS),
                   "gated_move_instances": sum(
                       1 for m in _walk_trimuls(STATE["model"]) if getattr(m, "gated_move", False)),
                   "fp32_softmax_modules": len(FP32_OWNERS),
                   "sdpa_ckc_override": (None if PM._CKC_OVERRIDE is None
                                         else [str(PM._CKC_OVERRIDE[0]).rsplit(".", 1)[-1],
                                               *map(bool, PM._CKC_OVERRIDE[1:])]),
                   "fp32_softmax_on": [bool(getattr(m, "fp32_softmax", False)) for m in FP32_OWNERS][:4],
                   "pair_proj_mm": T._PAIR_PROJ_MM, "mm_block": {str(k): list(x) for k, x in sorted(T._MM_BLOCK.items())},
                   "oldkey_lookup": {"active": T._mm_block_for is not ORIG_MM_BLOCK_FOR,
                                     "hit": OLDKEY_HITS[0], "miss": OLDKEY_HITS[1]},
                   "sdpa_q_chunk_over_l1": sorted(str(k) for k in T._SDPA_Q_CHUNK_OVER_L1),
                   "loadavg": open("/proc/loadavg").read().split()[:3],
                   "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                               for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
                   "decisions": {k: dict(v) for k, v in sorted(DEC.items())}}
            for key, short in (("block:PairformerLayer", "block"),
                               ("block:PairUpdateBlock", "block"),
                               ("body:TriangleAttention", "triatt"),
                               ("body:TriangleMultiplication", "trimul")):
                w = rec["wall_ms"].get(key)
                if w and rec.get(f"{short}_wall_ms") is None:
                    rec[f"{short}_wall_ms"], rec[f"{short}_calls"] = w["ms"], w["calls"]
            if a.keep_cif is not None:
                dst = a.keep_cif / f"{size}_{arm}_{len(res['runs'])}"
                dst.mkdir(parents=True, exist_ok=True)
                for p in sorted(struct_dir.glob("*")):
                    if p.is_file():
                        (dst / p.name).write_bytes(p.read_bytes())
                rec["cif_dir"] = str(dst)
            res["runs"].append(rec)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm}: fold {fold_s:.2f}s  block {rec.get('block_wall_ms')} ms over "
                  f"{rec.get('block_calls')} calls  plddt {m.get('plddt')}", flush=True)
            print(f"      group {rec['trimul_inproj_groups']}  gated_move instances "
                  f"{rec['gated_move_instances']}", flush=True)
            print(f"      K1 {rec['head_major_qkv']['served']}/{rec['head_major_qkv']['declined']} "
                  f"{rec['head_major_qkv']['rejects']}  K2 {rec['persistent_mask']['served']}/"
                  f"{rec['persistent_mask']['declined']} {rec['persistent_mask']['rejects']}  "
                  f"E6 {rec['gated_kernel']}", flush=True)
            for k, v in sorted(DEC.items()):
                if k.startswith(("qkv_mm_config", "transpose", "tri_att_sdpa")):
                    print(f"      DEC {k:46s} {dict(v)}", flush=True)

    per = {}
    for size in sorted({r["size"] for r in res["runs"] if "fold_s" in r}):
        d = {}
        for r in res["runs"]:
            if r.get("size") != size or "fold_s" not in r:
                continue
            d.setdefault(r["arm"], {"fold_s": [], "block_ms": [], "plddt": []})
            d[r["arm"]]["fold_s"].append(r["fold_s"])
            d[r["arm"]]["block_ms"].append(r.get("block_wall_ms"))
            d[r["arm"]]["plddt"].append(r.get("plddt"))
        for arm, v in d.items():
            v["n"] = len(v["fold_s"])
            v["fold_s_median"] = round(st.median(v["fold_s"]), 3)
            v["fold_s_min"] = round(min(v["fold_s"]), 3)
            v["fold_s_spread"] = round(max(v["fold_s"]) - min(v["fold_s"]), 3)
            bm = [x for x in v["block_ms"] if x is not None]
            if bm:
                v["block_ms_median"] = round(st.median(bm), 2)
                v["block_ms_spread"] = round(max(bm) - min(bm), 2)
        if "on" in d and d["on"]["n"] > 1:
            d["aa_floor_fold_s"] = d["on"]["fold_s_spread"]
            d["aa_floor_block_ms"] = d["on"].get("block_ms_spread")
        per[size] = d
    res["arm_medians"] = per
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(per, indent=1), flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
