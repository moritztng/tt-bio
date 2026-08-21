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
      nos2   atom-level q pad as the pre-S2 ROW_MAJOR round trip; control for the shipped TILE pad
      nofp32 every `fp32_softmax=True` attention switched to the fused flash-SDPA route. openfold3
             is the only model that sets it (`openfold3_trunk.py:130`); this arm is the SCREEN for a
             fused fp32-softmax kernel, and it is an upper bound on that kernel by construction --
             it buys the whole traffic saving and pays none of the fp32 reduction cost, so a kernel
             that keeps the reduction in fp32 lands at or below it. It is NOT bit-exact and is not a
             shipping proposal; its plDDT against the `on` arm is the accuracy number it exists for.

A lever that reports `served 0, declined 0` is UNTESTED, not inactive -- the counters distinguish
"gated off with a reason" from "fired and did nothing", which is the whole point of Phase 0.
"""
import argparse, hashlib, json, os, shutil, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
GROUPS = Counter()
STATE = {"dev": None, "model": None}
FP32_OWNERS: list = []
# id(module) -> "trunk" | "msa" | "template" | "confidence". Populated AFTER the cold fold,
# because `OF3Fold.confidence_head` is None until the first `_confidence()` call
# (openfold3_fold.py:200,250-252) -- collecting before it exists silently leaves the head's
# 8 calls unflipped on every arm, which reads as "K2 declined 8" and is not that at all.
OWNED: dict = {}
CALLS = Counter()      # (owner, class, c_z, fp32_softmax) -> calls, reset per arm


def _collect_owned(model):
    """Partition the fp32_softmax modules by the site that owns them."""
    OWNED.clear()
    t = getattr(model, "trunk", None)
    for label, root in (("trunk", getattr(t, "pairformer", None)),
                        ("msa", getattr(t, "msa_module", None)),
                        ("template", getattr(t, "template", None)),
                        ("confidence", getattr(model, "confidence_head", None))):
        for m in _collect_fp32(root):
            OWNED.setdefault(id(m), label)          # first owner wins, no double-count
    return OWNED


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

ARMS = ("on", "e6", "noe6", "nok1", "nok2", "tr125", "nomm", "nofp32", "nofp32hifi",
        "nonewmm", "oldkey", "nofp32_trunk", "nofp32_msatmpl", "nos2",
        # sizes-recheck. `noqsplit` ablates the SDPA q-split that main ships ON up to
        # 1024 padded tokens; `tr250` restores the pre-227cdb41 transpose headroom.
        "noqsplit", "tr250",
        # openfold3-sizes-perf. `nofuse` is the pre-change baseline for the fp32-softmax chain
        # (scale as its own pass, out-of-place softmax); `norowblk` lifts the row-block budget so
        # the score tensor is one allocation whatever its size, which is what refuses at 1024 aa.
        # `hchunk16` and `noL1out` ablate the two levers whose gates are only live BELOW 384 aa.
        "nofuse", "norowblk", "blk4g", "hchunk16", "noL1out", "pre",
        # openfold3-to-3x-perdollar. `noshard` zeroes the per-core L1 budget so the fp32-softmax
        # tail runs DRAM-interleaved, i.e. main verbatim. `on` is the shipped default.
        "noshard",
        # qsplit: the triatt_sdpa q-split lever (TT_BIO_TRIATT_MASK_Q_SPLIT), written explicitly
        # per arm so "on" stays a pre-lever reference whatever the shipped default is (`noqsplit`
        # above is the same ablation, added first).
        "qsplit",
        # opendde-size-generality. `devcat` resolves the host-concat budget per part (the shipped
        # form of concat_host_bytes()); `on` pins it at the 12 GiB-Wormhole base, which is what
        # main shipped to every part. On a 31.875 GiB p150a that is 3.984 vs 1.5 GiB, so the
        # OpenDDE refiner's pair channel join runs on device from 768 aa up instead of on the host.
        # `devcat` is not byte-identical at 768 aa (p2_fold_ab_768_qb1c0.json), and there are two
        # live sites: the trimul channel join in tenstorrent.py and opendde.py's z_struct assembly
        # at the trunk-to-refiner seam. `devcat_trimul` widens the budget everywhere EXCEPT the
        # seam, `devcat_zstruct` widens ONLY the seam, so one 768 aa fold each says which site
        # carries the difference. Both work by rebinding the name opendde.py imported, so no
        # product code exists for the screen.
        # `hostcat` is the converse control: a 1-byte budget, so EVERY host-concat site takes the
        # host branch whatever the size. It exists because the budget's own threshold puts the
        # host/device split out of reach below 768 aa, and 768 aa's only fixture is the chimeric
        # cdk2x2 whose hinge cannot score a non-bit-exact change
        # (`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). Pairing `on` (device
        # branch below the base budget) with `hostcat` at 298 aa runs the same two branches on the
        # monomeric fixture, where an RMSD IS readable.
        "devcat", "devcat_trimul", "devcat_zstruct", "hostcat")

# Which sites each arm routes onto the fused SDPA. The confidence head is never in a flip set:
# it stays on `_fp32_softmax_attention` on every arm, deliberately, so plDDT reports on the
# trunk embeddings rather than on its own perturbation.
FLIP = {"nofp32":         {"trunk", "msa", "template"},
        "nofp32hifi":     {"trunk", "msa", "template"},
        "nofp32_trunk":   {"trunk"},
        "nofp32_msatmpl": {"msa", "template"}}

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


# `TT_BIO_AB_TRIMUL_POP=1` splits the TriangleMultiplication timer by call population instead of
# lumping all 1216 calls under one key. OpenDDE drives three of them -- trunk (H = seq, hidden 384),
# a narrow-hidden one, and the refiner at H ~ 1.95 * seq -- and only the refiner's fused
# in-projection loses a doubling between 640 and 768 aa. Off by default: every other arm in this
# harness shares the WALL key set and must keep seeing one key per class.
POP_SPLIT = os.environ.get("TT_BIO_AB_TRIMUL_POP", "0") == "1"


def _trimul_pop(key, self, args):
    try:
        return f"{key}|H={int(args[0].shape[1])}|hid={self._hidden}"
    except Exception:  # noqa: BLE001 -- a timer must never be the thing that fails a fold
        return key


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
    ap.add_argument("--dram-tags", default="",
                    help="Comma-separated tag prefixes the DRAM probe is allowed to sample. "
                         "dram_peak() fires at ~12k sites per opendde fold, most of them inside "
                         "chunked loops, and each one is a get_memory_view pipeline drain -- "
                         "enough to take a 768 aa devcat fold from 7 to 28+ min. Worse, the cost "
                         "is NOT arm-neutral: the drain is more expensive on the arm with more "
                         "resident device blocks, which is exactly the axis a residency lever "
                         "moves, so an unfiltered probed run cannot even be compared to itself "
                         "across arms. 'pairformer' keeps the per-block boundaries (620 samples), "
                         "where a resident pair tensor is still live, and drops the per-chunk "
                         "interior.")
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    global CKC_HIFI
    CKC_HIFI = (ttnn.MathFidelity.HiFi4, False, True, False)
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    # qb2 is two dual-chip p300 boards; a bare single-chip open fails without the mesh descriptor.
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    if a.dram_tags:
        _keep, _peak = tuple(a.dram_tags.split(",")), T.dram_peak
        # tenstorrent.py's sites call the module global, so rebinding it here reaches all of them.
        T.dram_peak = lambda tag=None: (_peak(tag) if tag is None or tag.startswith(_keep)
                                        else _peak(None))

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

    def ppc(x, w, bw_cap=-1, out_l1=False, **kw):
        # `**kw`, because this wrapper is a census and must not pin the signature it wraps.
        # `_pair_proj_config` gained `block_w` after this harness was written and every model this
        # script folds died on `ppc() got an unexpected keyword argument 'block_w'` before the cold
        # fold finished -- a census wrapper that rejects a new argument turns a neutrality check
        # into a TypeError.
        cfg = ORIG_PPC(x, w, bw_cap=bw_cap, out_l1=out_l1, **kw)
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

    # Read the in-projection divisor group back off the live fold rather than inferring it:
    # an arm that ships the divisor search and still returns 4 at 640 aa is an inert lever.
    ORIG_GROUP = T._trimul_inproj_group

    def group_census(seq_len, chunk, batch, n_pairs):
        g = ORIG_GROUP(seq_len, chunk, batch, n_pairs)
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

    def patch(mod, name, key, keyfn=None):
        cls = getattr(mod, name, None)
        if cls is None or not hasattr(cls, "__call__"):
            return
        f = cls.__call__
        if keyfn is None:
            cls.__call__ = (lambda g: lambda self, *x, **k: timed_call(key, g, self, *x, **k))(f)
        else:
            cls.__call__ = (lambda g: lambda self, *x, **k:
                            timed_call(keyfn(key, self, x), g, self, *x, **k))(f)
        installed.append(key)

    patch(T, "Pairformer", "stage:Pairformer")
    patch(T, "PairformerLayer", "block:PairformerLayer")
    for nm in ("TriangleMultiplication", "TriangleAttention", "AttentionPairBias",
               "PairWeightedAveraging"):
        patch(T, nm, f"body:{nm}",
              _trimul_pop if POP_SPLIT and nm == "TriangleMultiplication" else None)
    if a.model.startswith("esmfold2"):
        patch(E2, "PairUpdateBlock", "block:PairUpdateBlock")
        patch(E2, "FoldingTrunkModel", "stage:FoldingTrunk")

    # Per-owner CALL census. Module counts are not call counts -- 156 modules against 488 calls,
    # because `attn_pair_bias` is flipped but is not a TriangleAttention and template blocks are
    # re-entered once per template per cycle. Count calls, per owner, per c_z.
    def census(mod, name):
        cls = getattr(mod, name, None)
        if cls is None or not hasattr(cls, "__call__"):
            return
        f = cls.__call__

        def wrapped(self, *x, **k):
            cz = None
            for arg in x:
                sh = getattr(arg, "shape", None)
                if sh is not None and len(sh) >= 1:
                    cz = int(sh[-1])
                    break
            CALLS[(OWNED.get(id(self), "unowned"), name, cz,
                   bool(getattr(self, "fp32_softmax", False)))] += 1
            return f(self, *x, **k)

        cls.__call__ = wrapped

    for nm in ("TriangleAttention", "AttentionPairBias"):
        census(T, nm)

    # Arm `on` has to BE main, so read the shipped defaults off the module instead of
    # restating them as literals here. Both literals this replaced had gone stale: main moved
    # _TRANSPOSE_L1_HEADROOM 2.5 -> 1.25 at 227cdb41 (2026-08-15) and defaulted the SDPA q-split
    # ON at d31c1fa0/063f89db, so every `on` arm run after 08-15 measured a configuration main
    # does not ship -- and both levers are size-conditional, which is exactly where it matters.
    SHIPPED = {"headroom": T.TRANSPOSE_L1_HEADROOM,
               "q_split": __import__("tt_bio.triatt_sdpa", fromlist=["x"])._Q_SPLIT}

    def set_arm(name):
        """Every lever is written on every arm, so an arm provably runs its own state."""
        assert name in ARMS, f"unknown arm {name}"
        import tt_bio.reblock_permute as RB
        import tt_bio.triatt_qkv as HM
        import tt_bio.triatt_sdpa as PM

        RB.set_enabled(True)                         # main ships the forward move ON
        RB.set_enabled_back(True)                    # and the back move ON
        RB.set_enabled_gated(name != "noe6")         # main ships E6 ON; noe6 ablates it
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
        PM._Q_SPLIT = {"noqsplit": False, "qsplit": True}.get(name, SHIPPED["q_split"])
        PM.STATS[0] = PM.STATS[1] = 0
        PM.REJECTS.clear()

        T._TRANSPOSE_L1_HEADROOM = {"tr125": 1.25, "tr250": 2.5}.get(
            name, SHIPPED["headroom"])
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
        # `nofp32`/`nofp32hifi` flip every site except the confidence head, whatever the partition
        # says, so they reproduce the predecessor's arm exactly even if a module is unowned. Only
        # the two partial arms depend on OWNED being complete.
        flip = FLIP.get(name, frozenset())
        every_site = name in ("nofp32", "nofp32hifi")
        for mod in FP32_OWNERS:
            o = OWNED.get(id(mod), "unowned")
            mod.fp32_softmax = (o == "confidence") or not (every_site or o in flip)
        assert all(m.fp32_softmax for m in FP32_OWNERS
                   if OWNED.get(id(m)) == "confidence"), "confidence head must stay fp32"
        PM._CKC_OVERRIDE = CKC_HIFI if name == "nofp32hifi" else None

        # S2: pad the atom-level q in TILE instead of round-tripping through ROW_MAJOR. `nos2`
        # is the control -- the pre-S2 chain verbatim -- so the arm reads the same whichever way
        # the default is set.
        T._ATOM_PAD_IN_TILE = name != "nos2"

        # None means "resolve from this part's DRAM", i.e. exactly what a shipped fold does.
        # `on` pins the pre-change base so the A/B is the fix, not an unbounded budget.
        import tt_bio.opendde as OD
        T._CONCAT_HOST_BYTES = (None if name in ("devcat", "devcat_trimul")
                                else 1 if name == "hostcat"
                                else T.CONCAT_HOST_BYTES_BASE)
        # opendde.py from-imports the accessor, so rebinding it here moves the z_struct seam
        # alone. Resolved at call time, after the device is open, so it never reads 0.
        OD.concat_host_bytes = {
            "devcat_trimul": lambda: T.CONCAT_HOST_BYTES_BASE,
            "devcat_zstruct": lambda: T._concat_host_budget(T._dram_total_bytes()),
            "hostcat": lambda: 1,
        }.get(name, T.concat_host_bytes)

        T._PAIR_PROJ_L1_OUT = T._PAIR_BIAS_L1_NORM = True
        T._PWA_L1_NORM = T._TEMPLATE_L1_NORM = True
        # openfold3-sizes-perf arms. These come AFTER the blanket _PAIR_PROJ_L1_OUT assignment
        # above, or `noL1out` silently reads identical to `on`.
        # `pre` is main verbatim: neither lever. It is the control that says whether
        # 1024 aa folded before this branch, at the fold rather than off it.
        T._FP32_SOFTMAX_FUSED_ADD = name not in ("nofuse", "pre")
        T._FP32_SOFTMAX_BLOCK_BYTES = (1 << 62) if name in ("norowblk", "pre") else (
            (4 << 30) if name == "blk4g" else (8 << 30))
        T._FP32_SOFTMAX_L1_BYTES_PER_CORE = 0 if name == "noshard" else (768 << 10)
        T.FP32_SOFTMAX_STATS.update(calls=0, blocked=0, blocks=0, fused=0, unfused=0,
                                    l1=0, l1_blocks=0, l1_refused=0)
        T._FP32_SOFTMAX_L1_ROW_CAP.clear()
        T.TRANSITION_H_CHUNK_SIZE_BIG = 16 if name == "hchunk16" else 32
        T._PAIR_PROJ_L1_OUT = name != "noL1out"
        T._pair_proj_program_config.cache_clear()
        T._tri_att_q_chunks.cache_clear()
        T._L1_OUT_REFUSED.clear()
        # The L1-refusal memos are process-global, so an arm that throws retires a q_chunk for
        # every later arm -- that voided a reference arm at 1024 aa (wk/boltz2-sizes-perf e3b0b95b).
        # Every arm re-discovers its own refusals; the cost is one extra caught TT_THROW per shape.
        T._SDPA_Q_CHUNK_OVER_L1.clear()
        PM._PM_OVER_L1.clear()

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
            # ONLY now is the lazily-built confidence head present. Collect after the cold fold.
            FP32_OWNERS[:] = _collect_fp32(STATE["model"])
            _collect_owned(STATE["model"])
            by_owner = Counter(OWNED.get(id(m), "unowned") for m in FP32_OWNERS)
            res["fp32_softmax_modules"] = len(FP32_OWNERS)
            res["fp32_modules_by_owner"] = dict(by_owner)
            res["cold_call_census"] = {"|".join(map(str, k)): v for k, v in sorted(CALLS.items())}
            print(f"  fp32_softmax attention modules: {len(FP32_OWNERS)} {dict(by_owner)}",
                  flush=True)
            for k, v in sorted(CALLS.items()):
                print(f"      CENSUS {'|'.join(map(str, k)):48s} {v}", flush=True)
            if by_owner["unowned"] or (a.model == "openfold3"
                                       and set(by_owner) - {"unowned"} !=
                                       {"trunk", "msa", "template", "confidence"}):
                res["owner_partition_warning"] = dict(by_owner)
                print(f"  WARNING partition incomplete: {dict(by_owner)} -- the two partial arms "
                      f"are not interpretable; nofp32 still flips every non-confidence site",
                      flush=True)
        except Exception as e:                                                  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["runs"].append({"size": size, "arm": "cold",
                                "error": f"{type(e).__name__}: {e}"[:600]})
            a.out.write_text(json.dumps(res, indent=1))
            continue

        cif_keep = Path(__file__).resolve().parent / "cif"
        run_ix = Counter()
        for arm in a.arms.split(","):
            set_arm(arm)
            WALL.clear(); DEC.clear(); CALLS.clear(); GROUPS.clear()
            # Per-arm DRAM high-water mark. dram_peak() is a no-op returning 0 unless
            # TT_BIO_DRAM_PEAK names a file, and with it set the probe costs 2.4-3.7x wall
            # (its own docstring), so a run that reads this is a CAPACITY run and its
            # fold_s must not be quoted. Cleared per arm: the dict is module-global and
            # would otherwise carry arm N-1's peak into arm N and report the max of both.
            T._DRAM_PEAK.clear()
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
                   "group_search": "divisor",
                   "trimul_inproj_groups": dict(GROUPS),
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
                   "persistent_mask": {"enabled": PM._ENABLED, "q_split": PM._Q_SPLIT,
                                       "served": PM.STATS[0],
                                       "declined": PM.STATS[1],
                                       "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()},
                                       "pm_over_l1": sorted(str(k) for k in PM._PM_OVER_L1)},
                   "transpose_l1_headroom": T._TRANSPOSE_L1_HEADROOM,
                   # must differ between arms; equal values mean the arm did not take. The
                   # second is the z_struct seam, which the two isolation arms move on its own.
                   "dram_peak_gib": round(T.dram_peak() / 2 ** 30, 3) or None,
                   "dram_tags": a.dram_tags or None,
                   "concat_host_bytes": T.concat_host_bytes(),
                   "concat_host_bytes_zstruct": __import__(
                       "tt_bio.opendde", fromlist=["x"]).concat_host_bytes(),
                   "fp32_softmax_chain": {"block_bytes": T._FP32_SOFTMAX_BLOCK_BYTES,
                                          "fused_add": T._FP32_SOFTMAX_FUSED_ADD,
                                          "l1_bytes_per_core": T._FP32_SOFTMAX_L1_BYTES_PER_CORE,
                                          **dict(T.FP32_SOFTMAX_STATS)},
                   "transition_h_chunk_size_big": T.TRANSITION_H_CHUNK_SIZE_BIG,
                   "pair_proj_l1_out": T._PAIR_PROJ_L1_OUT,
                   "atom_pad_in_tile": T._ATOM_PAD_IN_TILE,
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
                   # VmHWM is a process high-water mark and never resets, and every arm of a run
                   # shares one process -- so this is monotone across arms and is NOT per-arm
                   # evidence. Equal values in two rows of the same run mean the later arm did not
                   # exceed the earlier one's peak, not that the two arms cost the same host RAM.
                   "maxrss_mb": round(int(next(l for l in open("/proc/self/status")
                                               if l.startswith("VmHWM")).split()[1]) / 1024, 1),
                   "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                               for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
                   "decisions": {k: dict(v) for k, v in sorted(DEC.items())}}
            # Keep the CIFs. The RMSD question died last time because the files were gone.
            run_ix[arm] += 1
            keep = cif_keep / f"{size}_{arm}_{run_ix[arm]}"
            keep.mkdir(parents=True, exist_ok=True)
            for p in sorted(struct_dir.glob("*")):
                if p.is_file():
                    shutil.copy2(p, keep / p.name)
            rec["cif_dir"] = str(keep.relative_to(ROOT))
            rec["call_census"] = {"|".join(map(str, k)): v for k, v in sorted(CALLS.items())}
            rec["fp32_on_by_owner"] = {
                o: sorted({bool(getattr(m, "fp32_softmax", False))
                           for m in FP32_OWNERS if OWNED.get(id(m)) == o})
                for o in sorted(set(OWNED.values()))}
            for key, short in (("block:PairformerLayer", "block"),
                               ("block:PairUpdateBlock", "block"),
                               ("body:TriangleAttention", "triatt"),
                               ("body:TriangleMultiplication", "trimul")):
                w = rec["wall_ms"].get(key)
                if w and rec.get(f"{short}_wall_ms") is None:
                    rec[f"{short}_wall_ms"], rec[f"{short}_calls"] = w["ms"], w["calls"]
            res["runs"].append(rec)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm}: fold {fold_s:.2f}s  block {rec.get('block_wall_ms')} ms over "
                  f"{rec.get('block_calls')} calls  plddt {m.get('plddt')}", flush=True)
            print(f"      owners {rec['fp32_on_by_owner']}", flush=True)
            for k, v in sorted(CALLS.items()):
                print(f"      CENSUS {'|'.join(map(str, k)):48s} {v}", flush=True)
            print(f"      FP32SOFT {rec['fp32_softmax_chain']}  hchunk "
                  f"{rec['transition_h_chunk_size_big']}  l1out {rec['pair_proj_l1_out']}",
                  flush=True)
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
