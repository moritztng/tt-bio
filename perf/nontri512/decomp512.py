#!/usr/bin/env python3
"""Phase 1 of fold-nontriangle-below-4x: where the 53.460 s at 512 aa actually goes.

The brief's split (TriMul ~16 s, TriAtt ~12 s, everything else ~25 s) is DERIVED. This measures
it. Two instrument levels, because they perturb differently:

  --level coarse   ~10 syncs/fold. Top-level stages only (trunk, diffusion, confidence, the host
                   legs). Cheap enough that the fold wall it reports is comparable to the
                   uninstrumented wall.
  --level fine     adds class-level wrappers inside the trunk block and inside the diffusion
                   sampler. Tens of thousands of syncs, so the fold wall from this level is
                   instrument-inflated and is an ATTRIBUTION instrument only. Never quote its
                   fold wall as a result.
  --level none     no wrappers at all. This is the honest uninstrumented fold wall, which is the
                   control for how much the 53.460 s baseline owes to its own instrumentation
                   (fold_ab512.py runs ~16.7k syncs per fold in both of its arms).

The arm is the integrated one (`wk/integrated-ab-h200-gap` @ c02f2e9b): E6 + K1 + K1b + K2 + the
L1 transpose destination. It is not asserted, it is proved from the gate counters after the fold
against the counts Part 2 of integrated-ab-h200-gap.md recorded (E6 2416, K1 1048, K1b 1048,
K2 1208, transpose L1 1048/1048).

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:fold-nontriangle-below-4x \
      python3 perf/nontri512/decomp512.py --level fine --repeat 2 \
        --out perf/nontri512/decomp512_fine.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEPTH = [0]
CTX = ["fold"]          # stack of enclosing instrumented names, so a Transition inside a
                        # Pairformer block is a different row from one inside the sampler
TOP = OrderedDict()
FOLD_MARKS = []
STATE = {"dev": None}


def _sync():
    import ttnn
    import tt_bio.tenstorrent as T
    ttnn.synchronize_device(STATE["dev"] or T._device or T.get_device())


def counted(key, fn, *a, **kw):
    """Class-level wall: sync on both sides, sum into WALL[key]."""
    _sync()
    t0 = time.perf_counter()
    CTX.append(key)
    try:
        return fn(*a, **kw)
    finally:
        CTX.pop()
        _sync()
        e = WALL[f"{CTX[-1]} > {key}"]
        e["n"] += 1
        e["s"] += time.perf_counter() - t0


def staged(name, fn):
    """Top-level stage wrapper: charged only at nesting depth 0, so a `_to_host` inside
    `edm_sample` lands on the diffusion stage instead of being counted twice."""
    def w(*a, **k):
        _sync()
        t0 = time.perf_counter()
        DEPTH[0] += 1
        CTX.append(name)
        try:
            return fn(*a, **k)
        finally:
            CTX.pop()
            DEPTH[0] -= 1
            _sync()
            dt = time.perf_counter() - t0
            if DEPTH[0] == 0:
                e = TOP.setdefault(name, [0, 0.0])
                e[0] += 1
                e[1] += dt
    return w


def install(level):
    import tt_bio.protenix as P
    import tt_bio.tenstorrent as T

    if level == "none":
        pass
    else:
        P.edm_sample = staged("diffusion", P.edm_sample)
        P.Trunk.__call__ = staged("trunk", P.Trunk.__call__)
        P.ConfidenceHead.confidence = staged("confidence", P.ConfidenceHead.confidence)
        if hasattr(P.ConfidenceHead, "confidence_device"):
            P.ConfidenceHead.confidence_device = staged("confidence", P.ConfidenceHead.confidence_device)
        P.Protenix._generate_relp = staticmethod(
            staged("relp_host", P.Protenix.__dict__["_generate_relp"].__func__))
        P.Protenix._to_host = staticmethod(
            staged("to_host", P.Protenix.__dict__["_to_host"].__func__))
        P.Protenix._diffusion_pair_cond = staged("diff_pair_cond", P.Protenix._diffusion_pair_cond)
        P.Protenix._plm_z_term = staged("plm_z_term", P.Protenix._plm_z_term)
        P.Protenix._atom_feat_inputs = staged("atom_feat_host", P.Protenix._atom_feat_inputs)

    if level == "fine":
        # (module, attr, key). Nested on purpose: PairformerLayer contains TriMul/TriAtt/
        # Transition, DiffusionTransformerLayer contains AttentionPairBias and the conditioned
        # transition. Report the residual, never the sum.
        for cls, key in [
            (T.Pairformer, "stage:Pairformer"),
            (T.PairformerLayer, "block:PairformerLayer"),
            (T.TriangleMultiplication, "body:TriangleMultiplication"),
            (T.TriangleAttention, "body:TriangleAttention"),
            (T.AttentionPairBias, "body:AttentionPairBias"),
            (T.PairWeightedAveraging, "body:PairWeightedAveraging"),
            (T.Transition, "body:Transition"),
            (T.OuterProductMean, "body:OuterProductMean"),
            (T.MSA, "stage:MSA"),
            (T.MSALayer, "block:MSALayer"),
            (T.Miniformer, "stage:Miniformer"),
            (T.MiniformerLayer, "block:MiniformerLayer"),
            (T.MiniTriangularUpdate, "body:MiniTriangularUpdate"),
            (T.DiffusionTransformer, "stage:DiffusionTransformer"),
            (T.DiffusionTransformerLayer, "block:DiffusionTransformerLayer"),
            (T.ConditionedTransitionBlock, "body:ConditionedTransitionBlock"),
            (P.AtomTransformer, "block:AtomTransformer"),
            (P.AtomAttentionEncoder, "stage:AtomAttentionEncoder"),
            (P.TrunkInput, "stage:TrunkInput"),
        ]:
            f = cls.__call__
            cls.__call__ = (lambda g, kk: lambda self, *x, **k: counted(kk, g, self, *x, **k))(f, key)

    # Fold boundary: snapshot the depth-0 stage tally per fold.
    orig = P.Protenix.fold

    def fold(*a, **k):
        TOP.clear()
        WALL.clear()
        del CTX[1:]
        _sync()
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _sync()
            total = time.perf_counter() - t0
            stages = {n: [c, round(t, 3)] for n, (c, t) in TOP.items()}
            FOLD_MARKS.append(dict(
                total_s=round(total, 3),
                stages=stages,
                unattributed_s=round(total - sum(t for _, t in stages.values()), 3),
                wall_ms={k2: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                         for k2, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
            ))
    P.Protenix.fold = fold


def set_integrated_arm():
    """The four levers of `int`, set explicitly. Everything else stays at the production
    default, which is what makes this the branch's shipped configuration."""
    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_bio.triatt_qkv as HM
    import tt_bio.triatt_sdpa as PM
    T._PAIR_PROJ_MM = True
    T._MM_BLOCK[8] = (4, 8, 1, 4, 1)
    T._TRANSPOSE_L1_HEADROOM = T.TRANSPOSE_L1_HEADROOM
    T._TRANSPOSE_L1_REFUSED.clear()
    T._PAIR_PROJ_L1_OUT = T._PAIR_BIAS_L1_NORM = T._PWA_L1_NORM = T._TEMPLATE_L1_NORM = True
    T._pair_proj_program_config.cache_clear()
    T._L1_OUT_REFUSED.clear()
    T._tri_att_q_chunks.cache_clear()
    T._TRIMUL_INPROJ_GROUP = 8
    RB.set_enabled_gated(True)
    RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
    HM._ENABLED = HM._TAIL_ENABLED = True
    HM._TAIL_OVER_L1 = True
    HM.STATS[0] = HM.STATS[1] = 0
    HM.TAIL_STATS[0] = HM.TAIL_STATS[1] = 0
    HM.REJECTS.clear()
    HM.TAIL_REJECTS.clear()
    PM._ENABLED = True
    PM.STATS[0] = PM.STATS[1] = 0
    PM.REJECTS.clear()


def gate_state():
    import tt_bio.reblock_permute as RB
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_qkv as HM
    import tt_bio.triatt_sdpa as PM
    return {
        "e6_gated_kernel": [RB._ENABLED_GATED, list(RB.STATS_GATED)],
        "k1_head_major_qkv": {"enabled": HM._ENABLED, "served": HM.STATS[0], "declined": HM.STATS[1]},
        "k1b_tail": {"enabled": HM._TAIL_ENABLED, "over_l1": HM._TAIL_OVER_L1,
                     "served": HM.TAIL_STATS[0], "declined": HM.TAIL_STATS[1]},
        "k2_persistent_mask": {"enabled": PM._ENABLED, "served": PM.STATS[0], "declined": PM.STATS[1]},
        "transpose_l1_headroom": T._TRANSPOSE_L1_HEADROOM,
        "transpose_l1_refused": [str(k) for k in T._TRANSPOSE_L1_REFUSED],
        "trimul_inproj_group": T._TRIMUL_INPROJ_GROUP,
        "sdpa_wide_q": T._SDPA_WIDE_Q,
        "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["none", "coarse", "fine"], default="coarse")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T  # noqa: F401  (import before patching)
    set_integrated_arm()
    install(a.level)

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res = B.measure("protenix-v2", a.repeat, ROOT / f".msa_s512_{a.size}", a.out,
                    tgt, a3m, f"{a.size} aa nontri decomp level={a.level}")

    set_arm_proof = gate_state()
    out = dict(res)
    out["level"] = a.level
    out["gate_state_after"] = set_arm_proof
    out["folds"] = FOLD_MARKS          # [0] cold, [1:] warm
    a.out.write_text(json.dumps(out, indent=1, default=str))

    if not FOLD_MARKS:
        print("  no fold marks -- Protenix.fold was never the entry point", flush=True)
        return 0
    warm = FOLD_MARKS[1] if len(FOLD_MARKS) > 1 else FOLD_MARKS[-1]
    print(f"\n=== 512 aa, integrated arm, level={a.level} — WARM fold {warm['total_s']} s ===",
          flush=True)
    for n, (c, t) in sorted(warm["stages"].items(), key=lambda kv: -kv[1][1]):
        print(f"  stage {n:22s} n={c:<5d} {t:8.3f}s  {100*t/warm['total_s']:5.1f}%", flush=True)
    print(f"  stage {'unattributed':22s}         {warm['unattributed_s']:8.3f}s  "
          f"{100*warm['unattributed_s']/warm['total_s']:5.1f}%", flush=True)
    if warm.get("wall_ms"):
        print("  --- class-level walls (nested; residuals, not sums) ---", flush=True)
        for k, v in warm["wall_ms"].items():
            print(f"  {k:38s} n={v['calls']:<6d} {v['ms']/1000:8.3f}s  "
                  f"{100*v['ms']/1000/warm['total_s']:5.1f}%", flush=True)
    print("\n  gate proof:", json.dumps(set_arm_proof, default=str), flush=True)
    print("  wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
