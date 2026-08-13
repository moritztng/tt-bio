#!/usr/bin/env python3
"""ESMFold2 512 aa decomposition: where does the fold wall actually go?

Not a lever A/B. This is Phase 1 of THE PERF METHOD for a model that shares almost nothing
with protenix: it measures the fold wall untimed (the number that counts), then measures a
separate instrumented fold that attributes that wall to named components with inclusive and
EXCLUSIVE time, so a parent's cost is never double-counted against its children.

Every timed region brackets `ttnn.synchronize_device`, so the attribution is honest and the
instrumented fold is SLOWER than the plain one by construction. Both walls are reported and
the overhead is stated rather than hidden.

Component set is esmfold2's own, probed from `tt_bio.esmfold2` / `tt_bio.esmfold2_runtime`:
the shared engine only contributes TriangleMultiplication (`esmfold2.py:28-32`), so the
shared TriangleAttention / Pairformer timers used by the protenix-family harnesses would
install onto classes this model never constructs.
"""
import argparse, hashlib, json, os, statistics as st, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

TIMERS_ON = [False]
INCL = defaultdict(lambda: {"n": 0, "s": 0.0})
CHILD = defaultdict(float)          # key -> time spent inside nested timed regions
STACK = []                          # (key, child_accum_index)
DEV = [None]


def timed(key, fn, *a, **kw):
    if not TIMERS_ON[0]:
        return fn(*a, **kw)
    import ttnn
    ttnn.synchronize_device(DEV[0])
    STACK.append([key, 0.0])
    t0 = time.perf_counter()
    try:
        out = fn(*a, **kw)
    finally:
        import ttnn as _t
        _t.synchronize_device(DEV[0])
        el = time.perf_counter() - t0
        me = STACK.pop()
        rec = INCL[key]
        rec["n"] += 1
        rec["s"] += el
        CHILD[key] += me[1]
        if STACK:
            STACK[-1][1] += el
    return out


def patch(mod, name, key, meth="__call__"):
    cls = getattr(mod, name, None)
    if cls is None:
        return None
    f = getattr(cls, meth, None)
    if f is None:
        return None
    setattr(cls, meth, (lambda g: lambda self, *x, **k: timed(key, g, self, *x, **k))(f))
    return key


def patch_fn(mod, name, key):
    f = getattr(mod, name, None)
    if f is None:
        return None
    setattr(mod, name, (lambda g: lambda *x, **k: timed(key, g, *x, **k))(f))
    return key


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--plain", type=int, default=2, help="untimed folds (the real wall + A/A)")
    ap.add_argument("--timed", type=int, default=1, help="instrumented folds (attribution)")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import tt_bio.esmfold2 as E2
    import tt_bio.esmfold2_runtime as ER
    import tt_bio.esmc as EC
    sys.path.insert(0, str(ROOT / "perf" / "esm2sizes"))
    import levers as LV
    LV.install()

    installed = []
    # --- stages ---------------------------------------------------------------------------
    installed.append(patch(ER, "_ESMCAdapter", "stage:ESMC-6B"))
    installed.append(patch(E2, "LanguageModelShimModel", "stage:LMShim"))
    installed.append(patch(E2, "MSAEncoderModel", "stage:MSAEncoder"))
    installed.append(patch(E2, "FoldingTrunkModel", "stage:FoldingTrunk*"))  # trunk AND lm_encoder
    installed.append(patch(E2, "DiffusionModuleModel", "stage:DiffusionModule", meth="step"))
    installed.append(patch(E2, "DiffusionConditioningModel", "stage:DiffusionConditioning"))
    installed.append(patch(E2, "DistogramHead", "stage:DistogramHead"))
    # --- blocks / bodies ------------------------------------------------------------------
    installed.append(patch(E2, "PairUpdateBlock", "block:PairUpdateBlock"))
    installed.append(patch(T, "TriangleMultiplication", "body:TriangleMultiplication"))
    installed.append(patch(EC, "SwiGLUFFN", "body:SwiGLUFFN"))
    installed.append(patch(E2, "DiffusionTransformerModel", "block:TokenDiT"))
    installed.append(patch(E2, "AttentionPairBias", "body:AttentionPairBias"))
    installed.append(patch(E2, "ConditionedTransitionBlock", "body:CondTransition"))
    installed.append(patch_fn(E2, "_attn_fp32", "leaf:_attn_fp32"))
    installed.append(patch(E2, "AtomEncoder", "body:AtomEncoder"))
    installed.append(patch(E2, "AtomDecoder", "body:AtomDecoder"))
    installed.append(patch(E2, "SWAAtomTransformerModel", "body:SWAAtomTransformer"))
    installed.append(patch(E2, "SWAAttention", "leaf:SWAAttention"))
    installed.append(patch(E2, "OuterProductMean", "body:OuterProductMean"))
    # The three phases INSIDE fold_complex. p2 measured everything outside it at 0.043 s, so the
    # 5.16 s the p1 decomposition could not attribute is in here, not in featurization or the CIF
    # write. prepare_input and decode are host torch; model() is the device forward the stage
    # timers above already cover, so `model minus the stages` is what is genuinely unnamed.
    from tt_bio._vendor.esm.models.esmfold2 import processor as _P
    installed.append(patch(_P, "ESMFold2InputBuilder", "phase:prepare_input", meth="prepare_input"))
    installed.append(patch(_P, "ESMFold2InputBuilder", "phase:decode", meth="decode"))
    installed.append(patch(E2, "MSAPairWeightedAveraging", "body:MSAPWA"))
    installed = [k for k in installed if k]

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "timers_installed": installed, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{a.size}", tgt, a3m)
    DEV[0] = T.get_device()
    g = DEV[0].compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])

    print(f"=== {a.model} {a.size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS}: cold ===",
          flush=True)
    t0 = time.perf_counter()
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.3f}s (wall {time.perf_counter()-t0:.1f}s) "
          f"n_tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}", flush=True)
    res["cold_s"] = cold_s

    def record(tag, fold_s, m, detail=False):
        row = {"arm": tag, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "cif": sha_dir(struct_dir),
               "loadavg": open("/proc/loadavg").read().split()[0]}
        if detail:
            row["levers"] = LV.snapshot(reset_rb=True)
            comp = {}
            for k, v in sorted(INCL.items(), key=lambda kv: -kv[1]["s"]):
                comp[k] = {"n": v["n"], "incl_s": round(v["s"], 4),
                           "excl_s": round(v["s"] - CHILD[k], 4)}
            row["components"] = comp
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {tag}: {fold_s:.3f}s plddt={m.get('plddt')}", flush=True)

    TIMERS_ON[0] = False
    for i in range(a.plain):
        fold_s, m = one_fold()
        record(f"plain{i}", fold_s, m)

    for i in range(a.timed):
        INCL.clear(); CHILD.clear(); STACK.clear(); LV.reset()
        LV.snapshot(reset_rb=True)   # zero what the plain folds accumulated
        TIMERS_ON[0] = True
        LV.on(True)
        fold_s, m = one_fold()
        TIMERS_ON[0] = False
        LV.on(False)
        record(f"timed{i}", fold_s, m, detail=True)
        for k, v in sorted(INCL.items(), key=lambda kv: -(kv[1]["s"] - CHILD[kv[0]])):
            print(f"    {k:34s} n={v['n']:6d} incl={v['s']:8.3f}s excl={v['s']-CHILD[k]:8.3f}s",
                  flush=True)

    TIMERS_ON[0] = False
    for i in range(a.plain):
        fold_s, m = one_fold()
        record(f"plain_post{i}", fold_s, m)

    walls = [r["fold_s"] for r in res["runs"] if r["arm"].startswith("plain")]
    if len(walls) > 1:
        res["plain_median"] = st.median(walls)
        res["plain_spread"] = max(walls) - min(walls)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("plain_median", "plain_spread") if k in res}), flush=True)


if __name__ == "__main__":
    main()
