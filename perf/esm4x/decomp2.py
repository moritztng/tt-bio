#!/usr/bin/env python3
"""Where the 5.747 s that no timer covers actually goes.

`esmfold2-512aa-deep-perf` proved the remainder is inside `model()` and not in featurization, MSA
resolve, the CIF write, input prep or decode, and then called it "device work inside model()". That
last step was an inference, not a measurement: `model()` also runs the reference implementation's
host torch, and `patch_esmfold2` deliberately leaves some of it there -- the confidence head keeps
its O(L^2) glue (five pair-wide `nn.Linear`s, three `LayerNorm`s, a distance-bin `Embedding`, the
pae/pde heads and the row pooling) in fp32 on the CPU, and the parcae injection projection is an
`F.linear` on the host.

This harness is `perf/esm512/decomp.py` plus timers that can tell those apart:
  * every host `nn.Linear` / `nn.LayerNorm` / `nn.Embedding` / `F.linear` call inside the fold,
  * the vendor `ESMFold2Model.forward` and `ConfidenceHead.forward` walls,
  * every `TorchWrapper.forward` and `_Adapter.forward` boundary, which is where a submodule's
    host<->device transfer and tile-layout conversion live.

Timers are exclusive as well as inclusive, so `phase:model` minus its children is what is left.
"""
import argparse, hashlib, json, os, statistics as st, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

TIMERS_ON = [False]
INCL = defaultdict(lambda: {"n": 0, "s": 0.0})
CHILD = defaultdict(float)
STACK = []
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


def timed_host(key, fn, *a, **kw):
    """A host-only region: no device sync, so a CPU op is not charged for someone else's queue."""
    if not TIMERS_ON[0]:
        return fn(*a, **kw)
    STACK.append([key, 0.0])
    t0 = time.perf_counter()
    try:
        out = fn(*a, **kw)
    finally:
        el = time.perf_counter() - t0
        me = STACK.pop()
        rec = INCL[key]
        rec["n"] += 1
        rec["s"] += el
        CHILD[key] += me[1]
        if STACK:
            STACK[-1][1] += el
    return out


def patch(mod, name, key, meth="__call__", host=False):
    cls = getattr(mod, name, None)
    if cls is None:
        return None
    f = getattr(cls, meth, None)
    if f is None:
        return None
    t = timed_host if host else timed
    setattr(cls, meth, (lambda g: lambda self, *x, **k: t(key, g, self, *x, **k))(f))
    return key


def patch_fn(mod, name, key, host=False):
    f = getattr(mod, name, None)
    if f is None:
        return None
    t = timed_host if host else timed
    setattr(mod, name, (lambda g: lambda *x, **k: t(key, g, *x, **k))(f))
    return key


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--plain", type=int, default=1)
    ap.add_argument("--timed", type=int, default=1)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
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
    from tt_bio._vendor.esmfold2_hf import modeling_esmfold2 as MF

    installed = []
    # --- the two walls that bracket the hole -----------------------------------------------
    installed.append(patch(MF, "ESMFold2Model", "phase:model", meth="forward"))
    installed.append(patch(MF, "ConfidenceHead", "phase:confidence_head", meth="forward"))
    installed.append(patch(MF, "RowAttentionPooling", "host:RowAttentionPooling", meth="forward",
                           host=True))
    # --- host torch inside the fold: this is the claim being tested ------------------------
    installed.append(patch(torch.nn, "Linear", "host:nn.Linear", meth="forward", host=True))
    installed.append(patch(torch.nn, "LayerNorm", "host:nn.LayerNorm", meth="forward", host=True))
    installed.append(patch(torch.nn, "Embedding", "host:nn.Embedding", meth="forward", host=True))
    installed.append(patch_fn(torch.nn.functional, "linear", "host:F.linear", host=True))
    # --- the transfer boundaries -----------------------------------------------------------
    installed = [k for k in installed if k]
    for cls_name in ("FoldingTrunk", "MSAEncoder", "InputsEmbedder", "RelPosEncoding",
                     "LanguageModelShim", "DistogramHeadModel", "DiffusionConditioning",
                     "DiffusionTransformer", "SWAAtomTransformer", "DiffusionModule",
                     "StructureHead"):
        k = patch(E2, cls_name, f"wrap:{cls_name}", meth="forward")
        if k:
            installed.append(k)
    k = patch(E2, "StructureHead", "phase:structure_sample", meth="sample")
    if k:
        installed.append(k)
    k = patch(ER, "_Adapter", "wrap:_Adapter", meth="forward")
    if k:
        installed.append(k)
    # host<->device transfers, wherever a TorchWrapper does them
    for meth, key in (("_from_torch", "xfer:from_torch"), ("_to_torch", "xfer:to_torch")):
        f = getattr(T.TorchWrapper, meth)
        setattr(T.TorchWrapper, meth,
                (lambda g, kk: lambda self, *x, **k: timed(kk, g, self, *x, **k))(f, key))
        installed.append(key)
    # --- the same component set as perf/esm512/decomp.py, so the two are comparable --------
    installed.append(patch(ER, "_ESMCAdapter", "stage:ESMC-6B"))
    installed.append(patch(E2, "LanguageModelShimModel", "stage:LMShim"))
    installed.append(patch(E2, "MSAEncoderModel", "stage:MSAEncoder"))
    installed.append(patch(E2, "FoldingTrunkModel", "stage:FoldingTrunk*"))
    installed.append(patch(E2, "DiffusionModuleModel", "stage:DiffusionModule", meth="step"))
    installed.append(patch(E2, "DiffusionConditioningModel", "stage:DiffusionConditioning"))
    installed.append(patch(E2, "DistogramHead", "stage:DistogramHead"))
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
    installed.append(patch(E2, "MSAPairWeightedAveraging", "body:MSAPWA"))
    from tt_bio._vendor.esm.models.esmfold2 import processor as _P
    installed.append(patch(_P, "ESMFold2InputBuilder", "phase:prepare_input", meth="prepare_input"))
    installed.append(patch(_P, "ESMFold2InputBuilder", "phase:decode", meth="decode"))
    installed = [k for k in installed if k]

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "threads": torch.get_num_threads(), "timers_installed": installed, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_om512_{a.size}", tgt, a3m)
    DEV[0] = T.get_device()
    g = DEV[0].compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])

    print(f"=== {a.model} {a.size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS}: cold ===",
          flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.3f}s plddt={cold_m.get('plddt')}", flush=True)
    res["cold_s"] = cold_s

    def record(tag, fold_s, m):
        row = {"arm": tag, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "cif": sha_dir(struct_dir),
               "loadavg": open("/proc/loadavg").read().split()[0]}
        if TIMERS_ON[0]:
            row["components"] = {
                k: {"n": v["n"], "incl_s": round(v["s"], 4),
                    "excl_s": round(v["s"] - CHILD[k], 4)}
                for k, v in sorted(INCL.items(), key=lambda kv: -kv[1]["s"])}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {tag}: {fold_s:.3f}s plddt={m.get('plddt')}", flush=True)

    TIMERS_ON[0] = False
    for i in range(a.plain):
        fold_s, m = one_fold()
        record(f"plain{i}", fold_s, m)

    for i in range(a.timed):
        INCL.clear(); CHILD.clear(); STACK.clear()
        TIMERS_ON[0] = True
        fold_s, m = one_fold()
        TIMERS_ON[0] = False
        record(f"timed{i}", fold_s, m)
        for k, v in sorted(INCL.items(), key=lambda kv: -(kv[1]["s"] - CHILD[kv[0]])):
            print(f"    {k:32s} n={v['n']:7d} incl={v['s']:8.3f}s excl={v['s']-CHILD[k]:8.3f}s",
                  flush=True)

    walls = [r["fold_s"] for r in res["runs"] if r["arm"].startswith("plain")]
    if walls:
        res["plain_median"] = st.median(walls)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote " + str(a.out), flush=True)


if __name__ == "__main__":
    main()
