#!/usr/bin/env python3
"""Decompose OpenDDE's 512 aa fold, targeting the 39.6 % nobody has looked at.

`opendde-512aa-deep-perf` timed the shared trunk primitives and left two buckets undecomposed:
20.88 s of in-block time that is neither TriMul nor TriAtt, and 18.72 s outside the Pairformer
blocks. Together 39.6 % of a 90.874 s fold. This instruments both.

Every device-side timed region syncs immediately before the clock stops and once before it starts.
That perturbs the fold, so the run reports its own instrumented total against the uninstrumented
baseline and the perturbation is stated, never hidden.

Nesting is real and reported, not silently summed: `top:model.fold` contains everything;
`top:trunk` contains `stage:Pairformer` + `trunk:msa` + `trunk:template`; `block:PairformerLayer`
contains the `body:*` rows; `top:rollout` contains `diff:denoise` which contains the `dm:*` rows.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEV = {"d": None}


def _acc(key, dt):
    w = WALL[key]
    w["n"] += 1
    w["s"] += dt


def timed(key, fn, *a, **kw):
    import ttnn
    d = DEV["d"]
    if d is not None:
        ttnn.synchronize_device(d)
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    if d is not None:
        ttnn.synchronize_device(d)
    _acc(key, time.perf_counter() - t0)
    return out


def host_timed(key, fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    _acc(key, time.perf_counter() - t0)
    return out


def wrap_method(cls, name, key, host=False):
    f = getattr(cls, name, None)
    if f is None:
        return
    runner = host_timed if host else timed
    setattr(cls, name, (lambda g: lambda self, *x, **k: runner(key, g, self, *x, **k))(f))


def wrap_keyed(cls, name, keyfn):
    """Wrap a method whose timer key depends on its arguments (Transition: z vs s track)."""
    f = getattr(cls, name, None)
    if f is None:
        return

    def outer(g):
        def inner(self, *x, **k):
            return timed(keyfn(self, *x, **k), g, self, *x, **k)
        return inner
    setattr(cls, name, outer(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "opendde")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "opendde")

    import tt_bio.protenix as PX
    import tt_bio.opendde as OD
    import tt_bio.opendde_data as ODD
    import tt_bio.worker as W

    # ---- top level -----------------------------------------------------------------------
    wrap_method(OD.OpenDDE, "fold", "top:model.fold")
    wrap_method(PX.AtomAttentionEncoder, "__call__", "top:input_aae")
    wrap_method(PX.AtomFeaturization, "c_l", "top:diff_feat.c_l")
    wrap_method(PX.AtomFeaturization, "p_lm", "top:diff_feat.p_lm")
    wrap_method(PX.Trunk, "__call__", "top:trunk")
    wrap_method(PX.Trunk, "_template", "trunk:template")
    wrap_method(PX.Trunk, "_msa", "trunk:msa")
    wrap_method(PX.TrunkInput, "__call__", "trunk:trunk_input")
    wrap_method(OD.OpenDDE, "expand_and_refine", "top:expand_and_refine")
    wrap_method(OD.StructuralTokenExpander, "__call__", "expander:call")
    wrap_method(OD.StructuralTokenExpander, "_pair_project_full", "expander:pair_project_full")
    wrap_method(PX.Protenix, "_diffusion_pair_cond", "top:diffusion_pair_cond")
    wrap_method(PX.Protenix, "_plm_z_term", "top:plm_z_term")
    wrap_method(PX.ConfidenceHead, "confidence", "top:confidence")

    # ---- shared trunk primitives ---------------------------------------------------------
    wrap_method(T.Pairformer, "__call__", "stage:Pairformer")
    wrap_method(T.PairformerLayer, "__call__", "block:PairformerLayer")
    for nm in ("TriangleMultiplication", "TriangleAttention", "AttentionPairBias",
               "PairWeightedAveraging", "OuterProductMean", "MSALayer", "MiniformerLayer"):
        cls = getattr(T, nm, None)
        if cls is not None:
            wrap_method(cls, "__call__", f"body:{nm}")

    # Transition is the single biggest undecomposed in-block candidate. Key it by rank and
    # channel so the pair track (4D, c_z) and the single track (<=3D, c_s) never share a row.
    def _tkey(self, x, *rest):
        try:
            return f"body:Transition[{len(x.shape)}d,c={int(x.shape[-1])}]"
        except Exception:
            return "body:Transition[?]"
    wrap_keyed(T.Transition, "__call__", _tkey)

    # ---- the 200-step rollout ------------------------------------------------------------
    PX.edm_sample = (lambda g: lambda *x, **k: timed("top:rollout", g, *x, **k))(PX.edm_sample)
    wrap_method(PX.DiffusionModule, "denoise", "diff:denoise")
    wrap_method(PX.DiffusionModule, "denoise_traced", "diff:denoise_traced")
    wrap_method(PX.DiffusionModule, "_atom_cond", "dm:atom_cond")
    wrap_method(PX.DiffusionModule, "_denoise_device", "dm:denoise_device")
    wrap_method(PX.DiffusionModule, "_token_dit", "dm:token_dit_host")
    wrap_method(PX.DiffusionModule, "_token_dit_device", "dm:token_dit_device")
    wrap_method(PX.DiffusionModule, "_dit_block_biases", "dm:dit_block_biases")
    wrap_method(PX.DiffusionModule, "_dit_pair_biases", "cond:dit_pair_biases")
    wrap_method(PX.DiffusionModule, "_dit_z_device", "cond:dit_z_device")
    wrap_method(PX.AtomTransformer, "__call__", "dm:atom_transformer")
    wrap_method(T.DiffusionTransformer, "__call__", "dm:dit_stack")
    wrap_method(T.DiffusionTransformerLayer, "__call__", "dm:dit_layer")
    wrap_method(T.AdaLN, "__call__", "dm:adaln")
    wrap_method(T.ConditionedTransitionBlock, "__call__", "dm:cond_transition")

    # ---- host featurisation / write ------------------------------------------------------
    ODD.build_structural_token_features = (lambda g: lambda *x, **k: host_timed(
        "host:build_structural_token_features", g, *x, **k))(ODD.build_structural_token_features)

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta = B.build_fold("opendde", ROOT / f".msa_odde4x_{a.size}", tgt, a3m)[:2]
    DEV["d"] = T.get_device()
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "loadavg": open("/proc/loadavg").read().split()[:3], "runs": []}

    print("=== cold fold (JIT + program cache; discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    WALL.clear()
    print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
          f"n_atoms={cold_m.get('n_atoms')} plddt={cold_m.get('plddt')}", flush=True)

    for i in range(a.repeat):
        WALL.clear()
        fold_s, m = one_fold()
        rows = {k: {"calls": v["n"], "s": round(v["s"], 3),
                    "ms_per_call": round(v["s"] * 1e3 / max(v["n"], 1), 3),
                    "pct_of_fold": round(100.0 * v["s"] / fold_s, 2)}
                for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
        rec = {"run": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(struct_dir.glob("*")) if p.is_file()},
               "loadavg": open("/proc/loadavg").read().split()[:3], "regions": rows}
        res["runs"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\n=== run {i}: instrumented fold {fold_s:.3f}s  plddt {m.get('plddt')} "
              f"n_tokens {m.get('n_tokens')} n_atoms {m.get('n_atoms')} ===", flush=True)
        for k, v in rows.items():
            print(f"  {k:44s} {v['calls']:6d} calls {v['s']:9.3f} s  "
                  f"{v['ms_per_call']:9.3f} ms/call  {v['pct_of_fold']:6.2f} %", flush=True)

    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
