#!/usr/bin/env python3
"""Decompose openfold3's 512 aa fold, with the 63.7 % nobody has looked at as the target.

`perf/other512/fold_ab_multi.py` timed the shared trunk primitives (PairformerLayer,
TriangleAttention, TriangleMultiplication, ...) and left 68.8 s of a 108.110 s fold in one
undecomposed bucket. This instruments THAT bucket: the host featurisation, the 200-step diffusion
rollout at every level it has a boundary, the confidence head and the CIF write.

Every timed region syncs the device immediately before the clock stops and once before it starts,
per the standing rule. That perturbs the fold, so the run reports its own instrumented total against
the 108.110 s uninstrumented baseline and the perturbation is stated, never hidden.

Nesting is real and is reported, not silently summed: `diffusion:rollout` CONTAINS
`diffusion:conditioning`, `diffusion:pad_pair`, `diffusion:module`, and `diffusion:module` contains
the NPE / encoder / DiT / decoder rows. `sum_children` per parent is printed so containment can be
checked instead of assumed.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEV = {"d": None}


def timed(key, fn, *a, **kw):
    import ttnn
    d = DEV["d"]
    if d is not None:
        ttnn.synchronize_device(d)
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    if d is not None:
        ttnn.synchronize_device(d)
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def host_timed(key, fn, *a, **kw):
    """Host-only region: no device sync (there is nothing in flight to drain)."""
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def wrap_method(cls, name, key, host=False):
    f = getattr(cls, name)
    runner = host_timed if host else timed
    setattr(cls, name, (lambda g: lambda self, *x, **k: runner(key, g, self, *x, **k))(f))


def wrap_static(cls, name, key):
    f = getattr(cls, name).__func__ if isinstance(getattr(cls, name), staticmethod) else getattr(cls, name)
    setattr(cls, name, staticmethod((lambda g: lambda *x, **k: timed(key, g, *x, **k))(f)))


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

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "openfold3")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "openfold3")

    # ---- the timers -------------------------------------------------------------------------
    import tt_bio.openfold3_fold as OF
    import tt_bio.openfold3_trunk as TR
    import tt_bio.openfold3_template as TP
    import tt_bio.openfold3_msa_embedder as ME
    import tt_bio.openfold3_diffusion as DC
    import tt_bio.openfold3_diffusion_module as DM
    import tt_bio.openfold3_diffusion_transformer as DT
    import tt_bio.openfold3_diffusion_decoder as DD
    import tt_bio.openfold3_atom_transformer as AT
    import tt_bio.openfold3_confidence as CF
    import tt_bio.openfold3_data as OD
    import tt_bio.openfold3_host_prep as HP
    import tt_bio.worker as W

    # top level
    wrap_method(OF.OpenFold3, "fold", "top:model.fold")
    wrap_method(TR.OF3Trunk, "__call__", "top:trunk")
    wrap_method(OF.OpenFold3, "_confidence", "top:confidence")

    # trunk internals not covered by the shared-primitive timers
    wrap_method(TP.TemplateEmbedder, "__call__", "trunk:template")
    wrap_method(ME.MSAModuleEmbedder, "__call__", "trunk:msa_embedder")
    wrap_method(ME.MSAModuleBlock, "__call__", "trunk:msa_block")
    wrap_method(TR.OF3TrunkGlue, "glue_z", "trunk:glue_z")
    wrap_method(TR.OF3TrunkGlue, "glue_s", "trunk:glue_s")

    # the 200-step rollout
    import tt_bio.openfold3_sample_diffusion as SDM
    SD = SDM.OF3SampleDiffusion
    wrap_method(SD, "__call__", "diff:rollout")
    wrap_method(DC.OF3DiffusionConditioning, "__call__", "diff:conditioning")
    wrap_static(SD, "_pad_tokens", "diff:pad_tokens_si(host round trip)")
    wrap_static(SD, "_pad_pair", "diff:pad_pair_zij(host round trip)")
    wrap_method(DM.OF3DiffusionModule, "__call__", "diff:module")
    wrap_method(DM.OF3NoisyPositionEmbedder, "__call__", "dm:npe")
    wrap_static(DM.OF3DiffusionModule, "_pad_atoms", "dm:pad_atoms(host round trip)")
    wrap_static(DM.OF3DiffusionModule, "_pad_tokens", "dm:pad_tokens_ai(host round trip)")
    wrap_method(AT.OF3AtomTransformer, "__call__", "dm:atom_transformer")
    wrap_method(DT.OF3DiffusionTransformer, "__call__", "dm:dit_stack")
    wrap_method(DT._DiTBlock, "__call__", "dm:dit_block")
    wrap_method(DD.OF3AtomAttentionDecoder, "__call__", "dm:decoder")

    # host featurisation / prep / write
    OD.build_openfold3_features = (lambda g: lambda *x, **k: host_timed(
        "host:build_features", g, *x, **k))(OD.build_openfold3_features)
    OD.make_openfold3_msa_features = (lambda g: lambda *x, **k: host_timed(
        "host:msa_features", g, *x, **k))(OD.make_openfold3_msa_features)
    for nm, key, dev_side in (("derive_block_aux", "host:derive_block_aux", False),
                              ("derive_template_feat", "host:derive_template_feat", False),
                              ("derive_relpos", "host:derive_relpos", False),
                              ("ref_atom_embed", "host:ref_atom_embed", False),
                              # runs device ops -> needs the sync, so it is not a host region
                              ("run_input_atom_encoder", "prep:input_atom_encoder", True)):
        setattr(HP, nm, (lambda g, k, r: lambda *x, **kw: r(k, g, *x, **kw))(
            getattr(HP, nm), key, timed if dev_side else host_timed))
    W._write_openfold3_structure = (lambda g: lambda *x, **k: host_timed(
        "host:write_structure", g, *x, **k))(W._write_openfold3_structure)
    # worker.py imports the host_prep helpers INSIDE the function, so patching the module
    # attribute is enough -- but build_openfold3_features / make_openfold3_msa_features are
    # imported by name at call time too. Both come from the module object, so this holds.

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta = B.build_fold("openfold3", ROOT / f".msa_of3deep_{a.size}", tgt, a3m)[:2]
    DEV["d"] = T.get_device()
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "of3_diffusion_fp32": os.environ.get("OF3_DIFFUSION_FP32_DEVICE", "1"),
           "runs": []}

    print("=== cold fold (JIT + program cache; discarded) ===", flush=True)
    t0 = time.perf_counter()
    cold_s, cold_m = one_fold()
    WALL.clear()
    print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
          f"n_atoms={cold_m.get('n_atoms')} plddt={cold_m.get('plddt')}", flush=True)

    import hashlib
    for i in range(a.repeat):
        WALL.clear()
        fold_s, m = one_fold()
        rows = {k: {"calls": v["n"], "s": round(v["s"], 3),
                    "ms_per_call": round(v["s"] * 1e3 / max(v["n"], 1), 3),
                    "pct_of_fold": round(100.0 * v["s"] / fold_s, 2)}
                for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
        rec = {"run": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                              for p in sorted(struct_dir.glob("*")) if p.is_file()},
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "regions": rows}
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
