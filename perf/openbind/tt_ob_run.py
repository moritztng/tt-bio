#!/usr/bin/env python3
"""Time one OpenBind-0 (or OpenFold3-preview2) cell on Tenstorrent, at the same input
and the same protocol as `perf/openbind/gpu_reference.json`.

The GPU reference's device boundary is `predict_step`, cuda-synchronised on both sides,
with featurisation in the dataloader and the CIF write in a callback. This runner puts
its device clock on `OpenFold3.fold` with `ttnn.synchronize_device` on both sides, and
reports host featurisation and the structure write separately, so `device_s` here and
`h200_device_s` there measure the same region.

Protocol, matched to the reference: 3 recycles, 200 sampling steps, seed 42, single
sequence (no MSA search, no templates), 1 cold fold discarded + N warm folds in one
process, median of the warm folds reported.

`--decomp` additionally wraps the trunk / rollout / diffusion-module regions, the same
set `perf/of3deep/decomp.py` uses. That perturbs the fold (every region syncs the
device), so a decomposed run reports its own instrumented total and must never be
quoted as a fold time.

    python3 perf/openbind/tt_ob_run.py --model openbind \
        --input perf/openbind/inputs/ob_apo_512.tt.yaml --repeat 3 \
        --out perf/openbind/tt_results/ob_apo_512_s1.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, statistics, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WALL: dict = defaultdict(lambda: {"n": 0, "s": 0.0})
MISSING: list = []
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
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def wrap_method(cls, name, key, host=False):
    """Skip silently when the attribute is gone: the trunk is refactored often and a
    decomposition that dies on a renamed helper is worth less than one missing a row."""
    if not hasattr(cls, name):
        MISSING.append(f"{cls.__name__}.{name}")
        return
    f = getattr(cls, name)
    runner = host_timed if host else timed
    setattr(cls, name, (lambda g: lambda self, *x, **k: runner(key, g, self, *x, **k))(f))


def wrap_static(cls, name, key):
    if not hasattr(cls, name):
        MISSING.append(f"{cls.__name__}.{name}")
        return
    a = getattr(cls, name)
    f = a.__func__ if isinstance(a, staticmethod) else a
    setattr(cls, name, staticmethod((lambda g: lambda *x, **k: timed(key, g, *x, **k))(f)))


def install_core_timers():
    """The three regions every run needs: host featurisation, the device fold, the write."""
    import tt_bio.openfold3_fold as OF
    import tt_bio.openfold3_data as OD
    import tt_bio.worker as W

    wrap_method(OF.OpenFold3, "fold", "device:model.fold")
    OD.build_openfold3_features = (lambda g: lambda *x, **k: host_timed(
        "host:build_features", g, *x, **k))(OD.build_openfold3_features)
    W._write_atom_array_structure = (lambda g: lambda *x, **k: host_timed(
        "host:write_structure", g, *x, **k))(W._write_atom_array_structure)


def install_decomp_timers():
    """The region set of perf/of3deep/decomp.py. Nesting is real and reported, not summed:
    device:model.fold CONTAINS trunk/rollout/confidence, diff:rollout CONTAINS
    diff:conditioning + diff:module, diff:module CONTAINS the dm:* rows."""
    import tt_bio.openfold3_fold as OF
    import tt_bio.openfold3_trunk as TR
    import tt_bio.openfold3_template as TP
    import tt_bio.openfold3_msa_embedder as ME
    import tt_bio.openfold3_diffusion as DC
    import tt_bio.openfold3_diffusion_module as DM
    import tt_bio.openfold3_diffusion_transformer as DT
    import tt_bio.openfold3_diffusion_decoder as DD
    import tt_bio.openfold3_atom_transformer as AT
    import tt_bio.openfold3_sample_diffusion as SDM
    import tt_bio.openfold3_host_prep as HP

    wrap_method(TR.OF3Trunk, "__call__", "top:trunk")
    wrap_method(OF.OpenFold3, "_confidence", "top:confidence")
    wrap_method(TP.TemplateEmbedder, "__call__", "trunk:template")
    wrap_method(ME.MSAModuleEmbedder, "__call__", "trunk:msa_embedder")
    wrap_method(ME.MSAModuleBlock, "__call__", "trunk:msa_block")
    wrap_method(TR.OF3TrunkGlue, "glue_z", "trunk:glue_z")
    wrap_method(TR.OF3TrunkGlue, "glue_s", "trunk:glue_s")
    if hasattr(TR, "OF3PairformerStack"):
        wrap_method(TR.OF3PairformerStack, "__call__", "trunk:pairformer_stack")

    SD = SDM.OF3SampleDiffusion
    wrap_method(SD, "__call__", "diff:rollout")
    wrap_method(DC.OF3DiffusionConditioning, "__call__", "diff:conditioning")
    for _m, _k in (("pair", "diff:cond_pair"), ("single", "diff:cond_single")):
        if hasattr(DC.OF3DiffusionConditioning, _m):
            wrap_method(DC.OF3DiffusionConditioning, _m, _k)
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
    for nm, key, dev_side in (("derive_block_aux", "host:derive_block_aux", False),
                              ("derive_template_feat", "host:derive_template_feat", False),
                              ("derive_relpos", "host:derive_relpos", False),
                              ("ref_atom_embed", "host:ref_atom_embed", False),
                              ("run_input_atom_encoder", "prep:input_atom_encoder", True)):
        if not hasattr(HP, nm):
            MISSING.append(f"host_prep.{nm}")
            continue
        setattr(HP, nm, (lambda g, k, r: lambda *x, **kw: r(k, g, *x, **kw))(
            getattr(HP, nm), key, timed if dev_side else host_timed))
    install_pair_stack_timers()


# Which pair stack is on the stack right now. Every OF3 pair stack is built from the SAME four
# shared primitives, so timing the primitives alone would pool the 48-block trunk with the
# template, MSA and confidence stacks and answer nobody's question. `_STACK` is pushed by the
# stack's own __call__ and read by the primitive's, so each row is attributed to one stack.
_STACK = ["?"]


def stack_scoped(cls, name, label):
    """`label` may be a callable taking the instance, so one class serving two stacks
    (`Pairformer` is both the 48-block trunk and the 4-block confidence stack) reports two rows."""
    if not hasattr(cls, name):
        MISSING.append(f"{cls.__name__}.{name}")
        return
    f = getattr(cls, name)

    def wrapper(self, *x, **k):
        lb = label(self) if callable(label) else label
        _STACK.append(lb)
        try:
            return timed(f"stack:{lb}", f, self, *x, **k)
        finally:
            _STACK.pop()
    setattr(cls, name, wrapper)


def op_scoped(cls, name, op):
    if not hasattr(cls, name):
        MISSING.append(f"{cls.__name__}.{name}")
        return
    f = getattr(cls, name)

    def wrapper(self, *x, **k):
        return timed(f"{_STACK[-1]}:{op}", f, self, *x, **k)
    setattr(cls, name, wrapper)


def install_pair_stack_timers():
    """Per-op attribution INSIDE each pair stack: TriMul, TriAtt, Transition, AttentionPairBias.

    Every row is a synchronised region, so the instrumented total is not a fold time
    (`tt-bio-isolated-op-timing-oversync-inflates-cost`). What it IS good for is the SPLIT
    between the four ops within one run, which is the question pass 1 left open: the trunk's
    ~110 ms/block was a subtraction, not a measurement.
    """
    import tt_bio.tenstorrent as T
    import tt_bio.openfold3_template as TP
    import tt_bio.openfold3_msa_embedder as ME

    stack_scoped(T.Pairformer, "__call__", lambda o: f"pf{len(o.blocks)}")
    stack_scoped(TP.TemplatePairStack, "__call__", "tps")
    stack_scoped(ME.MSAModuleBlock, "__call__", "msab")
    op_scoped(T.PairformerLayer, "__call__", "block")
    op_scoped(T.TriangleMultiplication, "__call__", "trimul")
    op_scoped(T.TriangleAttention, "__call__", "triatt")
    op_scoped(T.Transition, "__call__", "transition")
    op_scoped(T.AttentionPairBias, "__call__", "attn_pair_bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbind", choices=("openbind", "openfold3"))
    ap.add_argument("--input", type=Path, required=True, help="a *.tt.yaml from perf/openbind/inputs")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3, help="warm folds after the discarded cold one")
    ap.add_argument("--decomp", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    from rdkit import Chem
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    import ttnn  # noqa: F401
    import tt_bio.tenstorrent as T
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    _E.set_progress(lambda *x, **k: None)

    recycles = _resolve_recycling_steps(None, a.model)
    steps = _resolve_sampling_steps(None, a.model)
    assert (recycles, steps) == (3, 200), f"protocol drift: {recycles} recycles, {steps} steps"

    work = Path(tempfile.mkdtemp(prefix=f"ttob-{a.model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"
    msa_dir.mkdir()

    cfg = dict(model=a.model, fast=False, output_format="cif",
               recycling_steps=recycles, sampling_steps=steps,
               diffusion_samples=a.samples, seed=42, trace=False,
               msa_dir=str(msa_dir), struct_dir=str(struct_dir),
               single_sequence=True, use_msa_server=False, msa_db_path=None,
               use_envdb=False, msa_endpoint=None, msa_server_url=None,
               msa_pairing_strategy="greedy", msa_server_username=None,
               msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
               write_pae=False, write_pde=False, write_embeddings=False, method=None)
    _ensure_local_artifacts(cfg)

    T.get_device()
    DEV["d"] = T.get_device()
    install_core_timers()
    if a.decomp:
        install_decomp_timers()

    state = _WorkerState("tenstorrent")
    t0 = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t0
    state.bind_run("ttob", cfg)
    state.pfn = lambda *x, **k: None

    import importlib.metadata as im
    inp_sha = hashlib.sha256(a.input.read_bytes()).hexdigest()
    res = {"model": a.model, "label": a.label or a.input.stem, "input": a.input.name,
           "input_sha256": inp_sha, "samples": a.samples,
           "recycles": recycles, "sampling_steps": steps, "seed": 42,
           "single_sequence": True, "decomp": a.decomp,
           "ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": T.arch_name(),
           "load_s": round(load_s, 2), "loadavg_start": open("/proc/loadavg").read().split()[:3],
           "timers_missing": list(MISSING),
           "runs": []}

    def one_fold():
        for p in struct_dir.glob("*"):
            p.unlink()
        WALL.clear()
        t = time.perf_counter()
        metrics, _best, _feats = state.predict_one(a.input, dict(cfg))
        return time.perf_counter() - t, metrics

    print("=== cold fold (JIT + program cache; discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s tokens={cold_m.get('n_tokens')} atoms={cold_m.get('n_atoms')} "
          f"plddt={cold_m.get('plddt')}", flush=True)
    res["cold_s"] = round(cold_s, 3)

    for i in range(a.repeat):
        fold_s, m = one_fold()
        rows = {k: {"calls": v["n"], "s": round(v["s"], 3),
                    "ms_per_call": round(v["s"] * 1e3 / max(v["n"], 1), 3),
                    "pct_of_fold": round(100.0 * v["s"] / fold_s, 2)}
                for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
        rec = {"run": i, "fold_s": round(fold_s, 3),
               "device_s": round(WALL["device:model.fold"]["s"], 3),
               "host_feat_s": round(WALL["host:build_features"]["s"], 3),
               "host_write_s": round(WALL["host:write_structure"]["s"], 3),
               "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
               "n_atoms": m.get("n_atoms"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                              for p in sorted(struct_dir.glob("*")) if p.is_file()},
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "regions": rows}
        res["runs"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\n=== run {i}: fold {fold_s:.3f}s device {rec['device_s']:.3f}s "
              f"feat {rec['host_feat_s']:.3f}s write {rec['host_write_s']:.3f}s "
              f"plddt {m.get('plddt')} tokens {m.get('n_tokens')} atoms {m.get('n_atoms')} ===",
              flush=True)
        for k, v in rows.items():
            print(f"  {k:44s} {v['calls']:6d} calls {v['s']:9.3f} s  "
                  f"{v['ms_per_call']:9.3f} ms/call  {v['pct_of_fold']:6.2f} %", flush=True)

    # Which levers this process actually served. A silently-declined config is
    # indistinguishable from an absent one, so an A/B is only believable if the fold says
    # which arm it ran (`pcc-gate-can-pass-without-the-op-it-names`).
    try:
        import tt_bio.tenstorrent as _T
        res["levers"] = {
            "sdpa_low_div_k": _T._sdpa_low_div_k(),
            "sdpa_wide_k": _T._sdpa_wide_k(),
            "sdpa_k_chunk_stats": list(_T.SDPA_K_CHUNK_STATS),
            "sdpa_chunk_picks": {f"{q}x{k}": v for (q, k), v in _T.SDPA_CHUNK_PICKS.items()},
            "triatt_fused_hifi_stats": dict(_T.TRIATT_FUSED_HIFI_STATS),
            "fp32_softmax_stats": dict(_T.FP32_SOFTMAX_STATS),
            "latch": {k: {"served": v["served"], "refused": v["refused"],
                          "blocked": v["blocked"], "declined": v["declined"],
                          "why": v["why"][:4]}
                      for k, v in _T.LATCH_STATS.items()
                      if v["served"] or v["refused"] or v["blocked"] or v["declined"]},
        }
    except Exception as exc:  # noqa: BLE001
        res["levers"] = {"error": repr(exc)}

    try:
        from tt_bio import reblock_permute as RP
        res["reblock_permute"] = {"gated_enabled": RP._ENABLED_GATED,
                                  "stats_gated": list(RP.STATS_GATED),
                                  "stats": list(getattr(RP, "STATS", []) or []),
                                  "rejects": dict(getattr(RP, "REJECTS", {}) or {})}
    except Exception as exc:  # noqa: BLE001 -- a counter that moved is not a reason to lose the run
        res["reblock_permute"] = {"error": repr(exc)}

    dv = [r["device_s"] for r in res["runs"]]
    fv = [r["fold_s"] for r in res["runs"]]
    res["device_s_median"] = round(statistics.median(dv), 3)
    res["fold_s_median"] = round(statistics.median(fv), 3)
    res["device_spread_pct"] = round(100.0 * (max(dv) - min(dv)) / min(dv), 2) if dv else None
    res["cif_reproducible"] = len({json.dumps(r["cif_sha256"], sort_keys=True)
                                   for r in res["runs"]}) == 1
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nmedian device {res['device_s_median']}s  fold {res['fold_s_median']}s  "
          f"spread {res['device_spread_pct']}%  cif_reproducible {res['cif_reproducible']}", flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
