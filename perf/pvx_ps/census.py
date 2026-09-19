#!/usr/bin/env python3
"""Whole-fold block census for one model, as a call TREE that sums.

Why a tree and not a flat timer list. `perf/other512/fold_ab_multi.py` times
`Pairformer` and `PairformerLayer` under one key each, which is right for an A/B on a
pairformer lever and wrong for a census: protenix runs a Pairformer in the trunk AND a
second one inside the confidence head, and a flat key adds them together. Every region
here is keyed by its CALL PATH, so the same class under two parents is two rows, and a
parent's self time is `inclusive - sum(direct children)`. That is what makes the census
sum: the top level adds up to `model.fold`, and `fold_s - fold` is the host residual,
named rather than dropped.

The syncs ARE the instrument (`tenstorrent.StageWall` says the same thing): without them
the first `to_torch` collects the whole track and every stage before it reads as free.
They also cost -- `synced-bracket-inflates-op-level-fixed-cost` puts a `sync;call;sync`
bracket at ~0.05 ms of host floor -- so this harness ALWAYS runs uninstrumented folds
first and reports the census overhead as the difference, instead of quoting a census
fold as the model's wall.

Clock: `perf/clocksample.py`, sampled DURING every fold, per the campaign standard.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))

STACK: list[str] = []
NODES: dict[str, dict] = {}
DEV = {"d": None}
ON = {"v": False}


def _sync():
    import ttnn
    if DEV["d"] is None:
        import tt_bio.tenstorrent as T
        DEV["d"] = T.get_device()
    ttnn.synchronize_device(DEV["d"])


def timed(key, fn, *a, **kw):
    if not ON["v"]:
        return fn(*a, **kw)
    _sync()
    t0 = time.perf_counter()
    STACK.append(key)
    try:
        return fn(*a, **kw)
    finally:
        STACK.pop()
        _sync()
        dt = time.perf_counter() - t0
        path = "/".join(STACK + [key])
        nd = NODES.setdefault(path, {"n": 0, "incl": 0.0, "child": 0.0})
        nd["n"] += 1
        nd["incl"] += dt
        if STACK:
            NODES.setdefault("/".join(STACK),
                             {"n": 0, "incl": 0.0, "child": 0.0})["child"] += dt


def wrap_method(cls, meth, key, installed):
    f = getattr(cls, meth, None)
    if f is None:
        return
    def w(self, *a, **k):
        return timed(key, f, self, *a, **k)
    setattr(cls, meth, w)
    installed.append(f"{cls.__name__}.{meth}->{key}")


def wrap_func(mod, name, key, installed):
    f = getattr(mod, name, None)
    if f is None:
        return
    def w(*a, **k):
        return timed(key, f, *a, **k)
    setattr(mod, name, w)
    installed.append(f"{mod.__name__}.{name}->{key}")


def install(model, deep, installed):
    """Stage timers for the model's own pipeline, then the shared body classes."""
    import tt_bio.tenstorrent as T
    if model in ("protenix-v2", "protenix-v1", "opendde", "opendde-abag"):
        import tt_bio.protenix as P
        wrap_method(P.Protenix, "fold", "fold", installed)
        wrap_method(P.Protenix, "_trunk_cond", "trunk_cond", installed)
        wrap_method(P.Protenix, "_atom_feat_inputs", "atom_feats", installed)
        wrap_method(P.AtomAttentionEncoder, "__call__", "input_aae", installed)
        wrap_method(P.AtomFeaturization, "c_l", "diff_atom_cache", installed)
        wrap_method(P.AtomFeaturization, "p_lm", "diff_atom_cache", installed)
        wrap_method(P.Trunk, "__call__", "trunk", installed)
        wrap_method(P.Trunk, "_msa", "msa", installed)
        wrap_method(P.Trunk, "_template", "template", installed)
        wrap_method(P.Trunk, "_noisy_structure", "noisy_struct", installed)
        wrap_method(P.Trunk, "_noisy_structure_dist", "noisy_struct", installed)
        wrap_method(P.Protenix, "_diffusion_pair_cond", "diff_pair_cond", installed)
        wrap_method(P.Protenix, "_plm_z_term", "plm_z_term", installed)
        wrap_func(P, "edm_sample", "sampler", installed)
        wrap_method(P.DiffusionModule, "denoise", "denoise", installed)
        wrap_method(P.DiffusionModule, "denoise_traced", "denoise", installed)
        wrap_method(P.DiffusionModule, "_denoise_multiplicity", "denoise", installed)
        wrap_method(P.ConfidenceHead, "confidence", "confidence", installed)
        wrap_method(P.ConfidenceHead, "confidence_device", "confidence", installed)
        wrap_method(P.ConfidenceHead, "z_base_device", "conf_zbase", installed)
        wrap_method(P.ConfidenceHead, "_postprocess", "conf_postprocess", installed)
    # shared device modules -- the same classes Boltz-2 executes, so a row here is
    # comparable across models by construction
    for nm, key in (("Pairformer", "pairformer"), ("MSA", "msa_stack"),
                    ("Diffusion", "diffusion_mod"), ("DiffusionTransformer", "dit")):
        cls = getattr(T, nm, None)
        if cls is not None:
            wrap_method(cls, "__call__", key, installed)
    if deep:
        for nm, key in (("PairformerLayer", "pf_layer"), ("MSALayer", "msa_layer"),
                        ("TriangleMultiplication", "trimul"),
                        ("TriangleAttention", "triatt"),
                        ("AttentionPairBias", "attn_pair_bias"),
                        ("PairWeightedAveraging", "pwa"),
                        ("OuterProductMean", "opm"), ("Transition", "transition"),
                        ("DiffusionTransformerLayer", "dit_layer")):
            cls = getattr(T, nm, None)
            if cls is not None:
                wrap_method(cls, "__call__", key, installed)


def tree():
    out = []
    for path, nd in sorted(NODES.items()):
        out.append({"path": path, "n": nd["n"], "incl_s": round(nd["incl"], 4),
                    "self_s": round(nd["incl"] - nd["child"], 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--clean", type=int, default=2, help="uninstrumented folds (wall + A/A)")
    ap.add_argument("--stage", type=int, default=1, help="stage-level census folds")
    ap.add_argument("--deep", type=int, default=1, help="body-level census folds")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn                                                    # noqa: F401
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        from fold_ab_multi import patch_boltz2_cfg
        patch_boltz2_cfg()

    installed: list[str] = []
    install(a.model, deep=True, installed=installed)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "timers": installed, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxps_{a.size}", tgt, a3m)
    DEV["d"] = T.get_device()
    g = DEV["d"].compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]

    plan = ([("cold", False, False)] + [("clean", False, False)] * a.clean
            + [("stage", True, False)] * a.stage + [("deep", True, True)] * a.deep)
    for i, (tag, on, _deep) in enumerate(plan):
        ON["v"] = on
        NODES.clear(); STACK.clear()
        with clocksample.during(period=2.0) as clk:
            t0 = time.perf_counter()
            fold_s, m = one_fold()
            wall = time.perf_counter() - t0
        rec = {"ix": i, "tag": tag, "instrumented": on, "fold_s": round(fold_s, 4),
               "wall_s": round(wall, 4), "n_tokens": m.get("n_tokens"),
               "plddt": m.get("plddt"), "clock": clk.summary(), "clock_line": clk.line(0)}
        if on:
            rec["tree"] = tree()
        res["runs"].append(rec)
        print(f"[{tag}] fold {fold_s:.3f}s n_tokens={m.get('n_tokens')} "
              f"plddt={m.get('plddt')} | {clk.line(0)}", flush=True)
        if on:
            for r in rec["tree"]:
                print(f"    {r['path']:<62s} n={r['n']:<6d} incl={r['incl_s']:8.3f} "
                      f"self={r['self_s']:8.3f}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
