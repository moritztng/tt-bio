#!/usr/bin/env python3
"""Where a 512 aa x 1 sample OpenFold3 / OpenBind-0 fold spends its time, on the page protocol.

    python perf/spd_gap_of3/census.py --model openfold3 --tree T --out OUT.json [--fast] [--repeat 3]
        [--steps N] [--recycles N]

Same fold as perf/of3_4xpd/xmodel_ab.py (page fixture cdk2x2_512, seed 0, 1 sample, predict_one
boundary, cold fold discarded), with two layers of brackets on the warm folds:
* tt_baseline's PHASES instrument: host featurisation vs model.fold vs CIF write, outermost only.
* model sections, device-synchronised at both ends: input glue, trunk, sampler, confidence.
* inside the sampler, unsynchronised: the host time spent issuing each diffusion-module call
  (`dm_issue`) and each conditioning `single` call. A step's wall minus its issue time is the
  time the host waited for the device; issue ~= wall means the rollout is host-bound.
--steps / --recycles are ablations for the census only (they change the model's work, so a
wall taken with them is never a page cell).
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--plain", action="store_true", help="no brackets: the uninstrumented wall")
    ap.add_argument("--deep", action="store_true",
                    help="also device-synchronised brackets on every trunk / diffusion sub-module; run it with "
                         "OF3_DIFFUSION_TRACE=0 (a sync cannot sit inside a trace capture)")
    ap.add_argument("--sites", action="store_true",
                    help="bin every device upload and every read by its tt_bio call site: count, MB, the wait for "
                         "the device queue before it (synchronised) and the call itself")
    ap.add_argument("--prof-feat", action="store_true",
                    help="cProfile build_openfold3_features on the warm folds; the top 40 functions by cumulative "
                         "time land in OUT beside the result (prof_feat.txt)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    assert Path(T.__file__).resolve().is_relative_to(tree)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.steps is not None:
        B.SAMPLING_STEPS = a.steps
    if a.recycles is not None:
        B.RECYCLING_STEPS = a.recycles

    import ttnn
    from tt_bio import openfold3 as M0, openfold3_fold as MF, openfold3_trunk as MT
    from tt_bio import openfold3_sample_diffusion as MS, openfold3_diffusion_module as MD
    from tt_bio import openfold3_diffusion as MC

    acc = {}
    on = {"v": False}

    def sync():
        d = getattr(T, "_device", None)
        if d is not None:
            ttnn.synchronize_device(d)

    def wrap(owner, attr, name, synced, when=None):
        fn = getattr(owner, attr)

        def inner(*x, **k):
            if not on["v"] or (when and not when(x, k)):
                return fn(*x, **k)
            if synced:
                sync()
            t0 = time.perf_counter()
            try:
                return fn(*x, **k)
            finally:
                if synced:
                    sync()
                r = acc.setdefault(name, [0.0, 0])
                r[0] += time.perf_counter() - t0
                r[1] += 1
        setattr(owner, attr, staticmethod(inner) if isinstance(owner.__dict__.get(attr), staticmethod)
                else inner)

    if not a.plain:
        wrap(MF.OpenFold3, "fold", "model.fold", True)
        wrap(M0.InputEmbedderGlue, "__call__", "input_glue", True)
        wrap(MT.OF3Trunk, "__call__", "trunk", True)
        wrap(MS.OF3SampleDiffusion, "__call__", "sampler", True)
        wrap(MF.OpenFold3, "_confidence", "confidence", True)
        wrap(MD.OF3DiffusionModule, "__call__", "dm_issue", False)
        wrap(MC.OF3DiffusionConditioning, "single", "cond_single_issue", False)
        wrap(ttnn, "to_torch", "to_torch(all)", False)
        # The rollout's per-step host work between a read and the next replay, while the device idles.
        wrap(MS.OF3SampleDiffusion, "_augment", "augment", False)
        wrap(ttnn, "from_torch", "from_torch(all)", False)
        # No device = a host-side tilize: the rollout's per-step n_emb and rl staging.
        wrap(ttnn, "from_torch", "from_torch(host)", False, when=lambda x, k: k.get("device") is None)
        wrap(ttnn, "copy_host_to_device_tensor", "h2d_copy", False)
    if a.sites:
        import traceback

        # One-line upload helpers (`_ft`, `ft = lambda ...`) would bin every caller on their own line.
        helpers = {"_ft", "ft", "<lambda>", "_to_dev", "to_dev", "up", "host", "bf", "_w_tt", "torch_to_tt"}

        def site():
            for f in reversed(traceback.extract_stack(limit=14)[:-2]):
                if "/tt_bio/" in f.filename and f.name not in helpers:
                    return f"{Path(f.filename).name}:{f.lineno}"
            return "?"

        def by_site(attr, kind, when):
            fn = getattr(ttnn, attr)

            def inner(*x, **k):
                if not on["v"] or not when(x, k):
                    return fn(*x, **k)
                t0 = time.perf_counter()
                sync()
                t1 = time.perf_counter()
                out = fn(*x, **k)
                t2 = time.perf_counter()
                t = x[0] if attr == "from_torch" else out
                r = acc.setdefault(f"{kind} {site()}", [0.0, 0, 0.0, 0.0])
                r[0] += t2 - t1
                r[1] += 1
                r[2] += t.numel() * t.element_size() / 1e6 if hasattr(t, "element_size") else 0.0
                r[3] += t1 - t0
                return out
            setattr(ttnn, attr, inner)

        by_site("from_torch", "up", lambda x, k: k.get("device") is not None)
        by_site("to_torch", "read", lambda x, k: True)
    if a.deep:
        from tt_bio import openfold3_template as MTP, openfold3_msa_embedder as MME
        from tt_bio import openfold3_diffusion_transformer as MDT, openfold3_atom_transformer as MAT
        from tt_bio import openfold3_diffusion_decoder as MDD
        for owner in (T.TriangleMultiplication, T.TriangleAttention, T.AttentionPairBias, T.Transition,
                      T.OuterProductMean, MME.MSAModuleBlock, MTP.TemplateEmbedder, MDT._DiTBlock,
                      MAT.OF3AtomTransformer, MDD.OF3AtomAttentionDecoder, MD.OF3NoisyPositionEmbedder,
                      MD.AtomPairUpdate, MC.OF3DiffusionConditioning):
            wrap(owner, "__call__", owner.__name__, True)

    if a.prof_feat:
        import cProfile, io, pstats
        from tt_bio import openfold3_data as MDATA, worker as MW
        prof = cProfile.Profile()
        feat = MDATA.build_openfold3_features

        def feat_prof(*x, **k):
            if not on["v"]:
                return feat(*x, **k)
            prof.enable()
            try:
                return feat(*x, **k)
            finally:
                prof.disable()
        MDATA.build_openfold3_features = feat_prof
        if getattr(MW, "build_openfold3_features", None) is feat:
            MW.build_openfold3_features = feat_prof

    fixdir = tree / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / "cdk2x2_512.yaml", fixdir / "cdk2x2_512.a3m"
    one_fold, meta = B.build_fold(a.model, tree / f".msa_xmodel_{a.model}_512", tgt, a3m,
                                  fast=a.fast, instrument=not a.plain)[:2]
    res = {"model": a.model, "tree": str(tree), "fast": a.fast, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "steps": B.SAMPLING_STEPS,
           "recycles": B.RECYCLING_STEPS, "plain": a.plain, "folds": []}

    def digest():
        # sha1 of every ATOM/HETATM line the fold wrote: the cheapest bit-exactness signal.
        import hashlib
        h = hashlib.sha1()
        for f in sorted(Path(meta["struct_dir"]).glob("**/*.cif")):
            h.update(b"".join(l for l in f.read_bytes().splitlines(True)
                              if l.startswith((b"ATOM", b"HETATM"))))
        return h.hexdigest()[:12]

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    s, m = one_fold()
    res["cold_s"] = round(s, 3)
    flush()
    on["v"] = True
    for i in range(a.repeat):
        acc.clear()
        s, m = one_fold()
        row = {"fold_s": round(s, 3), "plddt": m.get("plddt", m.get("complex_plddt")),
               "digest": digest(), "sections": {k: [round(e, 4) for e in v] for k, v in acc.items()}}
        pt = meta.get("phase_times")
        if pt:
            row["phases"] = pt[-1]
        res["folds"].append(row)
        print(json.dumps(row), flush=True)
        flush()
    # The last fold's structure beside the result, so two arms can be compared in Angstrom.
    import shutil
    for f in sorted(Path(meta["struct_dir"]).glob("**/*.cif")):
        shutil.copy(f, a.out.parent / f.name)
    if a.prof_feat:
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(25)
        (a.out.parent / "prof_feat.txt").write_text(buf.getvalue())
    w = sorted(f["fold_s"] for f in res["folds"])
    res["median_s"] = round(statistics.median(w), 3)
    flush()


if __name__ == "__main__":
    main()
