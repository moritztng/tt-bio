#!/usr/bin/env python3
"""One arm of a cross-model A/B: fold a 512 aa page fixture out of ONE source tree.

`ab.py` alternates a module global inside one process, which is the cheapest honest A/B
for a lever behind a runtime gate. The levers this file exists to check are not behind a
gate on the arms that matter: the shared `tenstorrent.AdaLN` was split into a conditioning
half (`s_terms`) and the rest, and only OpenFold3 passes the split result back in. Every
other consumer takes the same ops in a different order, and there is no flag to flip. So
the arms are two SOURCE TREES, and this script folds out of exactly one of them.

    --tree PATH   the tree to import tt_bio from; asserted, not assumed
    --census      run ONE cold fold with reachability counters installed and report which
                  of the changed code paths the model actually reaches. Not timed: the
                  counters are Python wrappers on a function called ~1e5 times per fold,
                  which is minutes of host time on Protenix and would poison a wall.
    (default)     cold fold discarded, then --repeat warm folds, uninstrumented, which is
                  the fold wall a page cell is made of.

Digests are comparable BETWEEN ARMS OF ONE HOST AND CARD and nowhere else. Two runs of
main on qb2, cards 1 and 3, fold this fixture to 9eee996d71653d1b and da9b4ed68f8c0405
(perf/trimul_f1/census_openfold3_512_qb2c1.json against the page cell). Never compare a
digest from this file to one recorded on another card.
"""
import argparse, hashlib, importlib, json, os, shutil, statistics, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="boltz2 | esmfold2 | protenix-v2 | opendde | openfold3")
    ap.add_argument("--tree", type=Path, required=True, help="source tree to import tt_bio from")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--census", action="store_true")
    # `build_fold` takes `fast`, but this script never passed it, so esmfold2 could only ever be
    # folded here in normal precision. On Wormhole that is not a choice: the ESMC-6B LM is ~12.8 GB
    # and a chip has ~12 GB, and the `--fast` forcing that handles it lives in `main.py`'s CLI path
    # (main.py:2395), which `build_fold` does not go through. Default False, so every existing
    # result from this file was produced by the same code path it is recorded against.
    ap.add_argument("--fast", action="store_true",
                    help="fold in --fast mode. Required for esmfold2 on Wormhole; a --fast arm is "
                         "only comparable to another --fast arm.")
    # Stage ablation: vary the two loop counts and the fold-level walls give the split with no
    # instrument between the clock and the work. Both are module globals on tt_baseline, set from
    # the _resolve_* pair below; override AFTER those, never inside the resolver, so an unset flag
    # reproduces the shipped value exactly.
    ap.add_argument("--recycles", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    # Diffusion multiplicity. build_fold's `samples` reaches n_sample; the service offers up to 5.
    ap.add_argument("--samples", type=int, default=None)
    # Per-fold host / device / transfer split from tt_baseline's whole-function brackets (no device
    # sync added), recorded per fold beside the wall it splits. The cold fold's row is first.
    ap.add_argument("--instrument", action="store_true")
    # Replay the captured diffusion-step trace (the CLI's --trace for protenix-v2 / opendde,
    # --diffusion_trace for boltz2).
    # Unset folds at the shipped default (tt_bio.worker._trace_default: on for protenix-v1/v2
    # and boltz2 on Blackhole); --no-trace forces the untraced fold.
    ap.add_argument("--trace", action="store_const", const=True, default=None)
    ap.add_argument("--no-trace", dest="trace", action="store_const", const=False)
    # Paired trace A/B in ONE process: the region is reserved once, then each warm pair folds
    # traced and untraced in alternating order. --repeat counts pairs. Host load on a shared
    # box drifts by more than the trace saving, so only a paired delta can read it.
    ap.add_argument("--ab-trace", action="store_true")
    # The same pairing for any module-level bool, e.g. tt_bio.runtime:SERIAL_HOST.
    ap.add_argument("--ab-attr", metavar="MODULE:NAME[=ON/OFF][,...]", default=None)
    # The same pairing for one precision lever: NAME joins / leaves the `_levers` set of the loaded model
    # (and of any attribute of it that carries one), which every model reads per fold.
    ap.add_argument("--ab-lever", metavar="NAME[,NAME]", default=None)
    # Stage split inside the fold: each MODULE:Class.method is wrapped in a device sync on both
    # sides and its inclusive seconds summed per fold (nested stages each count their own time).
    # The syncs cost overlap, so a --phases fold is slower than a plain one: read it as a split,
    # never as a wall.
    ap.add_argument("--phases", default=None, metavar="MODULE:[Class.]name,...")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_sampling_steps, _resolve_recycling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    assert Path(T.__file__).resolve().is_relative_to(tree), \
        f"tt_bio came from {T.__file__}, not {tree}"
    assert Path(B.__file__).resolve().is_relative_to(tree), \
        f"tt_baseline came from {B.__file__}, not {tree}"
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.steps is not None:
        B.SAMPLING_STEPS = a.steps

    # `build_fold`'s cfg carries no Boltz-2 hyperparameters, so `_WorkerState.load_model`
    # raises KeyError('conf_kwargs'). `perf/other512/fold_ab_multi.py` already carries the
    # injector, holding exactly what `tt_bio.main` builds; import it rather than restate the
    # hyperparameters, so both arms cannot drift apart. It must run AFTER the two lines above,
    # because it reads them. Its module body inserts THIS tree's root at sys.path[0], which
    # would shadow `--tree` for the main arm, so the path is snapshotted and restored.
    if a.model == "boltz2":
        snap = list(sys.path)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "other512"))
        from fold_ab_multi import patch_boltz2_cfg
        sys.path[:] = snap
        B.RECYCLING_STEPS = _resolve_recycling_steps(a.recycles, a.model)   # the injector reads it
        patch_boltz2_cfg(diffusion_trace=True if a.ab_trace else a.trace)   # None: shipped default

    # Reachability counters. Installed only for --census: they wrap a function the diffusion
    # rollout calls tens of thousands of times, so they are host cost inside the timed region.
    reach = {"adaln_calls": 0, "adaln_with_s_terms": 0, "adaln_s_terms_calls": 0}
    if a.census:
        AdaLN = T.AdaLN
        _call, _terms = AdaLN.__call__, getattr(AdaLN, "s_terms", None)

        def counted_call(self, *args, **kw):
            reach["adaln_calls"] += 1
            reach["adaln_with_s_terms"] += kw.get("s_terms") is not None
            return _call(self, *args, **kw)
        AdaLN.__call__ = counted_call
        if _terms is not None:                      # absent on the pre-split tree
            def counted_terms(self, *args, **kw):
                reach["adaln_s_terms_calls"] += 1
                return _terms(self, *args, **kw)
            AdaLN.s_terms = counted_terms

    phase_s = {}
    if a.phases:
        import ttnn

        def _sync():
            d = T.get_device()
            ttnn.synchronize_device(d)

        def _timed(key, fn):
            def w(*args, **kw):
                _sync()
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kw)
                finally:
                    _sync()
                    phase_s[key] = phase_s.get(key, 0.0) + time.perf_counter() - t0
            return w
        for spec in a.phases.split(","):
            mod, path = spec.split(":")
            owner = importlib.import_module(mod)
            *cls, meth = path.split(".")        # Class.method, or a module-level function
            for c in cls:
                owner = getattr(owner, c)
            setattr(owner, meth, _timed(path, getattr(owner, meth)))

    fixdir = tree / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    msa_dir = tree / f".msa_xmodel_{a.model}_{a.size}"
    # The recycle count is build_fold's argument, not a tt_baseline global: setting B.RECYCLING_STEPS
    # was a no-op, so --recycles folded at the default and the record named the request.
    _bf = {} if a.samples is None else {"samples": a.samples}
    _bf["recycling_steps"] = a.recycles
    one_fold, meta, state = B.build_fold(a.model, msa_dir, tgt, a3m, fast=a.fast,
                                          instrument=a.instrument, trace=True if a.ab_trace else a.trace,
                                          **_bf)[:3]
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"label": a.label, "model": a.model, "tree": str(tree), "size": a.size,
           "census": a.census, "fast": a.fast, "trace": a.trace, "ab_trace": a.ab_trace, "ab_attr": a.ab_attr, "ab_lever": a.ab_lever, "ttnn": im.version("ttnn"),
           "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [int(T.COMPUTE_GRID_MAIN[0]), int(T.COMPUTE_GRID_MAIN[1])],
           "recycling_steps": meta["recycling_steps"], "sampling_steps": B.SAMPLING_STEPS,
           "samples": a.samples or B.DIFFUSION_SAMPLES,
           "flags": {k: getattr(T, k, "absent") for k in
                     ("ADALN_S_HOIST", "FP32_SOFTMAX_BIAS_HOIST", "TRIMUL_TAIL_F1",
                      "TRANSPOSE_L1_HEADROOM")},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "loadavg": open("/proc/loadavg").read().split()[:3], "folds": []}

    def one(tag):
        phase_s.clear()
        fold_s, m = one_fold()
        for p in sorted(struct_dir.glob("*.cif")):  # the structure itself, for an RMSD between arms
            shutil.copyfile(p, a.out.parent / f"{tag}_{p.name}")
        plddt = m.get("plddt")
        if plddt is None:
            plddt = m.get("complex_plddt")
        return {"tag": tag, "fold_s": round(fold_s, 3), "plddt": plddt,
                "conf": {k: m[k] for k in ("confidence_score", "ptm", "iptm", "complex_plddt",
                                           "complex_iplddt") if k in m},
                "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
                "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                               for p in sorted(struct_dir.glob("*")) if p.is_file()},
                "loadavg": open("/proc/loadavg").read().split()[:3],
                **({"phases": {k: round(v, 3) for k, v in phase_s.items()}} if a.phases else {})}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    cold = one("cold")
    res["cold"] = cold
    # What the chip carries after the load, so a default-path run shows whether it can replay.
    res["trace_region_bytes"] = T.trace_region_size()
    print(f"[{a.label}] cold {cold['fold_s']:.3f}s plddt={cold['plddt']} "
          f"cif={list(cold['cif_sha256'].values())}", flush=True)
    flush()

    if (a.ab_trace or a.ab_attr or a.ab_lever) and not a.census:
        if a.ab_trace:
            on, off, key = "trace", "plain", "ab_trace_s"

            def arm_set(v):
                meta["job_cfg"]["trace"] = v
                if a.model == "boltz2":     # Boltz-2 reads its DiT trace flag off the model
                    state.model.structure_module._diffusion_trace = v
        elif a.ab_lever:
            on, off, key = "on", "off", "ab_lever_s"
            # One name or a comma list, A/B'd as one stack. An unknown name fails here, not as a silent null A/B.
            ab = T.parse_levers(a.ab_lever)
            holders = [o for o in [state.model, *vars(state.model).values()]
                       if isinstance(getattr(o, "_levers", None), frozenset)]
            if holders:
                base = {id(o): o._levers - ab for o in holders}

                def arm_set(v):
                    for o in holders:
                        o._levers = base[id(o)] | ab if v else base[id(o)]
            else:
                # ESMFold2 enters its lever set per fold (esmfold2_runtime.fold_scope), from TT_BIO_LEVERS when
                # set, else the mode's set (always the fast one on Wormhole, where ESMFold2 is fast-only).
                assert a.model in ("esmfold2", "esmfold2-fast"), "the loaded model carries no _levers set"
                from tt_bio import esmfold2_runtime as ER
                base_set = T.parse_levers(os.environ.get("TT_BIO_LEVERS") or (
                    ER.fold_levers(a.fast or ER.is_wormhole()))) - ab

                def arm_set(v):
                    os.environ["TT_BIO_LEVERS"] = ",".join(sorted(base_set | ab if v else base_set)) or "none"
        else:
            # MODULE:NAME toggles a bool; MODULE:NAME=ON/OFF sets two values instead, integers where they
            # parse (tt_bio.trimul_tail:T1_ROWS=16/32), else strings (tt_bio.tenstorrent:_TRIMUL_EINSUM_FID=hifi2/):
            # the "on" arm takes ON.
            attrs = []
            for x in a.ab_attr.split(","):
                mod, name = x.split(":")
                name, _, vals = name.partition("=")
                attrs.append((mod, name, tuple(int(t) if t.lstrip("-").isdigit() else t for t in vals.split("/")) if vals else (True, False)))
            on, off, key = "on", "off", "ab_attr_s"

            def arm_set(v):
                for mod, name, (von, voff) in attrs:
                    setattr(importlib.import_module(mod), name, von if v else voff)
        arm = {True: [], False: []}
        for i in range(a.repeat):
            for v in ((True, False) if i % 2 == 0 else (False, True)):
                arm_set(v)
                rec = one(f"warm{i}-{on if v else off}")
                res["folds"].append(rec)
                arm[v].append(rec["fold_s"])
                print(f"[{a.label}] {rec['tag']} {rec['fold_s']:.3f}s "
                      f"cif={list(rec['cif_sha256'].values())}", flush=True)
                flush()
        d = [round(x - y, 3) for x, y in zip(arm[True], arm[False])]
        res[key] = {on: arm[True], off: arm[False], "delta": d,
                    "median_delta": round(statistics.median(d), 3),
                    **({"attr": a.ab_attr} if a.ab_attr else {}),
                    **({"lever": a.ab_lever} if a.ab_lever else {})}
        res["median_s"] = round(statistics.median(arm[True]), 3)
        res["spread_s"] = round(max(arm[True]) - min(arm[True]), 3)
        print(f"[{a.label}] {on} minus {off} {d} s, median {res[key]['median_delta']}",
              flush=True)
    elif not a.census:
        for i in range(a.repeat):
            rec = one(f"warm{i}")
            res["folds"].append(rec)
            print(f"[{a.label}] warm {i} {rec['fold_s']:.3f}s plddt={rec['plddt']} "
                  f"cif={list(rec['cif_sha256'].values())}", flush=True)
            flush()
        w = sorted(f["fold_s"] for f in res["folds"])
        res["median_s"] = round(statistics.median(w), 3)
        res["spread_s"] = round(w[-1] - w[0], 3)
        print(f"[{a.label}] median {res['median_s']:.3f}s spread {res['spread_s']:.3f}s",
              flush=True)

    if a.instrument:
        res["phase_times"] = meta["phase_times"]
    res["reach"] = dict(reach)
    # Which fused kernels actually served a call, and why the declines. Every one of them is
    # gated on a constant fitted on a 130-core Blackhole grid, so "ships default-ON" says nothing
    # about whether it fired on an 8x9 Wormhole. These counters live inside the kernels, so
    # reading them is free -- unlike the --census wrappers above they are always on.

    def _kern(name):
        try:
            mod = importlib.import_module("tt_bio." + name)
        except Exception as e:                                  # a tree without the kernel
            return {"absent": repr(e)}
        d = {}
        for attr, leg in (("STATS", "fwd"), ("STATS_BACK", "back"), ("STATS_GATED", "gated")):
            st = getattr(mod, attr, None)
            if st is not None:
                d[leg] = {"served": int(st[0]), "declined": int(st[1])}
        rej = getattr(mod, "REJECTS", None)
        if rej:
            d["rejects"] = {str(k): int(v) for k, v in rej.items()}
        pm = getattr(mod, "_PM_OVER_L1", None)
        if pm is not None:
            d["pm_over_l1"] = sorted(str(x) for x in pm)
        return d

    res["kernels"] = {n: _kern(n) for n in
                      ("triatt_sdpa", "trimul_tail", "reblock_permute", "rfd3_bias", "triatt_ln")}
    # Which trimul in-projection route fed the einsum (gin-ragged, gated-inproj-moved, gated-inproj,
    # rowblock, ...), and the trimul_tail sub-counters STATS above lumps together.
    res["trimul_routes"] = {"|".join(k): int(v) for k, v in getattr(T, "TRIMUL_MM_TRANSPOSE_STATS", {}).items()}
    # g_out riding the in-projection is what lets T2 serve the tail without T1: its served/declined count and why.
    res["trimul_gout"] = {"stats": list(getattr(T, "TRIMUL_GOUT_STATS", [])),
                          "rejects": dict(getattr(T, "TRIMUL_GOUT_REJECTS", {}))}
    try:
        TT_ = importlib.import_module("tt_bio.trimul_tail")
        res["trimul_tail_sub"] = {k: list(getattr(TT_, k)) for k in
                                  ("GIN_MOVE_STATS", "RES_STATS", "T1_STATS", "T2_STATS", "T2CM_STATS", "MASK_STATS")
                                  if hasattr(TT_, k)}
    except Exception as e:
        res["trimul_tail_sub"] = {"absent": repr(e)}
    # The grid-derived thresholds in effect in this process. _apply_grid_thresholds runs at
    # device open, not at import, so this has to be read after a fold, not from the source.
    res["thresholds"] = {k: getattr(T, k, "absent") for k in (
        "_IS_SMALL_GRID", "SEQ_LEN_MORE_CHUNKING", "TRANSITION_BATCH_CHUNKING_THRESHOLD",
        "TRANSITION_W_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST",
        "TRANSITION_W_CHUNK_SIZE", "TRIANGLE_MULT_L1_MAX_SEQ_FAST", "TRIANGLE_MULT_L1_MAX_SEQ",
        "SMALL_GRID_SEQ_TILE", "SMALL_GRID_PAIR_TILE_AREA", "SDPA_CHUNK_MAX",
        "PAIRFORMER_PAD_MULTIPLE", "TRIANGLE_MULT_CHUNK_SIZE")}
    res["fp32_softmax_stats"] = dict(T.FP32_SOFTMAX_STATS)
    res["fp32_softmax_l1_row_caps"] = {str(k): v for k, v in T._FP32_SOFTMAX_L1_ROW_CAP.items()}
    res["transpose_l1_refused_keys"] = sorted(
        str(k) for k in getattr(T, "_TRANSPOSE_L1_REFUSED", ()))
    flush()
    print(f"[{a.label}] reach {res['reach']}", flush=True)
    print(f"[{a.label}] FP32_SOFTMAX_STATS {res['fp32_softmax_stats']}", flush=True)
    print(f"[{a.label}] TRIMUL_ROUTES {res['trimul_routes']} {res['trimul_tail_sub']}", flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
