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
import argparse, hashlib, json, os, statistics, sys, time
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
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    assert Path(T.__file__).resolve().is_relative_to(tree), \
        f"tt_bio came from {T.__file__}, not {tree}"
    assert Path(B.__file__).resolve().is_relative_to(tree), \
        f"tt_baseline came from {B.__file__}, not {tree}"
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

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
        patch_boltz2_cfg()

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

    fixdir = tree / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    msa_dir = tree / f".msa_xmodel_{a.model}_{a.size}"
    one_fold, meta = B.build_fold(a.model, msa_dir, tgt, a3m)[:2]
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as im
    res = {"label": a.label, "model": a.model, "tree": str(tree), "size": a.size,
           "census": a.census, "ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [int(T.COMPUTE_GRID_MAIN[0]), int(T.COMPUTE_GRID_MAIN[1])],
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "flags": {k: getattr(T, k, "absent") for k in
                     ("ADALN_S_HOIST", "FP32_SOFTMAX_BIAS_HOIST", "TRIMUL_TAIL_F1",
                      "TRANSPOSE_L1_HEADROOM")},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "loadavg": open("/proc/loadavg").read().split()[:3], "folds": []}

    def one(tag):
        fold_s, m = one_fold()
        return {"tag": tag, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
                "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
                "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                               for p in sorted(struct_dir.glob("*")) if p.is_file()},
                "loadavg": open("/proc/loadavg").read().split()[:3]}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    cold = one("cold")
    res["cold"] = cold
    print(f"[{a.label}] cold {cold['fold_s']:.3f}s plddt={cold['plddt']} "
          f"cif={list(cold['cif_sha256'].values())}", flush=True)
    flush()

    if not a.census:
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

    res["reach"] = dict(reach)
    res["fp32_softmax_stats"] = dict(T.FP32_SOFTMAX_STATS)
    res["fp32_softmax_l1_refused_keys"] = sorted(str(k) for k in T._FP32_SOFTMAX_L1_REFUSED)
    res["transpose_l1_refused_keys"] = sorted(
        str(k) for k in getattr(T, "_TRANSPOSE_L1_REFUSED", ()))
    flush()
    print(f"[{a.label}] reach {res['reach']}", flush=True)
    print(f"[{a.label}] FP32_SOFTMAX_STATS {res['fp32_softmax_stats']}", flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
