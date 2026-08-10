#!/usr/bin/env python3
"""p3-l1-source-068 — the fold wall at ttnn 0.68.0, with the two new sites in the arm set.

Same instrument as `perf/p3l1/fold_ab.py` (X7's), with three differences that this leg needs:

  * `_SHARED_NORM_L1` is its own arm, so the PWA and template sites can be measured beside X7's
    two levers instead of on top of them.
  * `tt_bio.protenix` holds its OWN reference to `_narrow_proj_linear` (it imports the name), so
    patching only `tt_bio.tenstorrent` left the template z projection invisible to X7's op wall --
    which is why no `w=[256, 64]` row appears in any of X7's `ops_*.json`. Both modules are
    patched here.
  * `PairWeightedAveraging`'s and the template embedder's `layer_norm` are timed, because at both
    sites the norm is shared across several projections and its removed write is paid once per
    region, not once per projection.
"""
import argparse, json, re, statistics as st, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0, "us": []})
STATE = {"dev": None}


def record(key, dt):
    w = WALL[key]
    w["n"] += 1
    w["s"] += dt
    if len(w["us"]) < 2000:
        w["us"].append(round(dt * 1e6, 2))


def timed_call(key, fn, *a, **kw):
    import ttnn
    dev = STATE["dev"]
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(dev)
    record(key, time.perf_counter() - t0)
    return out


def residual_add_lines(path):
    lines = Path(path).read_text().splitlines()
    return {i for i, l in enumerate(lines, 1)
            if re.search(r"^\s+[sz] = ttnn\.add_\([sz], [sz]_update\)", l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--l1-out", type=int, default=1)
    ap.add_argument("--l1-bw", default="16")
    ap.add_argument("--bias-l1-norm", type=int, default=1)
    ap.add_argument("--shared-norm", type=int, default=1)
    ap.add_argument("--pair-bw", default="16")
    ap.add_argument("--instrument", default="ops", choices=["block", "ops", "none"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B

    T._PAIR_PROJ_L1_OUT = bool(a.l1_out)
    T._PAIR_PROJ_L1_BW = None if a.l1_bw == "none" else int(a.l1_bw)
    T._PAIR_BIAS_L1_NORM = bool(a.bias_l1_norm)
    T._PWA_L1_NORM = bool(a.shared_norm in (1, 3))
    T._TEMPLATE_L1_NORM = bool(a.shared_norm in (2, 3))
    T._PAIR_PROJ_BW = None if a.pair_bw == "none" else int(a.pair_bw)
    T._pair_proj_program_config.cache_clear()

    one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / ".msa_p3l1s068", a.target, a.a3m)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"

    times = []
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(round(t, 3))

    res = {"arm": a.arm, "card": "qb2 chip 2", "ttnn": "0.68.0",
           "l1_out": bool(a.l1_out), "l1_bw": T._PAIR_PROJ_L1_BW,
           "bias_l1_norm": bool(a.bias_l1_norm), "pwa_l1_norm": T._PWA_L1_NORM, "template_l1_norm": T._TEMPLATE_L1_NORM,
           "pair_bw": T._PAIR_PROJ_BW, "narrow_bw": T._NARROW_PROJ_BW,
           "instrument": a.instrument, "cold_s": round(cold_s, 3), "fold_s": times,
           "median_fold_s": st.median(times) if times else None,
           "n_tokens": cold_m.get("n_tokens"), "plddt": cold_m.get("plddt"),
           "grid": list(T.COMPUTE_GRID_MAIN)}

    if a.instrument != "none":
        STATE["dev"] = T.get_device()
        saved = []
        for cls, tag in ((T.Pairformer, "stage"), (T.PairformerLayer, "block")):
            f = cls.__call__
            saved.append((cls, "__call__", f))
            cls.__call__ = (lambda g, nm: lambda self, *x, **k:
                            timed_call(nm, g, self, *x, **k))(f, f"{tag}:{cls.__name__}")
        for cls in (T.TriangleMultiplication, T.TriangleAttention,
                    T.AttentionPairBias, T.PairWeightedAveraging):
            f = cls.__call__
            saved.append((cls, "__call__", f))
            cls.__call__ = (lambda g, nm: lambda self, *x, **k:
                            timed_call(f"body:{nm}", g, self, *x, **k))(f, cls.__name__)

        if a.instrument == "ops":
            # both namespaces: protenix.py imported the NAME, so patching T alone misses the
            # template z projection entirely
            for ns in (T, P):
                for nm in ("_narrow_proj_linear", "_pair_proj_linear"):
                    f = getattr(ns, nm, None)
                    if f is None:
                        continue
                    saved.append((ns, nm, f))
                    setattr(ns, nm, (lambda g, n: lambda x, w, *r, **k: timed_call(
                        f"{n}|in0={list(x.padded_shape)}|w={list(w.shape)}",
                        g, x, w, *r, **k))(f, nm))
            # the shared norms: one per region, several projections each
            f_ln = T._l1_layer_norm
            saved.append((T, "_l1_layer_norm", f_ln))

            def _ln_timed(x, headroom, **kw):
                return timed_call(f"shared_layer_norm|in={list(x.padded_shape)}",
                                  f_ln, x, headroom, **kw)
            T._l1_layer_norm = _ln_timed
            saved.append((P, "_l1_layer_norm", P._l1_layer_norm))
            P._l1_layer_norm = _ln_timed

            want = residual_add_lines(T.__file__)
            f_add = ttnn.add_

            def add_(*x, **k):
                fr = sys._getframe(1)
                if fr.f_lineno in want and fr.f_code.co_filename.endswith("tenstorrent.py"):
                    return timed_call("residual_add_", f_add, *x, **k)
                return f_add(*x, **k)
            saved.append((ttnn, "add_", f_add))
            ttnn.add_ = add_
            res["residual_add_lines"] = sorted(want)

        t_inst, m_inst = one_fold()
        for ns, nm, f in saved:
            setattr(ns, nm, f)
        res["instrumented_fold_s"] = round(t_inst, 3)
        res["instrumented_plddt"] = m_inst.get("plddt")
        res["wall"] = {k: {"calls": v["n"], "wall_ms": round(v["s"] * 1e3, 3),
                           "median_us": round(st.median(v["us"]), 2) if v["us"] else None}
                       for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
        res["l1_out_refused"] = [str(k) for k in T._L1_OUT_REFUSED]

    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "wall"}, indent=1), flush=True)
    for k, v in res.get("wall", {}).items():
        print(f"  {k[:80]:80s} {v['calls']:>6} calls {v['wall_ms']:>10.3f} ms  {v['median_us']} us",
              flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
