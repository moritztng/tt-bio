#!/usr/bin/env python3
"""p3-l1-output — the wall. One real 298 aa protenix-v2 fold per arm, block wall instrumented.

X2 priced the L1-output candidate at 298.7 ms/fold as a per-call delta times 3144 calls/fold. That
is a projection, and this script is what turns it into a wall figure or kills it.

Two instruments, chosen with --instrument:

  block  `PairformerLayer.__call__` synchronised on both sides and summed over the fold's 524
         c_z=256 executions. Nothing inside the block is serialised, so overlap survives and the
         number is the production one. This is the headline.
  ops    additionally `_pair_proj_linear`, `_narrow_proj_linear` and the layer's residual `add_`,
         each synchronised on both sides. Attribution, not a headline: syncing around an op
         removes the overlap it had with its neighbours, so the parts do not sum to the block.

The fold wall is recorded every run but is NOT the deliverable: base spread on this harness is
144 ms and X2's fold wall moved +68 ms against a real 31.5 ms win.
"""
import argparse, json, re, statistics as st, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0, "us": []})
STATE = {"dev": None, "on": False}


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
    """The Pairformer layer's residual adds, found in the source so a line edit cannot stale them."""
    lines = Path(path).read_text().splitlines()
    return {i for i, l in enumerate(lines, 1)
            if re.search(r"^\s+[sz] = ttnn\.add_\([sz], [sz]_update\)", l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label for this arm")
    ap.add_argument("--l1-out", type=int, default=1, help="_PAIR_PROJ_L1_OUT")
    ap.add_argument("--l1-bw", default="1", help="_PAIR_PROJ_L1_BW ('none' or an int)")
    ap.add_argument("--bias-l1-norm", type=int, default=1, help="_PAIR_BIAS_L1_NORM")
    ap.add_argument("--pair-bw", default="16", help="_PAIR_PROJ_BW ('none' or an int)")
    ap.add_argument("--instrument", default="block", choices=["block", "ops", "none"])
    ap.add_argument("--repeat", type=int, default=2, help="timed folds after the cold one")
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    T._PAIR_PROJ_L1_OUT = bool(a.l1_out)
    T._PAIR_PROJ_L1_BW = None if a.l1_bw == "none" else int(a.l1_bw)
    T._PAIR_BIAS_L1_NORM = bool(a.bias_l1_norm)
    T._PAIR_PROJ_BW = None if a.pair_bw == "none" else int(a.pair_bw)
    T._pair_proj_program_config.cache_clear()

    one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / ".msa_p3l1", a.target, a.a3m)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"

    times = []
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(round(t, 3))

    res = {"arm": a.arm, "l1_out": bool(a.l1_out), "l1_bw": T._PAIR_PROJ_L1_BW,
           "bias_l1_norm": bool(a.bias_l1_norm), "pair_bw": T._PAIR_PROJ_BW,
           "narrow_bw": T._NARROW_PROJ_BW, "instrument": a.instrument,
           "cold_s": round(cold_s, 3), "fold_s": times,
           "median_fold_s": st.median(times) if times else None,
           "min_fold_s": min(times) if times else None,
           "n_tokens": cold_m.get("n_tokens"), "plddt": cold_m.get("plddt"),
           "grid": list(T.COMPUTE_GRID_MAIN)}

    if a.instrument != "none":
        STATE["dev"] = T.get_device()
        saved = []

        for cls in (T.Pairformer,):
            f = cls.__call__
            saved.append((cls, "__call__", f))
            cls.__call__ = (lambda g, nm: lambda self, *x, **k:
                            timed_call(f"stage:{nm}", g, self, *x, **k))(f, cls.__name__)
        for cls in (T.PairformerLayer,):
            f = cls.__call__
            saved.append((cls, "__call__", f))
            cls.__call__ = (lambda g, nm: lambda self, *x, **k:
                            timed_call(f"block:{nm}", g, self, *x, **k))(f, cls.__name__)
        for cls in (T.TriangleMultiplication, T.TriangleAttention,
                    T.AttentionPairBias, T.PairWeightedAveraging):
            f = cls.__call__
            saved.append((cls, "__call__", f))
            cls.__call__ = (lambda g, nm: lambda self, *x, **k:
                            timed_call(f"body:{nm}", g, self, *x, **k))(f, cls.__name__)

        if a.instrument == "ops":
            for nm in ("_pair_proj_linear", "_narrow_proj_linear"):
                f = getattr(T, nm)
                saved.append((T, nm, f))
                setattr(T, nm, (lambda g, n: lambda x, w, *r, **k: timed_call(
                    f"{n}|in0={list(x.padded_shape)}|w={list(w.shape)}", g, x, w, *r, **k))(f, nm))
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
                           "median_us": round(st.median(v["us"]), 2) if v["us"] else None,
                           "us_head": v["us"][:6]}
                       for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
        res["l1_out_refused"] = [str(k) for k in T._L1_OUT_REFUSED]

    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "wall"}, indent=1), flush=True)
    for k, v in res.get("wall", {}).items():
        print(f"  {k[:88]:88s} {v['calls']:>6} calls {v['wall_ms']:>10.3f} ms  "
              f"{v['median_us']} us", flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
