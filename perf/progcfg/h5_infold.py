#!/usr/bin/env python3
"""Q-A — the in-fold census and the narrow/wide walls that settle the 13x.

The disagreement this exists to settle:

  isolated OFF-minus-ON region delta at [1,512,512,64]  =  0.172 ms/region  (h5_cells.py)
  in-fold body:TriangleMultiplication OFF-minus-ON      = +367.4 ms / 160   =  2.30 ms/region
                                                                                       (size512-ab)

and the fact that frames it: the isolated OFF arm's WHOLE region costs 0.7461 ms, so a saving of
2.30 ms/region is 3.08x the entire cost of the thing supposed to be doing the saving. Either the
wall contains ops outside the region, or the in-fold region is >3x its isolated self, or the
denominator is wrong. Predictions HA1-HA4 are in perf/progcfg/PREDICTIONS_INFOLD.md, committed
before this file opened a device.

Three nested walls, all synchronised on both sides (ttnn-sync-before-every-timed-region: an unsynced
`to_torch` drain has inverted a ranking in this codebase):

  narrow = p_out + g_out + the region multiply_        <- the unit h5_cells.py calls "region"
  body   = TriangleMultiplication.__call__             <- the unit size512-ab's +367.4 belongs to
  wide   = body + the residual add_ that consumes it   <- narrow + the cascade

THE DIFFERENCE BETWEEN THEM IS THE ANSWER. Reported side by side, per track (c=64 template and
c_z=256 pair are never averaged), in ms/region and in ms/fold.

The census is COUNTED, not assumed. Every `_pair_proj_linear` call is tallied by
(padded shape, weight shape, dtype) and by THE BRANCH ACTUALLY TAKEN, read back from the returned
tensor's own `memory_config().buffer_type` rather than inferred from the shape or from whether the
program config was non-None. This org has had a denominator slip already (2.30 ms/region quoted
against 160 trimul executions when _trimul_out_proj runs twice per trimul).

Three sync depths, chosen per arm, because the instrument is part of the measurement:

  body  syncs only at the body edges and the residual add_ -- as close as this harness gets to
        size512-ab's instrument, so the +367.4 can be reproduced or not before anything is
        concluded from it.
  wall  adds a sync around each projection and around the region multiply_. This is the arm that
        answers the question. It costs ~4 extra drains per trimul, identical in both arms.
  ops   adds a sync around every op inside the trimul body. A deliberately distorted instrument:
        it destroys host/device pipelining inside the trimul, so its ABSOLUTES are not comparable
        to the other depths and only its SHARES of the OFF-minus-ON delta should be quoted.

Usage (qb2 chip 0, card 0 lease):

  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-h5-infold \\
  TT_MESH_GRAPH_DESC_PATH=$TTNN/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \\
  python3 perf/progcfg/h5_infold.py --size 512 \\
      --arms on:wall,off:wall,on:wall,off:wall,on:ops,off:ops \\
      --out perf/progcfg/h5_infold_512_qb2c0.json

qb2 runs ttnn 0.68.0: every absolute here is a RATIO owing a qb1/0.67.4 re-take (charter 4.8).
"""
import argparse, hashlib, json, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
CENSUS = Counter()
STATE = {"dev": None, "depth": "wall", "in_trimul": 0, "n_proj": 0,
         "expect_region_mul": False, "pending_out": None, "trimul_c": None}


def sync():
    import ttnn
    ttnn.synchronize_device(STATE["dev"])


def timed(key, fn, *a, **kw):
    """Synchronised on BOTH sides. The only timing primitive in this file."""
    if STATE["dev"] is None:
        return fn(*a, **kw)
    sync()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    sync()
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def chan(t):
    """The channel width that names the track: 64 = template, 256 = pair."""
    try:
        return int(t.shape[-1])
    except Exception:                                                          # noqa: BLE001
        return -1


def sha_dir(d):
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def install(T, ttnn):
    """Patch the census point, the three walls, and (at depth=ops) every op inside the trimul."""
    ORIG = {
        "ppl": T._pair_proj_linear,
        "top": T._trimul_out_proj,
        "mul_": ttnn.multiply_,
        "add_": ttnn.add_,
        "tm": T.TriangleMultiplication.__call__,
        "pfl": T.PairformerLayer.__call__,
        "pf": T.Pairformer.__call__,
        "ppc": T._pair_proj_config,
    }

    # ---- the census: branch actually taken, read off the returned tensor -------------------------
    def ppl(x, w, ckc, dtype, l1_out=False):
        c = chan(x)
        track = "trimul" if STATE["in_trimul"] else "other"
        key = f"proj|c{c}|{track}"
        out = timed(key, ORIG["ppl"], x, w, ckc, dtype, l1_out=l1_out)
        try:
            buf = "L1" if out.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
        except Exception:                                                      # noqa: BLE001
            buf = "?"
        CENSUS[(str(tuple(int(d) for d in x.padded_shape)),
                str(tuple(int(d) for d in w.shape)), str(dtype),
                "l1_out" if l1_out else "plain", track, buf)] += 1
        return out

    # ---- the region: two projections then the sigmoid multiply_ ---------------------------------
    def top(x, weight, ckc):
        out = ORIG["top"](x, weight, ckc)
        STATE["n_proj"] += 1
        if STATE["n_proj"] == 2:                 # g_out has just landed; the region mul is next
            STATE["expect_region_mul"] = True
        return out

    def mul_(a, b, *rest, **kw):
        if STATE["expect_region_mul"]:
            STATE["expect_region_mul"] = False
            return timed(f"regionmul|c{chan(a)}", ORIG["mul_"], a, b, *rest, **kw)
        if STATE["depth"] == "ops" and STATE["in_trimul"]:
            return timed(f"op:multiply_|c{STATE['trimul_c']}", ORIG["mul_"], a, b, *rest, **kw)
        return ORIG["mul_"](a, b, *rest, **kw)

    # ---- the cascade: the residual add_ that consumes the trimul's output ------------------------
    # Caller-agnostic on purpose: the template track's residual is not in PairformerLayer, and a
    # guard keyed on the caller would have missed it.
    def add_(a, b, *rest, **kw):
        c = STATE["pending_out"]
        if c is not None:
            STATE["pending_out"] = None
            return timed(f"residadd|c{c}", ORIG["add_"], a, b, *rest, **kw)
        return ORIG["add_"](a, b, *rest, **kw)

    def tm(self, x, *a, **kw):
        c = chan(x)
        STATE["in_trimul"] += 1
        STATE["n_proj"] = 0
        STATE["trimul_c"] = c
        try:
            out = timed(f"body:TriangleMultiplication|c{c}", ORIG["tm"], self, x, *a, **kw)
        finally:
            STATE["in_trimul"] -= 1
            STATE["expect_region_mul"] = False
        STATE["pending_out"] = c
        return out

    T._pair_proj_linear = ppl
    T._trimul_out_proj = top
    ttnn.multiply_ = mul_
    ttnn.add_ = add_
    T.TriangleMultiplication.__call__ = tm
    T.PairformerLayer.__call__ = lambda self, *x, **k: timed(
        "block:PairformerLayer", ORIG["pfl"], self, *x, **k)
    T.Pairformer.__call__ = lambda self, *x, **k: timed(
        "stage:Pairformer", ORIG["pf"], self, *x, **k)

    # ---- depth=ops: every op inside the trimul, so "the excess is elsewhere" gets an address ------
    def wrap_op(name, orig):
        def f(*a, **kw):
            if STATE["depth"] == "ops" and STATE["in_trimul"]:
                return timed(f"op:{name}|c{STATE['trimul_c']}", orig, *a, **kw)
            return orig(*a, **kw)
        return f

    for name in ("layer_norm", "matmul", "permute", "transpose", "concat", "clone",
                 "reallocate", "chunk", "typecast"):
        o = getattr(ttnn, name)
        ORIG[name] = o
        setattr(ttnn, name, wrap_op(name, o))
    ORIG["mm"] = ttnn.experimental.minimal_matmul
    ttnn.experimental.minimal_matmul = wrap_op("minimal_matmul", ORIG["mm"])
    ORIG["cm"] = T._channel_move
    T._channel_move = wrap_op("channel_move", ORIG["cm"])
    return ORIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="on:wall,off:wall,on:wall,off:wall",
                    help="comma list of arm:depth, run in this order. depth in {body,wall,ops}")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import importlib.metadata as im

    def set_arm(name):
        on = name == "on"
        T._PAIR_PROJ_L1_OUT = on
        T._pair_proj_program_config.cache_clear()
        T._L1_OUT_REFUSED.clear()

    ORIG = install(T, ttnn)

    size = a.size
    set_arm("on")
    one_fold, meta, _s = B.build_fold(
        "protenix-v2", ROOT / f".msa_infold_{size}",
        a.fixdir / f"cdk2x2_{size}.yaml", a.fixdir / f"cdk2x2_{size}.a3m")
    STATE["dev"] = T.get_device()
    struct_dir = Path(meta["struct_dir"])

    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"), "size": size,
           "grid": list(T.COMPUTE_GRID_MAIN),
           "l1_bank_bytes": T._l1_bank_bytes(),
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
           "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
           "trimul_mm_out": getattr(T, "_TRIMUL_MM_OUT", None),
           "note": "qb2 at ttnn 0.68.0 -- every absolute is a RATIO owing a qb1/0.67.4 re-take",
           "runs": []}

    # `_PAIR_PROJ_L1_OUT` is the ONLY flag moved. C2FIX, _PAIR_BIAS_L1_NORM, _PWA_L1_NORM and
    # _TEMPLATE_L1_NORM stay at their production values in both arms: size512-ab moved five flags
    # together, which is why its body decomposition could not attribute anything on its own.
    print("=== cold fold ===", flush=True)
    STATE["depth"] = "wall"
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.1f}s tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}",
          flush=True)

    for spec in a.arms.split(","):
        arm, _, depth = spec.partition(":")
        depth = depth or "wall"
        set_arm(arm)
        STATE["depth"] = depth
        WALL.clear()
        CENSUS.clear()
        t0 = time.perf_counter()
        try:
            fold_s, m = one_fold()
        except Exception as e:                                                 # noqa: BLE001
            res["runs"].append({"arm": arm, "depth": depth, "error": f"{type(e).__name__}: {e}"[:400]})
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm}:{depth} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        rec = {"arm": arm, "depth": depth, "fold_s": round(fold_s, 3),
               "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
               "cif_sha256": sha_dir(struct_dir),
               "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED],
               "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                           for k, v in sorted(WALL.items())},
               "census": [{"padded_shape": k[0], "weight": k[1], "dtype": k[2], "l1_out": k[3],
                           "track": k[4], "branch_taken": k[5], "calls": n}
                          for k, n in sorted(CENSUS.items())]}
        # narrow / body / wide, per track, from the counted denominators
        derived = {}
        for c in (64, 256):
            proj = WALL.get(f"proj|c{c}|trimul", {"n": 0, "s": 0.0})
            rmul = WALL.get(f"regionmul|c{c}", {"n": 0, "s": 0.0})
            body = WALL.get(f"body:TriangleMultiplication|c{c}", {"n": 0, "s": 0.0})
            radd = WALL.get(f"residadd|c{c}", {"n": 0, "s": 0.0})
            regions = rmul["n"] or (proj["n"] // 2)
            if not regions:
                continue
            narrow_ms = (proj["s"] + rmul["s"]) * 1e3
            body_ms, wide_ms = body["s"] * 1e3, (body["s"] + radd["s"]) * 1e3
            derived[f"c{c}"] = {
                "regions_counted": regions, "projections_counted": proj["n"],
                "trimul_calls": body["n"], "resid_adds": radd["n"],
                "narrow_ms_total": round(narrow_ms, 2),
                "body_ms_total": round(body_ms, 2),
                "wide_ms_total": round(wide_ms, 2),
                "narrow_ms_per_region": round(narrow_ms / regions, 4),
                "body_ms_per_region": round(body_ms / regions, 4),
                "wide_ms_per_region": round(wide_ms / regions, 4),
                "body_minus_narrow_ms_per_region": round((body_ms - narrow_ms) / regions, 4),
            }
        rec["derived"] = derived
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}:{depth} fold {fold_s:.1f}s plddt {m.get('plddt')} "
              f"({time.perf_counter()-t0:.0f}s wall)", flush=True)
        for k, v in sorted(derived.items()):
            print(f"    {k}: narrow {v['narrow_ms_per_region']}  body {v['body_ms_per_region']}  "
                  f"wide {v['wide_ms_per_region']}  ms/region over {v['regions_counted']}", flush=True)
        for e in rec["census"]:
            print(f"    CENSUS {e['padded_shape']:>22s} @ {e['weight']:>12s} {e['l1_out']:>7s} "
                  f"{e['track']:>7s} -> {e['branch_taken']:>4s}  x{e['calls']}", flush=True)

    # ---- OFF minus ON, per depth, per track -------------------------------------------------------
    deltas = {}
    for depth in {r.get("depth") for r in res["runs"] if "derived" in r}:
        rs = [r for r in res["runs"] if r.get("depth") == depth and "derived" in r]
        for c in ("c64", "c256"):
            on = [r["derived"][c] for r in rs if r["arm"] == "on" and c in r["derived"]]
            off = [r["derived"][c] for r in rs if r["arm"] == "off" and c in r["derived"]]
            if not (on and off):
                continue
            e = {"n_on": len(on), "n_off": len(off),
                 "regions": on[0]["regions_counted"]}
            for w in ("narrow", "body", "wide", "body_minus_narrow"):
                f = f"{w}_ms_per_region"
                if f not in on[0]:
                    continue
                e[f"{w}_off_minus_on_ms_per_region"] = round(
                    st.median([o[f] for o in off]) - st.median([o[f] for o in on]), 4)
            if len(on) > 1:
                e["aa_floor_narrow_ms_per_region"] = round(
                    abs(on[0]["narrow_ms_per_region"] - on[1]["narrow_ms_per_region"]), 4)
                e["aa_floor_body_ms_per_region"] = round(
                    abs(on[0]["body_ms_per_region"] - on[1]["body_ms_per_region"]), 4)
            deltas[f"{depth}|{c}"] = e
    # per-op shares, depth=ops only
    for depth in ("ops",):
        rs = [r for r in res["runs"] if r.get("depth") == depth]
        on = [r for r in rs if r["arm"] == "on"]
        off = [r for r in rs if r["arm"] == "off"]
        if on and off:
            keys = set(on[0]["wall_ms"]) | set(off[0]["wall_ms"])
            deltas["ops_off_minus_on_ms"] = {
                k: round(off[0]["wall_ms"].get(k, {}).get("ms", 0.0)
                         - on[0]["wall_ms"].get(k, {}).get("ms", 0.0), 2)
                for k in sorted(keys)}
    res["deltas"] = deltas
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(deltas, indent=1), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
