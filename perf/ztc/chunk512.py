#!/usr/bin/env python3
"""z-transition-chunk -- QA/QB/QC in a live 512 aa protenix-v2 fold, all arms in one process.

`TRANSITION_H_CHUNK_SIZE` is a module global read at call time (tenstorrent.py:2422), so an arm is
a global flip between folds: no reload, no second device open, no cross-process term. This is what
the isolated fc1 probe that priced the 693 ms/fold could not do -- in-block L1 pressure only exists
inside a block.

Arms are chunk heights. Production at W=512 is 16 (`W <= 384` refuses TRANSITION_H_CHUNK_SIZE_BIG),
so arm `16` IS the production path and repeating it gives this session's own A/A floor.

Scope: the forced height applies only where production would have taken 16 at the pair shape
(4-D, W > 384, c <= 256). The MSA/template Transitions keep production behaviour unless --scope all.

NOTHING IN tt_bio/ IS CHANGED. This is a throwaway harness; Phase 2 forbids production edits.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics as st, sys, time, traceback
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
FC1 = Counter()          # realised fc1 shapes -> calls. Reads the chunk actually taken.
TRSHAPE = Counter()      # every 4-D Transition input shape seen, and whether we forced it
STATE = {"dev": None, "h": 16, "scope": "pair"}


def timed_call(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="16,32,16,24")
    ap.add_argument("--scope", default="pair", choices=["pair", "all"])
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--parity-heights", default="16,32,24,20,28")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    STATE["scope"] = a.scope

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    PROD_H = T.TRANSITION_H_CHUNK_SIZE
    ORIG_TR = T.Transition.__call__

    def forced(x):
        """True when this call site is the one the leg owns: the 512 aa pair Transition."""
        if len(x.shape) != 4:
            return False
        if STATE["scope"] == "all":
            return True
        return int(x.shape[2]) > 384 and int(x.shape[-1]) <= 256

    GRAB = {}

    def tr(self, x):
        f = forced(x)
        TRSHAPE[("x".join(str(int(d)) for d in x.shape), "forced" if f else "prod")] += 1
        if f and "inst" not in GRAB:
            GRAB["inst"] = self
            GRAB["x"] = ttnn.to_torch(x).clone()
        prev = T.TRANSITION_H_CHUNK_SIZE
        if f:
            T.TRANSITION_H_CHUNK_SIZE = STATE["h"]
        try:
            return timed_call("site:Transition", ORIG_TR, self, x)
        finally:
            T.TRANSITION_H_CHUNK_SIZE = prev

    T.Transition.__call__ = tr

    # --- read the branch actually taken: every fc1 [.,.,.,256] x [256,1024] call and its height ---
    _lin = ttnn.linear

    def _count(x, w, *ar, **kw):
        try:
            if len(w.shape) == 2 and int(w.shape[0]) == 256 and int(w.shape[1]) == 1024 \
               and len(x.shape) == 4:
                FC1["x".join(str(int(d)) for d in x.shape)] += 1
        except Exception:
            pass
        return _lin(x, w, *ar, **kw)

    ttnn.linear = _count
    T.ttnn.linear = _count

    saved = []
    for cls, nm in ((T.Pairformer, "stage:Pairformer"), (T.PairformerLayer, "block:PairformerLayer")):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g, k: lambda self, *x, **kw: timed_call(k, g, self, *x, **kw))(f, nm)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": "physical 2",
           "note": "qb2 / ttnn 0.68.0 -- every absolute here is a RATIO input owing a qb1/0.67.4 re-take",
           "prod_chunk_default": PROD_H, "scope": a.scope, "size": a.size, "runs": []}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    t0 = time.perf_counter()
    one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / f".msa_ztc_{a.size}", tgt, a3m)
    STATE["dev"] = T.get_device()
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    struct_dir = Path(meta["struct_dir"])
    print(f"model loaded {time.perf_counter()-t0:.1f}s grid={res['grid']}", flush=True)

    STATE["h"] = PROD_H
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    res["cold"] = {"s": round(cold_s, 3), "n_tokens": cold_m.get("n_tokens"),
                   "plddt": cold_m.get("plddt")}
    print("cold", res["cold"], flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))

    for arm in [int(s) for s in a.arms.split(",")]:
        STATE["h"] = arm
        WALL.clear(); FC1.clear(); TRSHAPE.clear()
        try:
            fold_s, m = one_fold()
            rec = {"h": arm, "fold_s": round(fold_s, 3), "n_tokens": m.get("n_tokens"),
                   "plddt": m.get("plddt"), "cif_sha256": sha_dir(struct_dir),
                   "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                               for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
                   "fc1_shapes": dict(FC1),
                   "transition_shapes": {f"{k[0]}|{k[1]}": v for k, v in TRSHAPE.items()}}
            rec["transition_site_ms"] = rec["wall_ms"].get("site:Transition", {}).get("ms")
            rec["transition_calls"] = rec["wall_ms"].get("site:Transition", {}).get("calls")
            rec["block_wall_ms"] = rec["wall_ms"].get("block:PairformerLayer", {}).get("ms")
            rec["block_calls"] = rec["wall_ms"].get("block:PairformerLayer", {}).get("calls")
            print(f"  h={arm}: fold {fold_s:.3f}s  block {rec['block_wall_ms']} ms  "
                  f"Transition {rec['transition_site_ms']} ms over {rec['transition_calls']} calls  "
                  f"plddt {m.get('plddt')}", flush=True)
            print(f"      fc1 {rec['fc1_shapes']}", flush=True)
        except Exception as e:                                                  # noqa: BLE001
            rec = {"h": arm, "THROW": f"{type(e).__name__}: {e}",
                   "traceback_tail": traceback.format_exc()[-2500:],
                   "fc1_shapes": dict(FC1),
                   "transition_shapes": {f"{k[0]}|{k[1]}": v for k, v in TRSHAPE.items()}}
            print(f"  h={arm}: THROW {type(e).__name__}: {str(e)[:600]}", flush=True)
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1, default=str))

    # --- QC: torch.equal at the fold's own shape, every height, incl. ragged ---------------------
    # The production Transition instance and a real pair-shaped input; one output per height,
    # compared on host. A chunk-height change is a claim about arithmetic and this is the test.
    par = {}
    try:
        xt = GRAB["x"]
        grab = GRAB
        par["shape"] = list(xt.shape)
        dev = STATE["dev"]
        outs = {}
        for h in [int(s) for s in a.parity_heights.split(",")]:
            xz = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            STATE["h"] = h
            try:
                T.TRANSITION_H_CHUNK_SIZE = h
                y = ORIG_TR(grab["inst"], xz)
                outs[h] = ttnn.to_torch(y).clone()
                ttnn.deallocate(y)
            except Exception as e:                                              # noqa: BLE001
                par[f"h{h}_THROW"] = f"{type(e).__name__}: {e}"[:600]
            finally:
                T.TRANSITION_H_CHUNK_SIZE = PROD_H
                ttnn.deallocate(xz)
        ref = outs.get(PROD_H)
        if ref is not None:
            H = int(xt.shape[1])
            for h, y in outs.items():
                n = -(-H // h)
                eff = -(-H // n)
                last = H - eff * (n - 1)
                d = (y.float() - ref.float())
                par[f"h{h}"] = {"n_chunks": n, "eff_h": eff, "last_chunk": last,
                                "ragged": last != eff,
                                "torch_equal": bool(torch.equal(y, ref)),
                                "max_abs": float(d.abs().max()),
                                "rel_rmsd": float((d.pow(2).mean().sqrt() /
                                                   ref.float().pow(2).mean().sqrt()))}
                print("  parity", h, par[f"h{h}"], flush=True)
    except Exception as e:                                                      # noqa: BLE001
        par["error"] = f"{type(e).__name__}: {e}"[:600]
        par["tb"] = traceback.format_exc()[-1500:]
    res["parity"] = par

    # --- ratios --------------------------------------------------------------------------------
    ok = [r for r in res["runs"] if "fold_s" in r]
    base = [r for r in ok if r["h"] == PROD_H]
    rat = {}
    if len(base) > 1:
        rat["aa_floor_block_ms"] = round(abs(base[0]["block_wall_ms"] - base[1]["block_wall_ms"]), 2)
        rat["aa_floor_transition_ms"] = round(abs(base[0]["transition_site_ms"]
                                                  - base[1]["transition_site_ms"]), 2)
        rat["aa_floor_fold_s"] = round(abs(base[0]["fold_s"] - base[1]["fold_s"]), 3)
    if base:
        b_blk = st.median([r["block_wall_ms"] for r in base])
        b_tr = st.median([r["transition_site_ms"] for r in base])
        b_fold = st.median([r["fold_s"] for r in base])
        for r in ok:
            if r["h"] == PROD_H:
                continue
            rat[f"h{r['h']}"] = {
                "transition_saving_ms_per_fold": round(b_tr - r["transition_site_ms"], 2),
                "block_saving_ms_per_fold": round(b_blk - r["block_wall_ms"], 2),
                "fold_saving_ms": round((b_fold - r["fold_s"]) * 1e3, 1),
                "transition_ratio": round(b_tr / r["transition_site_ms"], 4)}
    res["ratios"] = rat
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(rat, indent=1), flush=True)
    print("wrote", a.out, flush=True)


main()
