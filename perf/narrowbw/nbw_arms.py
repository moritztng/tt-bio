#!/usr/bin/env python3
"""The _NARROW_PROJ_BW cap sweep at 512 aa and 298 aa: one cap at a time, bracketed.

Adapted from perf/survival512/surv_arms.py (protenix-trunk--z-survival-512) rather than re-derived.
Its census design, its bracketed arms, its per-key A/A floor and its counted denominators are
inherited unchanged; what changes is the arm table and one census field.

WHY. z-survival-512 measured _NARROW_PROJ_BW = 1 as worth 407.89 ms/fold at 512 aa against
60.37 at 298 aa, 6.76x, and the only flag in the merged family that is not capacity-gated. The
constant caps in0_block_w, and its own comment records 1.98x / 2.08x at cap 16 against 1.15x / 1.23x
at cap 1 -- so the site the org has just learned is its largest single 512 aa row also has an
untaken doubling sitting on it, behind a parity decision. This sweeps the cap from OUTSIDE: the
module default stays at 1 and nothing in tt_bio/ is touched (Phase 2).

WHY THIS FILE EXISTS. `size512-ab` reported that 27.6 % of the org's merged L1-gated gain survives
at 512 aa (+476.96 ms of block wall against 1729.03 ms at 298 aa). `z-h5-infold` then showed that
number is under-powered: it moved FIVE flags and credited one, and its 476.96 is 0.68 % of the wall
it was taken on, where the same harness family's own single-shot arm manufactured +898.89 ms of pure
drift on that wall. This re-takes the decomposition with an instrument that can resolve it.

THIS IS A THROWAWAY EXPERIMENT HARNESS. It is not production code, it changes nothing in `tt_bio/`,
and every flag is flipped from outside on the module globals, which are read at call time.

THREE THINGS IT DOES DIFFERENTLY FROM ITS PREDECESSOR.

1. ONE FLAG AT A TIME, plus the two family arms for the total. `--arms` is an ordered list and every
   `off:*` arm is BRACKETED by an `on` arm on both sides, so each delta is taken against a baseline
   measured either side of it and linear drift cancels.

2. THE FLOOR COMES FROM >=3 ON ARMS, NOT FROM A PAIR. `size512-ab` quoted a 2.10 ms A/A floor at
   512 aa from two ON arms and called its 476.96 "227x the floor"; `z-h5-infold`'s doubled arms on
   the same wall on the same chip measured a spread of 64.9 ms, 31x larger. |a-b| from one pair is
   not an estimate of a drift band. Every `on` arm here contributes to a spread AND a stdev per wall
   key, and no delta in the writeup may claim an effect smaller than its own key's spread.

3. THE INSTRUMENT IS SYMMETRIC BY CONSTRUCTION. The timed things are ttnn ops (`layer_norm`,
   `linear`) and class bodies, which BOTH arms execute; the production helpers (`_l1_layer_norm`,
   `_narrow_proj_linear`, `_pair_proj_linear`, `_transpose_memory_config`) are wrapped for CENSUS
   ONLY, never timed. Timing `_l1_layer_norm` would have timed the norm in the ON arm and not in the
   OFF arm, because the OFF arm does not call it -- a wall that exists in one arm only.

THE CENSUS IS THE CHEAPEST HALF OF THE ANSWER. A flag whose fit test refuses at 512 aa emits the
identical ttnn call in both arms, so its arm is an A/A BY CONSTRUCTION and its value is zero without
any timing argument. That is read off the branch actually taken -- `_l1_layer_norm`'s own returned
bool, the returned tensor's `memory_config().buffer_type`, whether `_narrow_proj_linear` returned
None, and whether `_transpose_memory_config` returned L1 -- never inferred from the shape.

Usage (qb2 chip 0; ttnn 0.68.0, so every absolute is a RATIO, charter 4.8):

  SP=~/tt-bio-dev/env/lib/python3.10/site-packages
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-survival-512 \\
  TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \\
  ~/tt-bio-dev/env/bin/python3 perf/survival512/surv_arms.py --size 512 \\
      --arms on,off:ab5,on,off:narrowbw,on,off:projl1,on,off:norms,on,off:c2fix,on,off:family,on \\
      --out perf/survival512/surv_512_qb2c0.json

Results are written after every fold, so a turn that runs out of time lands what it measured.
"""
import argparse, hashlib, json, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
CENSUS = Counter()
STATE = {"dev": None, "site": ["other"], "c2fix_off": False}

# One arm = one dict of module-global overrides on tt_bio.tenstorrent. Everything not named here
# stays at its production default in every arm.
#   family  the five flags of the merged 685.1 ms/fold at 298 aa (X7 561.8 + X2 31.5 + X10 91.8)
#   ab5     the five `size512-ab` actually moved for its 476.96 -- C2FIX instead of _NARROW_PROJ_BW.
#           Read off perf/size512/fold_ab512.py, not off the brief.
# One arm = one dict of module-global overrides on tt_bio.tenstorrent. Everything not named here
# stays at its production default in every arm.
#   on            the production default, _NARROW_PROJ_BW = 1. Run 6-7 times for the A/A floor.
#   off:narrowbw  cap None, the pre-X2 core_grid= path. Reproduces z-survival-512's 407.89 ms/fold
#                 at 512 aa and 60.37 at 298 aa, which is the cross-check on this instrument: if
#                 this arm misses those, nothing after it is quotable.
#   bw:2/4/8/16   the sweep. At k_tiles = 8 (c_z = 256) the helper picks
#                 in0_block_w = max(d in (k_tiles,8,4,2,1) : d <= cap and k_tiles % d == 0),
#                 so cap 16 and cap 8 both select 8 and bw:16 is a FREE A/A control on bw:8 --
#                 identical program config, identical plDDT, identical CIF, delta inside the floor.
#   bw:8+pairbw1  independence: cap 8 on the narrow sites while _PAIR_PROJ_BW and _PAIR_PROJ_L1_BW
#                 drop to 1. The three constants are separate arguments to one helper and their
#                 call sites are disjoint in the census, so the narrow site walls here must match
#                 bw:8's inside the floor. If they do not, the interaction is the finding.
ARMS = {
    "on": {},
    "off:narrowbw": {"_NARROW_PROJ_BW": None},
    "bw:2": {"_NARROW_PROJ_BW": 2},
    "bw:4": {"_NARROW_PROJ_BW": 4},
    "bw:8": {"_NARROW_PROJ_BW": 8},
    "bw:16": {"_NARROW_PROJ_BW": 16},
    "bw:8+pairbw1": {"_NARROW_PROJ_BW": 8, "_PAIR_PROJ_BW": 1, "_PAIR_PROJ_L1_BW": 1},
}


def sync():
    import ttnn
    ttnn.synchronize_device(STATE["dev"])


def timed(key, fn, *a, **kw):
    """Synchronised on BOTH sides (ttnn-sync-before-every-timed-region). The only timing primitive."""
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
    try:
        return int(t.shape[-1])
    except Exception:                                                          # noqa: BLE001
        return -1


def shp(t):
    try:
        return str(tuple(int(d) for d in t.padded_shape))
    except Exception:                                                          # noqa: BLE001
        return "?"


def site():
    return STATE["site"][-1]


def sha_dir(d):
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def install(T, P, ttnn):
    """Time ttnn ops and class bodies (symmetric); census the production helpers (not timed)."""
    O = {"ln": ttnn.layer_norm, "lin": ttnn.linear,
         "l1ln": T._l1_layer_norm, "npl": T._narrow_proj_linear, "ppl": T._pair_proj_linear,
         "tmc": T._transpose_memory_config}

    # ---- timed: every layer_norm and linear at a tracked site, in BOTH arms -----------------------
    def ln(x, *a, **kw):
        s = site()
        if s == "other":
            return O["ln"](x, *a, **kw)
        return timed(f"norm|{s}|c{chan(x)}", O["ln"], x, *a, **kw)

    def lin(x, w, *a, **kw):
        s = site()
        if s == "other":
            return O["lin"](x, w, *a, **kw)
        try:
            outw = int(list(w.shape)[-1])
        except Exception:                                                      # noqa: BLE001
            outw = -1
        return timed(f"lin|{s}|c{chan(x)}@{outw}", O["lin"], x, w, *a, **kw)

    # ---- census only: the branch each production helper actually took -----------------------------
    def l1ln(x, headroom, **kw):
        out, in_l1 = O["l1ln"](x, headroom, **kw)
        CENSUS[("l1_layer_norm", site(), shp(x), f"headroom={headroom}",
                "L1" if in_l1 else "DRAM")] += 1
        return out, in_l1

    def npl(x, w, ckc, dtype, l1_out=False):
        out = O["npl"](x, w, ckc, dtype, l1_out=l1_out)
        if out is None:
            br = "None(core_grid fallback)"
        else:
            br = "L1" if out.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
        # The cap's ENTIRE mechanism is in0_block_w, and a cap the L1 budget refuses returns None
        # and falls back to core_grid -- slower than cap 1. Read the config, never assume it.
        cfg = T._pair_proj_config(x, w, bw_cap=T._NARROW_PROJ_BW, out_l1=l1_out)
        if cfg is None and l1_out:
            cfg = T._pair_proj_config(x, w, bw_cap=T._NARROW_PROJ_BW)
        arg = (f"cap={T._NARROW_PROJ_BW} bw={cfg.in0_block_w} pcM={cfg.per_core_M} "
               f"sw={cfg.out_subblock_w}" if cfg is not None
               else f"cap={T._NARROW_PROJ_BW} cfg=None")
        CENSUS[("narrow_proj_linear", site(), shp(x), str(tuple(int(d) for d in w.shape)),
                f"{br} [{arg}]")] += 1
        return out

    def ppl(x, w, ckc, dtype, l1_out=False):
        out = O["ppl"](x, w, ckc, dtype, l1_out=l1_out)
        try:
            br = "L1" if out.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
        except Exception:                                                      # noqa: BLE001
            br = "?"
        CENSUS[("pair_proj_linear", site(), shp(x), str(tuple(int(d) for d in w.shape)),
                ("l1_out " if l1_out else "plain ") + br)] += 1
        return out

    def tmc(t):
        mc = ttnn.DRAM_MEMORY_CONFIG if STATE["c2fix_off"] else O["tmc"](t)
        CENSUS[("transpose_memory_config", site(), shp(t), "",
                "L1" if mc is ttnn.L1_MEMORY_CONFIG else "DRAM")] += 1
        return mc

    ttnn.layer_norm = ln
    ttnn.linear = lin
    for M in (T, P):                    # protenix.py imports both helpers by name
        M._l1_layer_norm = l1ln
        M._narrow_proj_linear = npl
    T._pair_proj_linear = ppl
    T._transpose_memory_config = tmc

    # ---- timed: stage, block and body walls, with the site stack the census keys off -------------
    def body(cls, name, tag):
        f = cls.__call__

        def g(self, *x, **k):
            c = chan(x[0]) if x and hasattr(x[0], "shape") else -1
            if name == "PairWeightedAveraging" and len(x) > 1:
                c = chan(x[1])                                   # PWA's pair operand is z, not m
            STATE["site"].append(tag)
            try:
                return timed(f"body:{name}|c{c}", f, self, *x, **k)
            finally:
                STATE["site"].pop()
        cls.__call__ = g

    body(T.TriangleMultiplication, "TriangleMultiplication", "trimul")
    body(T.TriangleAttention, "TriangleAttention", "triatt")
    body(T.AttentionPairBias, "AttentionPairBias", "pairbias")
    body(T.PairWeightedAveraging, "PairWeightedAveraging", "pwa")
    # PairformerLayer pushes its own site so the ops of a stack nested inside `_template` / `_msa` do
    # not get attributed to the template or MSA site. Measured at 298 aa before this was added:
    # `lin|template|c64@128` x1600 and `norm|template|c64` x840 are the template stack's internal
    # Pairformer ops, not the template z projection (which is `lin|template|c256@64` x40). The stage
    # walls are unaffected -- they are meant to include everything below them.
    fpfl = T.PairformerLayer.__call__

    def pfl(self, *x, **kw):
        STATE["site"].append("pfl")
        try:
            return timed("block:PairformerLayer", fpfl, self, *x, **kw)
        finally:
            STATE["site"].pop()
    T.PairformerLayer.__call__ = pfl
    fpf = T.Pairformer.__call__
    T.Pairformer.__call__ = lambda self, *x, **kw: timed("stage:Pairformer", fpf, self, *x, **kw)

    for meth, key, tag in (("_template", "stage:template", "template"), ("_msa", "stage:msa", "msa")):
        f = getattr(P.Trunk, meth)

        def g(self, *x, _f=f, _k=key, _t=tag, **kw):
            STATE["site"].append(_t)
            try:
                return timed(_k, _f, self, *x, **kw)
            finally:
                STATE["site"].pop()
        setattr(P.Trunk, meth, g)
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="on,off:narrowbw,on,bw:8,on,bw:4,on,bw:2,on,bw:16,on,"
                                      "bw:8+pairbw1,on")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--msadir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B
    import importlib.metadata as im

    DEFAULTS = {k: getattr(T, k) for k in ("_PAIR_PROJ_L1_OUT", "_PAIR_BIAS_L1_NORM", "_PWA_L1_NORM",
                                           "_TEMPLATE_L1_NORM", "_NARROW_PROJ_BW",
                                           "_PAIR_PROJ_BW", "_PAIR_PROJ_L1_BW")}

    def set_arm(name):
        over = ARMS[name]
        for k, v in DEFAULTS.items():
            setattr(T, k, over.get(k, v))
        STATE["c2fix_off"] = over.get("_C2FIX", True) is False
        T._pair_proj_program_config.cache_clear()
        T._L1_OUT_REFUSED.clear()
        return {k: getattr(T, k) for k in DEFAULTS} | {"_C2FIX": not STATE["c2fix_off"]}

    for spec in a.arms.split(","):
        if spec not in ARMS:
            sys.exit(f"unknown arm {spec!r}; known: {sorted(ARMS)}")

    install(T, P, ttnn)
    size = a.size
    set_arm("on")
    msadir = a.msadir or ROOT / f".msa_surv_{size}"
    one_fold, meta, _s = B.build_fold("protenix-v2", msadir,
                                      a.fixdir / f"cdk2x2_{size}.yaml",
                                      a.fixdir / f"cdk2x2_{size}.a3m")
    STATE["dev"] = T.get_device()
    struct_dir = Path(meta["struct_dir"])

    res = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"), "size": size,
           "grid": list(T.COMPUTE_GRID_MAIN),
           "l1_bank_bytes": T._l1_bank_bytes(),
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
           "l1_fit_budget_bytes": int(ttnn.get_max_worker_l1_unreserved_size())
                                  * T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1],
           "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
           "reblock_permute": getattr(__import__("tt_bio.reblock_permute", fromlist=["x"]),
                                      "REBLOCK_PERMUTE", "n/a"),
           "defaults": {k: str(v) for k, v in DEFAULTS.items()},
           "arms_spec": a.arms,
           "note": "qb2 at ttnn 0.68.0 -- every absolute is a RATIO owing a qb1/0.67.4 re-take",
           "runs": []}
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=str), flush=True)

    print("=== cold fold ===", flush=True)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    print(f"  cold {cold_s:.1f}s tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}",
          flush=True)

    for i, spec in enumerate(a.arms.split(",")):
        flags = set_arm(spec)
        WALL.clear()
        CENSUS.clear()
        try:
            fold_s, m = one_fold()
        except Exception as e:                                                 # noqa: BLE001
            res["runs"].append({"i": i, "arm": spec, "error": f"{type(e).__name__}: {e}"[:400]})
            a.out.write_text(json.dumps(res, indent=1, default=str))
            print(f"  [{i}] {spec} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        res["runs"].append({
            "i": i, "arm": spec, "flags": {k: str(v) for k, v in flags.items()},
            "fold_s": round(fold_s, 3), "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
            "cif_sha256": sha_dir(struct_dir),
            "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED],
            "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                        for k, v in sorted(WALL.items())},
            "census": [{"helper": k[0], "site": k[1], "padded_shape": k[2], "arg": k[3],
                        "branch_taken": k[4], "calls": n} for k, n in sorted(CENSUS.items())]})
        a.out.write_text(json.dumps(res, indent=1, default=str))
        print(f"  [{i}] {spec} fold {fold_s:.1f}s plddt {m.get('plddt')}", flush=True)
        for e in res["runs"][-1]["census"]:
            print(f"    CENSUS {e['helper']:>22s} {e['site']:>9s} {e['padded_shape']:>22s} "
                  f"{e['arg']:>14s} -> {e['branch_taken']:<24s} x{e['calls']}", flush=True)

    res["analysis"] = analyse(res["runs"])
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["analysis"], indent=1, default=str), flush=True)
    print("wrote", a.out, flush=True)


def analyse(runs):
    """Per-key A/A floor from every `on` arm, and each off arm against the arms bracketing it."""
    ok = [r for r in runs if "wall_ms" in r]
    ons = [r for r in ok if r["arm"] == "on"]
    keys = sorted({k for r in ok for k in r["wall_ms"]})
    out = {"n_on_arms": len(ons), "aa_floor_ms": {}, "on_baseline_ms": {}, "deltas_ms": {}}
    for k in keys:
        v = [r["wall_ms"][k]["ms"] for r in ons if k in r["wall_ms"]]
        if not v:
            continue
        out["aa_floor_ms"][k] = {"n": len(v), "spread": round(max(v) - min(v), 2),
                                 "stdev": round(st.stdev(v), 2) if len(v) > 1 else None,
                                 "median": round(st.median(v), 2),
                                 "calls": ons[0]["wall_ms"].get(k, {}).get("calls")}
        out["on_baseline_ms"][k] = round(st.median(v), 2)
    for r in ok:
        if r["arm"] == "on":
            continue
        before = [x for x in ok if x["i"] < r["i"] and x["arm"] == "on"]
        after = [x for x in ok if x["i"] > r["i"] and x["arm"] == "on"]
        br = ([before[-1]] if before else []) + ([after[0]] if after else [])
        d = {"bracketed_by": [x["i"] for x in br], "fold_s_delta": None, "walls": {}}
        if br:
            d["fold_s_delta"] = round(r["fold_s"] - st.mean([x["fold_s"] for x in br]), 3)
        for k in keys:
            base = [x["wall_ms"][k]["ms"] for x in br if k in x["wall_ms"]]
            if not base or k not in r["wall_ms"]:
                continue
            delta = r["wall_ms"][k]["ms"] - st.mean(base)
            calls = r["wall_ms"][k]["calls"]
            floor = out["aa_floor_ms"].get(k, {}).get("spread")
            d["walls"][k] = {"off_minus_on_ms_per_fold": round(delta, 2), "calls": calls,
                             "ms_per_call": round(delta / calls, 5) if calls else None,
                             "aa_spread_ms": floor,
                             "resolved": (floor is not None and abs(delta) > floor)}
        d["walls"] = dict(sorted(d["walls"].items(),
                                 key=lambda kv: -abs(kv[1]["off_minus_on_ms_per_fold"])))
        out["deltas_ms"][f"{r['i']}:{r['arm']}"] = d
    return out


if __name__ == "__main__":
    main()
