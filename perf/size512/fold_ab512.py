#!/usr/bin/env python3
"""Deliverable 1 — the ON/OFF fold A/B for the org's L1-capacity-gated wins, swept over size.

The question is whether C2FIX + X7 + X10 (1664.5 of the org's 1704.4 ms/fold, all behind a silent
`if it fits` branch) still pay at 512 aa. The analytic pass says all three take the DRAM branch at
512; this measures the fold, reads the branch each call actually took, and hashes the CIF so a null
result is provably a no-op rather than a broken run.

One process, one device context, several targets and several arms. The gate flags are module
globals read at call time, so an arm is a flag flip between folds -- no reload, no second device
open, and the weights and the MSA cache are shared, which is what makes a 4-arm sweep fit in a turn.

Arms:
  on   production defaults; every gate decides for itself.
  off  the five capacity-gated wins forced off:
         _transpose_memory_config -> DRAM   (C2FIX)
         _PAIR_PROJ_L1_OUT = False          (X7 L1 output)
         _PAIR_BIAS_L1_NORM = False         (X7 L1 layer_norm source)
         _PWA_L1_NORM = False               (X10)
         _TEMPLATE_L1_NORM = False          (X10)
       `_PAIR_PROJ_BW` and `_NARROW_PROJ_BW` are identical in both arms: they are program config,
       not capacity, and moving them would change parity as well as the thing under test.
  rb_fit  every L1 gate exactly as `on` leaves it, plus `_TRANSPOSE_ROWBLOCK`: the row-blocked
          pair transpose at its shipping default, R=64 and the group the L1 budget takes at 2.5x
          headroom. This is the arm a merge would ship.
  rb_max  the same, with the group forced to every block, so the whole transposed tensor is L1
          resident and the assembly is a single concat. Fastest in the probe and the largest L1
          footprint; measured here so the interaction checks can price the difference.

The headline instrument is the block wall, `PairformerLayer.__call__` synchronised on both sides and
summed over its executions, not the fold wall -- this harness's fold-wall A/A floor is 758.3 ms and
cannot resolve a null. Run the `on` arm twice to get this session's own A/A floor.

Results are written after every fold, so a turn that runs out of time still lands what it measured.
"""
import argparse, hashlib, json, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
STATE = {"dev": None, "gates": "on"}
HOST, CHIP, PROVENANCE = "qb1", "3", ""


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


def sha_dir(d: Path):
    """sha256 of every structure file the fold just wrote, by name."""
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512", help="comma list of fixture token counts")
    ap.add_argument("--arms", default="on,on,off", help="comma list, run in this order per size")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default="qb1")
    ap.add_argument("--chip", default="3")
    a = ap.parse_args()
    global HOST, CHIP, PROVENANCE
    HOST, CHIP = a.host, a.chip
    PROVENANCE = ("qb1 at ttnn 0.67.4 -- campaign absolute" if a.host == "qb1" else
                  "qb2 is ttnn 0.68.0 -- every absolute here is a ratio input, not a campaign number")

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B

    # ---- decision counters: read the branch actually taken, do not infer it from the shape ----
    ORIG_TMC = T._transpose_memory_config
    ORIG_LN = T._l1_layer_norm
    ORIG_PPC = T._pair_proj_config

    def is_l1(mc):
        return mc.buffer_type == ttnn.BufferType.L1

    def tmc(t):
        mc = ttnn.DRAM_MEMORY_CONFIG if STATE["gates"] == "off" else ORIG_TMC(t)
        DEC[f"transpose|{'x'.join(str(int(d)) for d in t.shape)}"]["L1" if is_l1(mc) else "DRAM"] += 1
        return mc

    def ln(x, headroom, **kw):
        out, in_l1 = ORIG_LN(x, headroom, **kw)
        DEC[f"l1_layer_norm|h={headroom}|{'x'.join(str(int(d)) for d in x.shape)}"][
            "L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    def ppc(x, w, bw_cap=-1, out_l1=False):
        cfg = ORIG_PPC(x, w, bw_cap=bw_cap, out_l1=out_l1)
        if out_l1:
            DEC[f"pair_proj_out_l1|{'x'.join(str(int(d)) for d in x.shape)}"
                f"@{int(list(w.shape)[-1])}"]["L1" if cfg is not None else "DRAM"] += 1
        return cfg

    # `_pair_transpose` looks `_rowblock_plan` up as a module global, so this reads the branch the
    # fold actually took -- blocked or not, and at which (R, group).
    ORIG_PLAN = T._rowblock_plan

    def plan(t):
        pl = ORIG_PLAN(t)
        DEC[f"rowblock|{'x'.join(str(int(d)) for d in t.shape)}"][
            f"blocked R={pl[0]},group={pl[1]}" if pl else "unblocked"] += 1
        return pl

    # Peak L1 occupancy is the moment a group is live and not yet flushed, which is exactly the
    # entry to `_rowblock_flush`. `get_memory_view` drains the pipeline, so sample a few times.
    ORIG_FLUSH = T._rowblock_flush
    L1FREE = {"n": 0, "samples": []}

    def flush(live):
        if L1FREE["n"] < 8:
            L1FREE["n"] += 1
            try:
                mv = ttnn.get_memory_view(STATE["dev"], ttnn.BufferType.L1)
                L1FREE["samples"].append({
                    "live_blocks": len(live),
                    "banks": int(mv.num_banks),
                    "total_bytes_per_bank": int(mv.total_bytes_per_bank),
                    "free_bytes_per_bank": int(mv.total_bytes_free_per_bank),
                    "largest_contiguous_free_per_bank": int(
                        mv.largest_contiguous_bytes_free_per_bank)})
            except Exception as e:                                              # noqa: BLE001
                L1FREE["samples"].append({"error": f"{type(e).__name__}: {e}"[:120]})
        return ORIG_FLUSH(live)

    T._rowblock_plan = plan
    T._rowblock_flush = flush
    T._transpose_memory_config = tmc
    T._l1_layer_norm = ln
    P._l1_layer_norm = ln          # protenix.py imports it by name, so patch both namespaces
    T._pair_proj_config = ppc

    saved = []
    for cls in (T.Pairformer,):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g: lambda self, *x, **k: timed_call("stage:Pairformer", g, self, *x, **k))(f)
    for cls in (T.PairformerLayer,):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g: lambda self, *x, **k: timed_call("block:PairformerLayer", g, self, *x, **k))(f)
    for cls in (T.TriangleMultiplication, T.TriangleAttention, T.AttentionPairBias,
                T.PairWeightedAveraging):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g, nm: lambda self, *x, **k: timed_call(f"body:{nm}", g, self, *x, **k))(f, cls.__name__)

    def set_arm(name):
        STATE["gates"] = name
        on = name != "off"          # every arm but `off` leaves production's L1 gates alone
        T._PAIR_PROJ_L1_OUT = on
        T._PAIR_BIAS_L1_NORM = on
        T._PWA_L1_NORM = on
        T._TEMPLATE_L1_NORM = on
        T._TRANSPOSE_ROWBLOCK = name.startswith("rb")
        T._TRANSPOSE_ROWBLOCK_R = 0                     # 0 = the shipping default, R=64
        T._TRANSPOSE_ROWBLOCK_GROUP = 10 ** 6 if name == "rb_max" else 0   # clamped to ceil(S/R)
        T._pair_proj_program_config.cache_clear()
        T._L1_OUT_REFUSED.clear()
        L1FREE["n"], L1FREE["samples"] = 0, []

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": HOST, "chip": CHIP, "note": PROVENANCE, "runs": []}

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        set_arm("on")
        one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / f".msa_s512_{size}", tgt, a3m)
        STATE["dev"] = T.get_device()
        struct_dir = Path(meta["struct_dir"])
        print(f"=== size {size}: cold fold ===", flush=True)
        try:
            cold_s, cold_m = one_fold()
        except Exception as e:                                                  # noqa: BLE001
            res["runs"].append({"size": size, "arm": "cold", "error": f"{type(e).__name__}: {e}"[:400]})
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  COLD FOLD FAILED at {size}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        assert cold_m.get("msa"), "fold ran without an MSA"
        print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}",
              flush=True)
        WALL.clear()
        DEC.clear()

        for arm in a.arms.split(","):
            set_arm(arm)
            WALL.clear()
            DEC.clear()
            t0 = time.perf_counter()
            try:
                fold_s, m = one_fold()
            except Exception as e:                                              # noqa: BLE001
                rec = {"size": size, "arm": arm, "error": f"{type(e).__name__}: {e}"[:400]}
                res["runs"].append(rec)
                a.out.write_text(json.dumps(res, indent=1))
                print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            rec = {"size": size, "arm": arm, "fold_s": round(fold_s, 3),
                   "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                   "cif_sha256": sha_dir(struct_dir),
                   "grid": list(T.COMPUTE_GRID_MAIN),
                   "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                               for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
                   "decisions": {k: dict(v) for k, v in sorted(DEC.items())},
                   "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED],
                   "l1_free_per_bank_at_rowblock_peak": L1FREE["samples"]}
            blk = rec["wall_ms"].get("block:PairformerLayer", {})
            rec["block_wall_ms"] = blk.get("ms")
            rec["block_calls"] = blk.get("calls")
            res["runs"].append(rec)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm}: fold {fold_s:.2f}s  block {blk.get('ms')} ms over {blk.get('calls')} "
                  f"calls  plddt {m.get('plddt')}  ({time.perf_counter()-t0:.0f}s)", flush=True)
            for k, v in sorted(DEC.items()):
                print(f"      DEC {k:52s} {dict(v)}", flush=True)

    # ---- per-size ON/OFF ratios ----------------------------------------------------------
    ratios = {}
    for size in {r["size"] for r in res["runs"] if "block_wall_ms" in r}:
        on = [r["block_wall_ms"] for r in res["runs"] if r["size"] == size and r["arm"] == "on"]
        off = [r["block_wall_ms"] for r in res["runs"] if r["size"] == size and r["arm"] == "off"]
        onf = [r["fold_s"] for r in res["runs"] if r["size"] == size and r["arm"] == "on"]
        offf = [r["fold_s"] for r in res["runs"] if r["size"] == size and r["arm"] == "off"]
        e = {"on_block_ms": on, "off_block_ms": off, "on_fold_s": onf, "off_fold_s": offf}
        if len(on) > 1:
            e["aa_floor_block_ms"] = round(abs(on[0] - on[1]), 2)
            e["aa_floor_fold_s"] = round(abs(onf[0] - onf[1]), 3)
        if on and off:
            e["off_over_on_block"] = round(st.median(off) / st.median(on), 4)
            e["off_minus_on_block_ms"] = round(st.median(off) - st.median(on), 2)
            e["off_over_on_fold"] = round(st.median(offf) / st.median(onf), 4)
        for arm in sorted({r["arm"] for r in res["runs"] if r["size"] == size} - {"on", "off"}):
            b = [r["block_wall_ms"] for r in res["runs"]
                 if r["size"] == size and r["arm"] == arm and r.get("block_wall_ms")]
            f = [r["fold_s"] for r in res["runs"] if r["size"] == size and r["arm"] == arm]
            if b and on:
                e[f"{arm}_block_ms"] = b
                e[f"on_minus_{arm}_block_ms"] = round(st.median(on) - st.median(b), 2)
                e[f"on_over_{arm}_block"] = round(st.median(on) / st.median(b), 4)
                e[f"on_minus_{arm}_fold_s"] = round(st.median(onf) - st.median(f), 3)
        ratios[size] = e
    res["ratios"] = ratios
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(ratios, indent=1), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
