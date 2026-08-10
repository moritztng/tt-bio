#!/usr/bin/env python3
"""One real 298 aa protenix-v2 fold, with the matmul family instrumented.

Two products, both taken from the production path (no square stand-in, no synthetic block):

  inventory  every `ttnn.linear` / `ttnn.matmul` / `minimal_matmul` call in the fold, keyed by
             tt_bio call site and padded operand shape, with each operand's BUFFER TYPE and
             whether the call passes `core_grid=` or a `program_config=`. That is deliverable 4:
             the DRAM-interleaved-in0 + `core_grid=` sites, read out of a live fold rather than
             out of the source.
  site wall  for the sites named by --time-site, the per-call device time measured IN PLACE with
             a synchronise on both sides, summed over the fold. 15 ms/fold is 0.05 % of a 31.9 s
             fold wall, so the fold wall cannot resolve it and the site wall is the production
             number that can.

Free L1 per bank is sampled at the first N calls of each timed site, which is the capacity
question deliverable 3 candidate 1 asks at the actual call site inside a live block.

  TT_VISIBLE_DEVICES=1 python3 perf/p3narrow/fold_probe.py --narrow-bw 1 --out x.json
"""
import argparse, json, statistics as st, sys, time, traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OPS = [(("linear",), None), (("matmul",), None)]
STATE = {"on": False, "dev": None, "time_sites": (), "l1_samples": defaultdict(list)}
INV = defaultdict(lambda: {"n": 0})
WALL = defaultdict(lambda: {"n": 0, "s": 0.0, "us": []})


def desc(t):
    import ttnn
    try:
        mc = t.memory_config()
        return {"shape": list(t.padded_shape), "logical": list(t.shape),
                "dtype": str(t.dtype).split(".")[-1],
                "buf": str(mc.buffer_type).split(".")[-1]}
    except Exception:                                                     # noqa: BLE001
        return None


def call_site():
    chain = [f"{fr.filename.split('/')[-1]}:{fr.lineno}"
             for fr in reversed(traceback.extract_stack())
             if "tt_bio/" in fr.filename]
    return (chain[0] if chain else "?"), chain[:4]


def make_wrap(name, fn):
    import ttnn

    def inner(*a, **kw):
        if not STATE["on"]:
            return fn(*a, **kw)
        site, chain = call_site()
        ins = [desc(v) for v in list(a) + list(kw.values()) if isinstance(v, ttnn.Tensor)]
        STATE["on"] = False
        try:
            out = fn(*a, **kw)
            first = out[0] if isinstance(out, (list, tuple)) and out else out
            key = json.dumps({"op": name, "site": site, "chain": chain,
                              "in": ins, "out": desc(first) if isinstance(first, ttnn.Tensor) else None,
                              "core_grid": "core_grid" in kw and kw["core_grid"] is not None,
                              "program_config": "program_config" in kw and kw["program_config"] is not None,
                              "bias": "bias" in kw and kw["bias"] is not None,
                              "activation": bool(kw.get("activation"))}, sort_keys=True)
            INV[key]["n"] += 1
            hit = next((c for c in chain if c in STATE["time_sites"]), None)
            if hit:
                # Key the wall by site AND operand class, not by line alone: `_KeyedWeights._lin`
                # is one line serving 12130 calls of a dozen different shapes in a fold, so a
                # per-line wall is not a per-op-class wall and the two arms are not comparable.
                i0 = ins[0]["shape"] if ins and ins[0] else None
                ow = (desc(first) or {}).get("shape", [None])[-1] if isinstance(first, ttnn.Tensor) else None
                hit = f"{hit}|in0={i0}|out_w={ow}"
                site = hit
                dev = STATE["dev"]
                w = WALL[hit]
                if len(STATE["l1_samples"][hit]) < 3:
                    mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
                    STATE["l1_samples"][hit].append(
                        {"largest_contig_free_per_bank": int(mv.largest_contiguous_bytes_free_per_bank),
                         "total_free_per_bank": int(mv.total_bytes_free_per_bank)})
                try:
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    extra = [fn(*a, **kw) for _ in range(2)]
                    ttnn.synchronize_device(dev)
                    dt = (time.perf_counter() - t0) / 2
                    del extra
                    w["n"] += 1
                    w["s"] += dt
                    w["us"].append(round(dt * 1e6, 2))
                except Exception as e:                                    # noqa: BLE001
                    w.setdefault("err", str(e)[:120])
                    ttnn.synchronize_device(dev)
            return out
        finally:
            STATE["on"] = True
    return inner


def patch(ttnn):
    saved = []
    for ns, names in ((ttnn, ["linear", "matmul"]),
                      (ttnn.experimental, ["minimal_matmul"])):
        for nm in names:
            f = getattr(ns, nm, None)
            if callable(f):
                saved.append((ns, nm, f))
                setattr(ns, nm, make_wrap(nm, f))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow-bw", default="1",
                    help="_NARROW_PROJ_BW for this arm: 'none' = production core_grid, or an int")
    ap.add_argument("--repeat", type=int, default=2, help="uninstrumented timed folds")
    ap.add_argument("--time-site", action="append", default=[],
                    help="tt_bio file:line to measure the site wall of; repeatable")
    ap.add_argument("--inventory", action="store_true", help="run one instrumented fold")
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    T._NARROW_PROJ_BW = None if a.narrow_bw == "none" else int(a.narrow_bw)
    T._pair_proj_program_config.cache_clear()

    one_fold, meta, _state = B.build_fold("protenix-v2", ROOT / ".msa_p3narrow", a.target, a.a3m)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"

    times = []
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(round(t, 3))

    res = {"narrow_bw": a.narrow_bw, "cold_s": round(cold_s, 3), "fold_s": times,
           "median_fold_s": st.median(times) if times else None,
           "min_fold_s": min(times) if times else None,
           "n_tokens": cold_m.get("n_tokens"), "plddt": cold_m.get("plddt"),
           "grid": list(T.COMPUTE_GRID_MAIN),
           "pair_proj_bw": T._PAIR_PROJ_BW, "narrow_proj_bw": T._NARROW_PROJ_BW}

    if a.inventory or a.time_site:
        STATE["dev"] = T.get_device()
        STATE["time_sites"] = tuple(a.time_site)
        saved = patch(ttnn)
        STATE["on"] = True
        t_inst, _ = one_fold()
        STATE["on"] = False
        for ns, nm, f in saved:
            setattr(ns, nm, f)
        res["instrumented_fold_s"] = round(t_inst, 3)
        res["inventory"] = sorted(
            ({**json.loads(k), "n": v["n"]} for k, v in INV.items()),
            key=lambda r: -r["n"])
        res["site_wall"] = {k: {"calls": v["n"], "wall_ms": round(v["s"] * 1e3, 3),
                                "median_us": round(st.median(v["us"]), 2) if v["us"] else None,
                                "us": v["us"][:8],
                                **({"err": v["err"]} if "err" in v else {})}
                            for k, v in WALL.items()}
        res["l1_free_at_site"] = {k: v for k, v in STATE["l1_samples"].items()}

    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "inventory"}, indent=1), flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
