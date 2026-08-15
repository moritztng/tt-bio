"""Byte model for Boltz-2's two repeated units, from ttnn graph capture.

Tracy is compiled out of the pip wheel the Galaxy runs, so per-op attribution has to come from
`ttnn.graph`. Capture records, per op, every tensor it touched with its buffer type and size in
bytes, and a `duration_ns`. Summing the DRAM-resident tensor sizes inside one top-level op's span
(deduplicated by tensor_id, so an op reading the same tensor twice is counted once) gives that
op's inputs+outputs in bytes, which is the quantity the DRAM roof prices.

Two units are captured, each on its SECOND call so the capture is warm and carries no compile:

  * `TrunkModule._iteration` -- one recycle of MSA + Pairformer. A fold runs recycling_steps+1.
  * `AtomDiffusion.preconditioned_network_forward` -- one diffusion step's denoiser. A fold runs
    sampling_steps of them.

Fold traffic is then (recycles+1) * trunk + steps * step, and the residue is whatever the measured
wall has left over. Cross-check before using it: total_bytes / measured_wall must not exceed the
measured 218.5 GB/s DRAM roof, and the trimul tails must come out at the 526 GB the state doc
already prices independently.
"""
import argparse, json, os, sys, time
from pathlib import Path


def spans(trace):
    """Top-level ops in one capture: (name, bytes by buffer type, duration_ns)."""
    out, i, n = [], 0, len(trace)
    while i < n:
        nd = trace[i]
        if nd["node_type"] == "function_start" and nd["stacking_level"] == 1:
            name = nd["params"]["name"]
            j, seen, by = i + 1, set(), {}
            dur = None
            while j < n:
                m = trace[j]
                if m["node_type"] == "function_end" and m["stacking_level"] == 1:
                    dur = m.get("duration_ns")
                    break
                if m["node_type"] == "tensor":
                    p = m["params"]
                    tid = p.get("tensor_id")
                    if tid not in seen:
                        seen.add(tid)
                        bt = "DRAM" if "DRAM" in str(p.get("buffer_type", "")) else "L1"
                        by[bt] = by.get(bt, 0) + int(p.get("size", 0))
                j += 1
            out.append({"name": name, "bytes": by, "duration_ns": dur, "n_tensors": len(seen)})
            i = j + 1
            continue
        i += 1
    return out


def summarise(trace):
    ops = spans(trace)
    agg = {}
    for o in ops:
        a = agg.setdefault(o["name"], {"calls": 0, "dram_b": 0, "l1_b": 0, "ns": 0})
        a["calls"] += 1
        a["dram_b"] += o["bytes"].get("DRAM", 0)
        a["l1_b"] += o["bytes"].get("L1", 0)
        a["ns"] += o["duration_ns"] or 0
    tot = {"ops": len(ops),
           "dram_b": sum(a["dram_b"] for a in agg.values()),
           "l1_b": sum(a["l1_b"] for a in agg.values()),
           "ns": sum(a["ns"] for a in agg.values())}
    top = sorted(agg.items(), key=lambda kv: -kv[1]["dram_b"])
    return {"total": tot, "by_op": dict(top)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(tree)

    B.RECYCLING_STEPS = a.recycles if a.recycles is not None else _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = a.steps if a.steps is not None else _resolve_sampling_steps(None, "boltz2")

    snap = list(sys.path)
    sys.path.insert(0, str(tree / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    from tt_bio import boltz2 as BZ

    res = {"size": a.size, "recycles": B.RECYCLING_STEPS, "steps": B.SAMPLING_STEPS,
           "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "units": {}}
    armed = {"on": False}
    counts = {"trunk": 0, "step": 0}

    def wrap(cls, attr, key, capture_on_call):
        orig = getattr(cls, attr)

        def w(self, *args, **kw):
            counts[key] += 1
            if not (armed["on"] and counts[key] == capture_on_call and key not in res["units"]):
                return orig(self, *args, **kw)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                r = orig(self, *args, **kw)
            finally:
                tr = ttnn.graph.end_graph_capture()
            res["units"][key] = summarise(tr)
            res["units"][key]["captured_call"] = counts[key]
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"[bytes] captured {key}: {res['units'][key]['total']}", flush=True)
            return r
        setattr(cls, attr, w)

    wrap(T.TrunkModule, "_iteration", "trunk", 2)
    wrap(BZ.AtomDiffusion, "preconditioned_network_forward", "step", 2)

    fixdir = tree / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    msa_dir = tree / f".msa_xmodel_boltz2_{a.size}"
    one_fold, meta = B.build_fold("boltz2", msa_dir, tgt, a3m)[:2]

    t0 = time.perf_counter()
    warm_s, _m = one_fold()
    print(f"[bytes] cold fold {warm_s:.3f}s", flush=True)
    res["cold_fold_s"] = round(warm_s, 3)

    armed["on"] = True
    counts["trunk"] = counts["step"] = 0
    cap_s, _m = one_fold()
    res["captured_fold_s"] = round(cap_s, 3)
    res["grid"] = [int(T.COMPUTE_GRID_MAIN[0]), int(T.COMPUTE_GRID_MAIN[1])]
    print(f"[bytes] capture fold {cap_s:.3f}s (inflated by capture, NOT a wall)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
