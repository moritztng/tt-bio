#!/usr/bin/env python3
"""Is the Boltz-2 pairformer's 2.34x cost-model deficit Blackhole's, or tt-metal's?

`b2x-op-cost-curve` measured, on one Blackhole processor of a p300c, that a PairformerLayer
costs 41.4152 ms against 17.67 ms of cost model: 428 ops at a measured 6.36 us per-op device
floor plus 6.651 GB at a measured 445.0 GB/s asymptote. 2.34x, and uniform 1.6-1.9x across the
byte-heavy sub-units. Nothing in the campaign has explained it, and it has only ever been
measured on Blackhole.

This script is that measurement, written once so it runs unchanged on Wormhole and on
Blackhole. Both roofs are measured on the part it is running on: no nameplate bandwidth, no
nameplate TFLOP/s (`roofline-roof-must-be-measured-not-asserted`). It reports the deficit as a
ratio, so a contended host changes the absolutes but not the answer.

    sweep   12 tile-aligned bf16 sizes, 64 KB to 134 MB per operand, `ttnn.add` (3 operands
            moved) and a square-ish `ttnn.matmul`. Every point is a ttnn trace holding R
            repetitions of the op, captured once and replayed, so the host pays ~1 us for the
            whole trace and what is timed is device time per op. Same 12 sizes and same
            instrument as `b2x-op-cost-curve`, so its 6.36 us / 445.0 GB/s fit on Blackhole is
            directly comparable to whatever this prints.

            BW_eff  the add curve's asymptote, the largest GB/s it reaches.
            t_fixed the flat floor at the small end, read off directly. A least-squares
                    intercept trades slope against intercept and lands below the floor
                    (4.67 us against a 6.36 us floor on Blackhole), so the fit is reported
                    for completeness and the floor is what the model uses.
            FLOP roof  the matmul curve's peak, 2*n^3 / t.

    block   one settled call each of the shipped PairformerLayer, TriangleMultiplication,
            TriangleAttention, pair-track Transition and AttentionPairBias, grabbed out of a
            live 512 aa fold and replayed from a ttnn trace. No model code is touched; the
            grabbed call runs with its own arguments.

    deficit measured device time against ops*t_fixed + bytes/BW_eff, per sub-unit and for the
            whole block, with this part's own two roofs.

The byte and op counts per sub-unit are properties of the shipped call graph, not of the part,
so they are constants here (`b2x-baseline-attrib` A1, buffer-address deduped per
`b2x-diffusion-layer-bytes`). Running the same commit on both parts is what makes that valid,
and the JSON records the commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

# Per pairformer block. Shape-derived, identical on both parts; see module docstring.
BLOCK_BYTES = 6.651e9
BLOCK_OPS = 428
SUBUNITS = {
    # class name: (calls per block, MB moved per block, ttnn ops per block)
    "TriangleMultiplication": (2, 4229.4, 62),
    "TriangleAttention":      (2, 2231.8, 54),
    "Transition":             (1, 474.2, 258),
    "AttentionPairBias":      (1, 151.4, 32),
}
# The Blackhole reading this task exists to test for arch-dependence.
BH_REFERENCE = {"card": "qb2 card 0, one Blackhole processor of a p300c, 11x10, ttnn 0.68.0",
                "t_fixed_us": 6.36, "BW_eff_GBs": 445.0,
                "PairformerLayer_ms": 41.4152, "deficit": 2.34,
                "source": "b2x-op-cost-curve, perf/b2x_op_cost/FINDINGS.md"}

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def sizes_64k_to_134m(max_mb: float):
    """The 12 tile-aligned square-ish bf16 shapes of `b2x-op-cost-curve`, doubling each step."""
    out, tiles = [], 1
    while True:
        rows, cols = 1, tiles
        while cols > 2 * rows:
            rows *= 2
            cols = max(1, tiles // rows)
        mb = rows * 32 * cols * 32 * 2 / 1e6
        if mb > max_mb:
            return out
        if mb >= 0.06:
            out.append((rows * 32, cols * 32, mb))
        tiles *= 2


def out_kwarg(op, args, out):
    """ttnn spells the preallocated-output parameter differently per op family."""
    for k in ("output_tensor", "optional_output_tensor", "memory_config_or_output"):
        try:
            op(*args, **{k: out})
            return k
        except TypeError:
            continue
    return None


def replay_cost(ttnn, dev, issue, R, n_med=5):
    """Device seconds per op: capture R copies of one op, replay the trace, divide."""
    for _ in range(3):
        issue()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(R):
        issue()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    per = []
    for _ in range(n_med):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        per.append((time.perf_counter() - t0) / R)
    ttnn.release_trace(dev, tid)
    return st.median(per), [round(1e6 * p, 4) for p in per]


def sweep(ttnn, dev, max_mb: float):
    import torch
    for h, w, mb in sizes_64k_to_134m(max_mb):
        nbytes = h * w * 2
        rec = {"h": h, "w": w, "MB": round(mb, 4), "bytes_per_operand": nbytes}
        try:
            mk = lambda a, b: ttnn.from_torch(torch.randn(a, b), layout=ttnn.TILE_LAYOUT,
                                              dtype=ttnn.bfloat16, device=dev,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x, y, yt = mk(h, w), mk(h, w), mk(w, h)
        except Exception as e:                                              # noqa: BLE001
            rec["error"] = f"alloc {type(e).__name__}: {e}"
            yield rec
            continue

        # add: 3 x size moved, output preallocated so the trace has a stable buffer
        try:
            z = ttnn.add(x, y)
            ttnn.synchronize_device(dev)
            R = int(min(400, max(4, round(0.04 / max(3 * nbytes / 400e9, 1e-6)))))
            t, all_us = replay_cost(ttnn, dev, lambda: ttnn.add(x, y, output_tensor=z), R)
            rec["add"] = {"R": R, "us_per_op": round(1e6 * t, 4), "all_us": all_us,
                          "moved_MB": round(3 * nbytes / 1e6, 4),
                          "GBs": round(3 * nbytes / t / 1e9, 2)}
            ttnn.deallocate(z)
        except Exception as e:                                              # noqa: BLE001
            rec["add"] = {"error": f"{type(e).__name__}: {e}"}
            ttnn.synchronize_device(dev)

        # matmul: [h,w] @ [w,h], 2*h*w*h FLOP, the compute roof of this part
        try:
            flop = 2 * h * w * h
            zm = ttnn.matmul(x, yt)
            ttnn.synchronize_device(dev)
            k = out_kwarg(ttnn.matmul, (x, yt), zm)
            if k is None:
                raise RuntimeError("ttnn.matmul takes no preallocated output on this build")
            ttnn.synchronize_device(dev)
            R = int(min(200, max(3, round(0.04 / max(flop / 50e12, 1e-6)))))
            t, all_us = replay_cost(ttnn, dev, lambda: ttnn.matmul(x, yt, **{k: zm}), R)
            rec["matmul"] = {"R": R, "us_per_op": round(1e6 * t, 4), "all_us": all_us,
                             "out_h": h, "out_w": h, "GFLOP": round(flop / 1e9, 4),
                             "TFLOPs": round(flop / t / 1e12, 3), "out_kwarg": k,
                             "moved_MB": round((2 * nbytes + h * h * 2) / 1e6, 4)}
            ttnn.deallocate(zm)
        except Exception as e:                                              # noqa: BLE001
            rec["matmul"] = {"error": f"{type(e).__name__}: {e}"}
            ttnn.synchronize_device(dev)

        for tns in (x, y, yt):
            ttnn.deallocate(tns)
        yield rec


def read_roofs(points):
    """t_fixed off the flat floor, BW_eff off the asymptote, plus the least-squares fit."""
    good = [p for p in points if "add" in p and "us_per_op" in p["add"]]
    r: dict = {}
    if good:
        t_min = min(p["add"]["us_per_op"] for p in good)
        flat = [p for p in good if p["add"]["us_per_op"] <= 1.05 * t_min]
        asym = max(p["add"]["GBs"] for p in good)
        knee = next((p["MB"] for p in good if p["add"]["GBs"] >= 0.9 * asym), None)
        r["add"] = {
            "t_fixed_us": t_min,
            "flat_band_points": len(flat),
            "flat_band_max_MB": max(p["MB"] for p in flat),
            "flat_band_spread_pct": round(
                100 * (max(p["add"]["us_per_op"] for p in flat) / t_min - 1), 2),
            "BW_eff_GBs": asym,
            "BW_eff_at_MB": next(p["MB"] for p in good if p["add"]["GBs"] == asym),
            "knee_MB_per_operand": knee,
            "knee_MB_moved": round(3 * knee, 4) if knee else None,
        }
        # the two-parameter fit, reported because the previous pass reported it
        for lim in (2, 8, 32):
            pts = [(p["add"]["moved_MB"] * 1e6, p["add"]["us_per_op"] * 1e-6)
                   for p in good if p["MB"] <= lim]
            if len(pts) < 3:
                continue
            n = len(pts)
            sx, sy = sum(b for b, _ in pts), sum(t for _, t in pts)
            sxx = sum(b * b for b, _ in pts)
            sxy = sum(b * t for b, t in pts)
            den = n * sxx - sx * sx
            if den == 0:
                continue
            slope = (n * sxy - sx * sy) / den
            r["add"][f"lsq_le_{lim}MB"] = {
                "t_fixed_us": round(1e6 * (sy - slope * sx) / n, 3),
                "BW_eff_GBs": round(1e-9 / slope, 2) if slope > 0 else None}
    mm = [p for p in points if "matmul" in p and "TFLOPs" in p["matmul"]]
    if mm:
        best = max(mm, key=lambda p: p["matmul"]["TFLOPs"])
        r["matmul"] = {"peak_TFLOPs": best["matmul"]["TFLOPs"],
                       "peak_at_shape": [best["h"], best["w"]],
                       "peak_at_MB": best["MB"],
                       "t_fixed_us": min(p["matmul"]["us_per_op"] for p in mm)}
    return r


def trace_floor(ttnn, dev, fn, args, kwargs, reps=16, n_med=5):
    for _ in range(2):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn(*args, **kwargs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    meds, issue = [], []
    for _ in range(n_med):
        t0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        issue.append((time.thread_time() - c0) / reps)
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    ttnn.release_trace(dev, tid)
    return {"device_ms_per_call": round(1e3 * st.median(meds), 4),
            "all_ms": [round(1e3 * m, 4) for m in meds],
            "replay_issue_cpu_us_per_call": round(1e6 * st.median(issue), 3)}


def deficit_table(roofs, floors):
    """measured / (ops * t_fixed + bytes / BW_eff), this part's own roofs."""
    a = roofs.get("add")
    if not a:
        return {"error": "no add roofs"}
    tf, bw = a["t_fixed_us"] * 1e-6, a["BW_eff_GBs"] * 1e9
    rows = {}
    for name, (calls, mb, ops) in SUBUNITS.items():
        f = floors.get(name, {})
        if "device_ms_per_call" not in f:
            continue
        meas = f["device_ms_per_call"] * calls
        fixed, byte = 1e3 * ops * tf, 1e3 * mb * 1e6 / bw
        rows[name] = {"calls_per_block": calls, "device_ms_per_call": f["device_ms_per_call"],
                      "device_ms_per_block": round(meas, 4), "MB_per_block": mb,
                      "ops_per_block": ops, "fixed_ms": round(fixed, 4),
                      "byte_ms": round(byte, 4), "model_ms": round(fixed + byte, 4),
                      "deficit": round(meas / (fixed + byte), 4)}
    f = floors.get("PairformerLayer", {})
    if "device_ms_per_call" in f:
        meas = f["device_ms_per_call"]
        fixed, byte = 1e3 * BLOCK_OPS * tf, 1e3 * BLOCK_BYTES / bw
        rows["PairformerLayer"] = {
            "calls_per_block": 1, "device_ms_per_call": meas, "device_ms_per_block": meas,
            "MB_per_block": BLOCK_BYTES / 1e6, "ops_per_block": BLOCK_OPS,
            "fixed_ms": round(fixed, 4), "byte_ms": round(byte, 4),
            "model_ms": round(fixed + byte, 4), "deficit": round(meas / (fixed + byte), 4),
            "pct_of_BW_roof": round(100 * BLOCK_BYTES / (meas * 1e-3) / bw, 2)}
    return rows


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="sweep,block")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-mb", type=float, default=256.0)
    ap.add_argument("--trace-mb", type=int, default=1024)
    ap.add_argument("--label", default="")
    ap.add_argument("--roofs-from", type=Path,
                    help="take t_fixed/BW_eff from a previous --phases sweep JSON of this part")
    a = ap.parse_args()
    OUT_PATH = a.out
    phases = [p for p in a.phases.split(",") if p]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device(trace_region_size=a.trace_mb << 20)
    OUT["env"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": a.label,
        "host": os.uname().nodename, "arch": T.arch_name(),
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": str(dev.compute_with_storage_grid_size()),
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "ttnn": __import__("importlib.metadata", fromlist=["version"]).version("ttnn"),
        "loadavg": open("/proc/loadavg").read().split()[:3],
        "size": a.size, "trace_region_MB": a.trace_mb,
    }
    OUT["bh_reference"] = BH_REFERENCE
    if a.roofs_from:
        prev = json.loads(a.roofs_from.read_text())
        OUT["roofs"] = prev["roofs"]
        OUT["roofs_from"] = {"file": str(a.roofs_from), "env": prev.get("env")}
    dump()
    print(f"=== {OUT['env']['arch']} {OUT['env']['grid']} on {OUT['env']['host']} "
          f"card {OUT['env']['card']}, commit {OUT['env']['commit']} ===", flush=True)

    if "sweep" in phases:
        print("=== size sweep, trace replay, host removed ===", flush=True)
        OUT["points"] = []
        gen = sweep(ttnn, dev, a.max_mb)
        for rec in gen:
            OUT["points"].append(rec)
            s, m = rec.get("add", {}), rec.get("matmul", {})
            print(f"  {rec['MB']:9.3f} MB/operand  add {s.get('us_per_op', -1):10.2f} us "
                  f"{s.get('GBs', -1):7.1f} GB/s   matmul {m.get('us_per_op', -1):10.2f} us "
                  f"{m.get('TFLOPs', -1):7.2f} TFLOP/s", flush=True)
            dump()
        OUT["roofs"] = read_roofs(OUT["points"])
        print(json.dumps(OUT["roofs"], indent=1), flush=True)
        dump()

    if "block" in phases:
        import tt_baseline as B
        from tt_bio.main import _resolve_recycling_steps
        B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
        B.SAMPLING_STEPS = 200
        snap = list(sys.path)
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        from fold_ab_multi import patch_boltz2_cfg
        sys.path[:] = snap
        patch_boltz2_cfg()

        fix = ROOT / "perf" / "size512" / "fixtures"
        tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
        one_fold, meta, _state = B.build_fold("boltz2", HERE / f".msa_{a.size}", tgt, a3m)
        OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
        dump()

        grabs: dict[str, dict] = {}
        counts: dict[str, int] = {}
        originals = []

        def _vol(sh):
            v = 1
            for d in sh:
                v *= int(d)
            return v

        def clone(x):
            return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

        def arm(cname, want):
            cls = getattr(T, cname)
            orig = cls.__dict__["__call__"]
            originals.append((cls, orig))

            def w(self_obj, *args, **kw):
                counts[cname] = counts.get(cname, 0) + 1
                if cname not in grabs and counts[cname] >= 3 and want(self_obj, args):
                    grabs[cname] = {"obj": self_obj,
                                    "args": tuple(clone(x) for x in args),
                                    "kwargs": {k: clone(v) for k, v in kw.items()}}
                    print(f"  grabbed {cname} on call {counts[cname]}", flush=True)
                return orig(self_obj, *args, **kw)
            cls.__call__ = w

        print("=== cold fold ===", flush=True)
        t, _m = one_fold()
        OUT["cold_s"] = round(t, 3)
        dump()
        targets = ["PairformerLayer"] + list(SUBUNITS)
        for cname in targets:
            if cname == "PairformerLayer":
                arm(cname, lambda o, ar: getattr(o, "transform_s", False))
            elif cname == "Transition":
                # the pair-track one: z is 512x512x128 at 512 aa, every other Transition is smaller
                arm(cname, lambda o, ar: bool(ar) and hasattr(ar[0], "shape")
                    and _vol(ar[0].shape) > 4_000_000)
            else:
                arm(cname, lambda o, ar: True)
        print("=== fold 2, grabbing ===", flush=True)
        t, m = one_fold()
        for cls, orig in originals:
            cls.__call__ = orig
        OUT["fold2_s"] = round(t, 3)
        OUT["plddt"] = round(float(m.get("plddt", -1)), 6)
        OUT["class_call_counts_per_fold"] = counts
        cifs = sorted(Path(meta["struct_dir"]).glob("*.cif"))
        OUT["cif"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cifs}
        print(f"  fold2 {t:.3f} s  plddt {OUT['plddt']}  cif {OUT['cif']}", flush=True)
        dump()

        print("=== device floors by trace replay ===", flush=True)
        OUT["floors"] = {}
        for cname in targets:
            if cname not in grabs:
                OUT["floors"][cname] = {"error": "not grabbed"}
                continue
            g = grabs[cname]
            try:
                r = trace_floor(ttnn, dev, g["obj"], g["args"], g["kwargs"])
                OUT["floors"][cname] = r
                print(f"  {cname:24s} {r['device_ms_per_call']:9.4f} ms/call "
                      f"(replay issue {r['replay_issue_cpu_us_per_call']:.2f} us)", flush=True)
            except Exception as e:                                          # noqa: BLE001
                OUT["floors"][cname] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {cname} FAILED {OUT['floors'][cname]['error']}", flush=True)
            dump()

    if OUT.get("roofs") and OUT.get("floors"):
        OUT["deficit"] = deficit_table(OUT["roofs"], OUT["floors"])
        print(json.dumps(OUT["deficit"], indent=1), flush=True)
    dump()
    print("DONE " + str(a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
