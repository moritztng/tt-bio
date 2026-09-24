#!/usr/bin/env python3
"""bcx-realcensus: is the fixed trunk backward device time or host time, and which ops hold it?

`bcx-afgrad` and `bcx-ckpt` read process CPU equal to wall for the n=256 backward and called it
host-bound. ttnn's completion wait spins, so a device-bound backward reads the same. This harness
takes the discriminator instead: device kernel time from the device profiler, against wall.

One real AF2 block, `stack` arm, checkpointed exactly as `stack.py whole` runs the trunk, so a
block's "bwd" is the checkpoint recompute forward plus the inner backward. All subcommands reuse
`perf/bcx_stack/stack.py` (`Levers`, `block_step`, `inputs`, `Clock`) unchanged.

  wall   unprofiled fwd and bwd per block at K = 1, 2, 4 blocks per step, AICLK sampled inside
         every window and loadavg per point. Run it on the shipped wheel and on the Tracy build.
  prof   the same steps with a tracy signpost around every phase, then the per-class roofs
         (matmul at each fidelity, add, clone, permute) in the same process. Run it under
         `python -m tracy`; wall under the profiler is recorded but is not the wall.
  table  parse the ops report `prof` produced: device sum and busy span per phase, the per-op
         table of the backward sorted by device time, each op against its own class roof.
  host   where the host time of one block's backward goes: per tape-node kind (the closure's
         qualname, timed unsynced around every `node.fn`), the group fires, and a cProfile of
         the walk, which prices `buffer_address` and the exceptions it throws.
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import csv
import glob
import io
import json
import os
import pathlib
import pstats
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack import stack as S  # noqa: E402

OUT = ROOT / "perf" / "bcx_realcensus"

try:
    from tracy import signpost  # present on the Tracy source build only
except Exception:                                   # the shipped wheel
    def signpost(header, message=None):
        pass


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


def build():
    import ttnn
    return {"ttnn": ttnn.__file__, "profiler_env": os.environ.get("TT_METAL_DEVICE_PROFILER")}


def points(args):
    return [(s, int(k)) for s in args.stacks.split(",") for k in args.ks.split(",")]


# ------------------------------------------------------------------------------ wall


def cmd_wall(args):
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    blob = {"stamp": S.stamp(args, clock), "build": build(), "arm": args.arm, "n": args.n,
            "ckpt": True, "points": []}
    for stack_name, k in points(args):
        rows = []
        for step in range(args.warm + args.steps):
            r, _ = S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=True)
            r["load1"] = os.getloadavg()[0]
            if step >= args.warm:
                rows.append(r)
        lv.take()
        pt = {"stack": stack_name, "K": k,
              "fwd": S.dist([r["fwd"] for r in rows]), "bwd": S.dist([r["bwd"] for r in rows]),
              "bwd_cpu": S.dist([r["bwd_cpu"] for r in rows]),
              "load1": [r["load1"] for r in rows],
              "aiclk": clock.window([s for r in rows for s in r["spans"]])}
        blob["points"].append(pt)
        print(json.dumps({"stack": stack_name, "K": k, "fwd": round(pt["fwd"]["median"], 4),
                          "bwd": round(pt["bwd"]["median"], 4),
                          "bwd_p10_p90": [round(pt["bwd"]["p10"], 4), round(pt["bwd"]["p90"], 4)],
                          "bwd_cpu": round(pt["bwd_cpu"]["median"], 4), "aiclk": pt["aiclk"],
                          "load1_max": max(pt["load1"])}), flush=True)
        save(args.out or "wall.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ prof


def roofs(dev, reps):
    """The per-class roofs, each op signposted so `table` reads its device time."""
    import torch
    ttnn = dev.ttnn
    d = dev.device

    def up(shape, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.randn(shape).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                               device=d, dtype=dtype)

    fid = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2,
           "LoFi": ttnn.MathFidelity.LoFi}
    a, b = up([4096, 4096]), up([4096, 4096])
    for name, f in fid.items():
        for acc in (True, False):
            cfg = ttnn.init_device_compute_kernel_config(d.arch(), math_fidelity=f,
                                                         fp32_dest_acc_en=acc, packer_l1_acc=True)
            for i in range(reps + 1):
                signpost(f"roof matmul4096 {name} fp32acc={acc} {i}")
                ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=cfg))
    x, y = up([8192, 8192]), up([8192, 8192])
    p = up([256, 256, 128])
    for i in range(reps + 1):
        signpost(f"roof add8192 {i}")
        ttnn.deallocate(ttnn.add(x, y))
        signpost(f"roof clone8192 {i}")
        ttnn.deallocate(ttnn.clone(x))
        signpost(f"roof permute256x256x128 {i}")
        ttnn.deallocate(ttnn.permute(p, [1, 0, 2]))
    dev.sync()
    signpost("roof end")


def cmd_prof(args):
    import ttnn
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    blob = {"stamp": S.stamp(args, clock), "build": build(), "arm": args.arm, "n": args.n,
            "note": "wall under the device profiler: perturbed, attribution only", "points": []}
    for stack_name, k in points(args):
        for _ in range(args.warm):
            S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=True)
    ttnn.ReadDeviceProfiler(dev.device)               # drain the warm-up out of the buffers

    ag = dev.ag
    for stack_name, k in points(args):
        for rep in range(args.steps):
            tag = f"{stack_name} K={k} rep={rep}"
            # block_step, inlined only to put a signpost at each phase boundary
            ke, kv = (k, 0) if stack_name == "extra" else (0, k)
            ml, zl = dev.leaf(m0), dev.leaf(z0)
            dev.sync()
            signpost(f"fwd {tag}")
            t0 = time.time()
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
            dev.sync()
            t1 = time.time()
            roots = [zo] if stack_name == "extra" else [mo, zo]
            seeds = [dev.seed(wz, zo)] if stack_name == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
            dev.sync()
            signpost(f"bwd {tag}")
            t2 = time.time()
            ag.backward(roots, seeds)
            dev.sync()
            t3 = time.time()
            signpost(f"end {tag}")
            ag.release_pins()
            del mo, zo, ml, zl, roots, seeds
            blob["points"].append({"tag": tag, "fwd": t1 - t0, "bwd": t3 - t2,
                                   "aiclk": clock.window([(t0, t3)]), "load1": os.getloadavg()[0]})
            print(json.dumps(blob["points"][-1]), flush=True)
            ttnn.ReadDeviceProfiler(dev.device)
    t0 = time.time()
    roofs(dev, args.roof_reps)
    blob["roof_aiclk"] = clock.window([(t0, time.time())])
    ttnn.ReadDeviceProfiler(dev.device)
    clock.stop()
    save(args.out or "prof_host.json", blob)


# ------------------------------------------------------------------------------ table


def _dims(row, pre):
    """Padded [W,Z,Y,X] of one operand from the ops report's `padded[logical]` columns."""
    out = []
    for ax in "WZYX":
        v = row.get(f"{pre}_{ax}_PAD[LOGICAL]") or row.get(f"{pre}_{ax}") or ""
        v = v.split("[")[0].strip()
        if not v:
            return None
        out.append(int(v))
    return out


BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "UINT32": 4,
         "INT32": 4, "UINT16": 2, "UINT8": 1}


def _nbytes(row, pre):
    d = _dims(row, pre)
    if d is None:
        return 0.0
    dt = (row.get(f"{pre}_DATATYPE") or "BFLOAT16").split(".")[-1].upper()
    return float(np.prod(d)) * BYTES.get(dt, 2)


def _mem(row, pre):
    m = (row.get(f"{pre}_MEMORY") or "").upper()
    return "L1" if "L1" in m else ("DRAM" if "DRAM" in m else "?")


CLASS = [
    ("matmul", ("Matmul", "Linear", "MinimalMatmul")),
    ("softmax", ("Softmax",)),
    ("layernorm", ("LayerNorm", "RMSNorm")),
    ("reduction", ("Reduce", "Moreh", "Sum", "Mean")),
    ("layout", ("Permute", "Transpose", "Reshape", "Concat", "Slice", "Pad", "Tilize", "Untilize",
                "Copy", "Clone", "Typecast", "Nlp", "Interleaved", "Sharded", "Fold", "Repeat",
                "Split", "Fill", "Reallocate")),
    ("eltwise", ("Binary", "Unary", "Eltwise", "Where", "Ternary")),
]


def op_class(code):
    for cls, keys in CLASS:
        if any(k.lower() in code.lower() for k in keys):
            return cls
    return "other"


def _read_report(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cmd_table(args):
    rep = args.report or sorted(glob.glob(f"{args.profdir}/reports/*/ops_perf_results_*.csv"))[-1]
    rows = _read_report(rep)
    host = json.loads((OUT / (args.host or "prof_host.json")).read_text())

    # split rows into signposted segments
    seg, segs = None, collections.defaultdict(list)
    for r in rows:
        if (r.get("OP TYPE") or "").lower() == "signpost":
            seg = r["OP CODE"]
            continue
        if seg is not None and r.get("DEVICE KERNEL DURATION [ns]"):
            segs[seg].append(r)

    def dur(r):
        return float(r["DEVICE KERNEL DURATION [ns]"]) * 1e-9

    def span(rs):
        st = [float(r["DEVICE FW START CYCLE"]) for r in rs if r.get("DEVICE FW START CYCLE")]
        en = [float(r["DEVICE FW END CYCLE"]) for r in rs if r.get("DEVICE FW END CYCLE")]
        return (min(st), max(en)) if st and en else (None, None)

    # roofs, from the same process
    roof = {}
    for name, rs in segs.items():
        if not name.startswith("roof "):
            continue
        key, i = name[5:].rsplit(" ", 1)
        if int(i) == 0 or not rs:                                   # rep 0 is the warm-up
            continue
        roof.setdefault(key, []).append(sum(dur(r) for r in rs))
    R = {}
    for key, ts in roof.items():
        t = float(np.median(ts))
        if key.startswith("matmul4096"):
            R[key] = {"s": t, "tflops": 2 * 4096 ** 3 / t / 1e12}
        elif key.startswith("add"):
            R[key] = {"s": t, "gbs": 3 * 8192 * 8192 * 2 / t / 1e9}
        elif key.startswith("clone"):
            R[key] = {"s": t, "gbs": 2 * 8192 * 8192 * 2 / t / 1e9}
        elif key.startswith("permute"):
            R[key] = {"s": t, "gbs": 2 * 256 * 256 * 128 * 2 / t / 1e9}
    dram = max(R["add8192"]["gbs"], R["clone8192"]["gbs"])
    copy = R["clone8192"]["gbs"]

    def mm_roof(fid, acc):
        k = f"matmul4096 {fid} fp32acc={acc}"
        return R[k]["tflops"] if k in R else R["matmul4096 HiFi4 fp32acc=True"]["tflops"]

    def dram_bytes(r):
        """DRAM traffic of one op: every DRAM-resident operand once, L1 operands excluded. A slice
        reads only what it writes, so its input counts at the output's size."""
        by = 0.0
        for pre in ("INPUT_0", "INPUT_1", "INPUT_2", "OUTPUT_0"):
            if _mem(r, pre) != "DRAM":
                continue
            if pre == "INPUT_0" and "Slice" in r["OP CODE"]:
                by += _nbytes(r, "OUTPUT_0")
            else:
                by += _nbytes(r, pre)
        return by

    def roof_s(r, cls):
        """Seconds this op would take at its own class roof, and which roof that is."""
        by = dram_bytes(r)
        attr = (r.get("ATTRIBUTES") or "").replace(" ", "")
        if cls == "matmul":
            a, o = _dims(r, "INPUT_0"), _dims(r, "OUTPUT_0")
            if not (a and o):
                return None, "?"
            flops = 2.0 * np.prod(o) * a[-1]
            fid = (r.get("MATH FIDELITY") or "HiFi4").strip()
            acc = "fp32_dest_acc_en=1" in attr or "fp32_dest_acc_en=true" in attr.lower()
            tf = mm_roof(fid, acc)
            tc, tb = flops / (tf * 1e12), by / (dram * 1e9)
            return max(tc, tb), (f"mm {fid}{' fp32acc' if acc else ''} {tf:.0f}TF" if tc >= tb
                                 else f"dram {dram:.0f}GB/s")
        if by == 0:
            return None, "L1-only"
        bw = copy if cls == "layout" else dram
        return by / (bw * 1e9), f"{'copy' if cls == 'layout' else 'dram'} {bw:.0f}GB/s"

    phases = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, rs in segs.items():
        if name.startswith(("fwd ", "bwd ")):
            ph, tag = name.split(" ", 1)
            phases[tag][ph] = rs
    hp = {p["tag"]: p for p in host["points"]}
    summary = []
    for tag, d in sorted(phases.items()):
        s = {"tag": tag}
        for ph in ("fwd", "bwd"):
            rs = d.get(ph, [])
            a, b = span(rs)
            s[ph] = {"ops": len(rs), "device_s": sum(dur(r) for r in rs),
                     "span_s": (b - a) / (args.aiclk * 1e6) if a is not None else None,
                     "wall_profiled_s": hp.get(tag, {}).get(ph)}
        summary.append(s)
        print(json.dumps(s), flush=True)

    # per-op table of the backward: one block = the chosen tag's bwd segment
    table = {}
    for tag in args.table_tags.split(","):
        rs = phases[tag]["bwd"]
        g = collections.defaultdict(lambda: {"n": 0, "s": 0.0, "roof_s": 0.0, "roofless": 0.0, "roof": None,
                                             "cores": set(), "mem": set()})
        for r in rs:
            code = r["OP CODE"]
            cls = op_class(code)
            shp = f"{_dims(r, 'INPUT_0')}" + (f"x{_dims(r, 'INPUT_1')}" if _dims(r, "INPUT_1") else "")
            key = (code, cls, shp, (r.get("MATH FIDELITY") or "").strip() if cls == "matmul" else "")
            e = g[key]
            e["n"] += 1
            e["s"] += dur(r)
            rs_, rn = roof_s(r, cls)
            e["roof_s"] += rs_ or 0.0
            e["roofless"] += 0 if rs_ else dur(r)
            e["roof"] = rn
            e["cores"].add(r.get("CORE COUNT"))
            e["mem"].add(f"{_mem(r, 'INPUT_0')}->{_mem(r, 'OUTPUT_0')}")
        tot = sum(e["s"] for e in g.values())
        out, cum = [], 0.0
        for key, e in sorted(g.items(), key=lambda kv: -kv[1]["s"]):
            cum += e["s"]
            out.append({"op": key[0], "class": key[1], "shape": key[2], "fidelity": key[3],
                        "n": e["n"], "ms": e["s"] * 1e3, "share": e["s"] / tot, "cum": cum / tot,
                        "util": e["roof_s"] / (e["s"] - e["roofless"]) if e["s"] > e["roofless"] else None, "roof": e["roof"],
                        "roof_ms": e["roof_s"] * 1e3, "roofless_ms": e["roofless"] * 1e3,
                        "cores": sorted(c for c in e["cores"] if c), "mem": sorted(e["mem"])})
        by_class = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0])
        for e in out:
            c = by_class[e["class"]]
            c[0] += e["ms"]
            c[1] += e["roof_ms"]
            c[2] += e["n"]
            c[3] += e["roofless_ms"]
        table[tag] = {"device_ms": tot * 1e3, "ops": len(rs), "rows": out,
                      "by_class": {k: {"ms": v[0], "share": v[0] / (tot * 1e3), "n": v[2],
                                       "roof_ms": v[1], "util_at_roof": v[1] / (v[0] - v[3]) if v[0] > v[3] else None}
                                   for k, v in sorted(by_class.items(), key=lambda kv: -kv[1][0])}}
    save(args.out or "table.json", {"report": rep, "roofs": R, "dram_gbs": dram, "copy_gbs": copy,
                                    "aiclk_mhz_for_cycles": args.aiclk, "phases": summary,
                                    "table": table})


# ------------------------------------------------------------------------------ host


def cmd_host(args):
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    ag = dev.ag
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    kinds = collections.defaultdict(lambda: [0, 0.0])
    on = [False]
    real_init = ag._Node.__init__

    def init(self, fn, parents, group=None):
        name = getattr(fn, "__qualname__", type(fn).__name__)
        mod = getattr(fn, "__module__", "") or ""

        def timed(g, _fn=fn, _name=f"{mod.split('.')[-1]}:{name}"):
            if not on[0]:
                return _fn(g)
            t0 = time.perf_counter()
            try:
                return _fn(g)
            finally:
                e = kinds[_name]
                e[0] += 1
                e[1] += time.perf_counter() - t0
        real_init(self, timed, parents, group)

    ag._Node.__init__ = init
    blob = {"stamp": S.stamp(args, clock), "build": build(), "arm": args.arm, "n": args.n,
            "note": "node times are host wall around each closure, UNSYNCED, nested closures "
                    "(a checkpoint group's inner walk) are charged to both levels", "points": []}
    for stack_name, k in points(args):
        for _ in range(args.warm):
            S.block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=True)
        for mode in ("nodes", "cprofile"):
            kinds.clear()
            ke, kv = (k, 0) if stack_name == "extra" else (0, k)
            ml, zl = dev.leaf(m0), dev.leaf(z0)
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
            roots = [zo] if stack_name == "extra" else [mo, zo]
            seeds = [dev.seed(wz, zo)] if stack_name == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
            dev.sync()
            on[0] = mode == "nodes"
            pr = cProfile.Profile() if mode == "cprofile" else None
            t0, c0 = time.time(), time.process_time()
            if pr:
                pr.enable()
            ag.backward(roots, seeds)
            if pr:
                pr.disable()
            t_enq = time.time() - t0
            dev.sync()
            t1, c1 = time.time(), time.process_time()
            on[0] = False
            ag.release_pins()
            del mo, zo, ml, zl, roots, seeds
            pt = {"stack": stack_name, "K": k, "mode": mode, "bwd": t1 - t0, "bwd_cpu": c1 - c0,
                  "host_returned_s": t_enq, "load1": os.getloadavg()[0],
                  "aiclk": clock.window([(t0, t1)])}
            if mode == "nodes":
                pt["kinds"] = sorted(([k_, n_, s_] for k_, (n_, s_) in kinds.items()),
                                     key=lambda x: -x[2])
            else:
                st = pstats.Stats(pr)
                top = []
                for (fn, ln, name), (cc, nc, tt, ct, _) in st.stats.items():
                    top.append([f"{pathlib.Path(fn).name}:{ln}:{name}", nc, tt, ct])
                pt["tottime_top"] = sorted(top, key=lambda x: -x[2])[:60]
                pt["cumtime_top"] = sorted(top, key=lambda x: -x[3])[:60]
                pt["profiled_total"] = sum(x[2] for x in top)
            blob["points"].append(pt)
            print(json.dumps({k_: v for k_, v in pt.items()
                              if k_ not in ("kinds", "tottime_top", "cumtime_top")}), flush=True)
    clock.stop()
    save(args.out or "host.json", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wall", "prof", "table", "host"])
    ap.add_argument("--params", default=S.A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arm", default="stack")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roof-reps", type=int, default=3)
    ap.add_argument("--profdir", default="/dev/shm/bcx-realcensus-prof")
    ap.add_argument("--report", default=None)
    ap.add_argument("--host", default=None, help="table: prof's host json")
    ap.add_argument("--aiclk", type=float, default=1350.0, help="table: MHz to convert cycles")
    ap.add_argument("--table-tags", default="evo K=1 rep=0,extra K=1 rep=0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    {"wall": cmd_wall, "prof": cmd_prof, "table": cmd_table, "host": cmd_host}[args.cmd](args)


if __name__ == "__main__":
    main()
