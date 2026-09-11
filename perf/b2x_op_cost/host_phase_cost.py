#!/usr/bin/env python3
"""Where the Boltz-2 512 aa fold's 21.8 s of main-thread CPU sits, phase by phase.

Two measurements in this campaign look like they contradict each other:

  * `state/bioir-dispatch-graph/` -- the eager sampler is 88.3 % device-busy in steady state
    (32.526 ms of replayed device work in a 36.84 ms step) and puts a 1.112x ceiling on any
    dispatch lever.
  * `b2x-baseline-attrib` -- one fold issues 487 202 ttnn calls and burns 21.846 s of
    main-thread CPU in a 26.037 s fold, 83.9 %.

They are not the same quantity: one is device kernel time over wall, the other host CPU over
wall, and in a pipelined dispatcher both can sit near 1 at once. What decides which side is on
the critical path is whether the host thread ever BLOCKS. If the device were the constraint the
command queue would fill, the host would wait in the kernel, and `time.thread_time()` -- which
counts only CPU the calling thread actually burns -- would fall well below the wall.

So this measures, in one process on one device open:

  table    one fold with every `tt_bio.tenstorrent` module bracketed for wall AND main-thread
           CPU, device synced only at the top of the tree, so a phase's wall is comparable with
           a plain fold's and its CPU is the host cost of that subtree. A second fold adds a
           nest-guarded per-call census over every ttnn entry point, giving calls, wall and CPU
           per (phase, op). Phase names are the class names, the same convention
           `b2x-baseline-attrib` used, so the columns line up with its wall/GB table.
  slope    the causal check. Inject a measured amount of extra host CPU per ttnn call and read
           d(phase wall)/d(phase host CPU). Slope 1 means the host is on that phase's critical
           path and removing host time gives wall time back; slope 0 means the added work hides
           under the device and a dispatch lever is worth nothing there.
  control  the negative control the CPU-vs-wall discriminator needs: a loop that is definitely
           device-bound (big matmuls) and one that is definitely host-bound (one-tile adds). If
           the device-bound loop reads CPU ~= wall, ttnn spins on a full queue on the calling
           thread, `thread_time` is not a discriminator, and the table above means nothing.
  block    the device floor for one real pairformer block: capture a ttnn trace of one shipped
           PairformerLayer call and replay it back-to-back with no host work between replays.
           Same method the sibling used for the denoiser. Turns "reachable <= host CPU" into a
           number.

No model code is changed. The block leg captures and replays shipped `PairformerLayer.__call__`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


# ---------------------------------------------------------------------------------------------
# brackets: wall + main-thread CPU per module, synced only near the top of the tree
# ---------------------------------------------------------------------------------------------
class PhaseBrackets:
    def __init__(self, dev, ttnn, sync_depth=1):
        self.dev, self.ttnn, self.sync_depth = dev, ttnn, sync_depth
        self.stack: list[str] = []
        self.wall: dict[str, list[float]] = defaultdict(list)
        self.cpu: dict[str, float] = defaultdict(float)
        self.child_wall: dict[str, float] = defaultdict(float)
        self.child_cpu: dict[str, float] = defaultdict(float)
        self._orig: list = []

    def call(self, name, fn, self_obj, a, k):
        path = "/".join(self.stack + [name])
        self.stack.append(name)
        if len(self.stack) <= self.sync_depth:
            self.ttnn.synchronize_device(self.dev)
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            return fn(self_obj, *a, **k) if self_obj is not None else fn(*a, **k)
        finally:
            if len(self.stack) <= self.sync_depth:
                self.ttnn.synchronize_device(self.dev)
            dt, dc = time.perf_counter() - t0, time.thread_time() - c0
            self.stack.pop()
            self.wall[path].append(dt)
            self.cpu[path] += dc
            if self.stack:
                parent = "/".join(self.stack)
                self.child_wall[parent] += dt
                self.child_cpu[parent] += dc

    def install(self, T):
        names = []
        for cname, obj in sorted(vars(T).items()):
            if not isinstance(obj, type):
                continue
            attr = None
            if issubclass(obj, T.Module) and "__call__" in obj.__dict__:
                attr = "__call__"
            elif issubclass(obj, T.TorchWrapper) and "forward" in obj.__dict__:
                attr = "forward"
            if attr is None:
                continue
            orig = obj.__dict__[attr]
            self._orig.append((obj, attr, orig))

            def mk(name, fn):
                def w(self_obj, *a, **k):
                    return BR.call(name, fn, self_obj, a, k)
                return w
            setattr(obj, attr, mk(cname, orig))
            names.append(cname)
        return names

    def remove(self):
        for obj, attr, orig in self._orig:
            setattr(obj, attr, orig)
        self._orig = []

    def tree(self):
        rows = {}
        for path, ts in sorted(self.wall.items()):
            rows[path] = {
                "calls": len(ts),
                "incl_wall_s": round(sum(ts), 5),
                "excl_wall_s": round(sum(ts) - self.child_wall.get(path, 0.0), 5),
                "incl_cpu_s": round(self.cpu[path], 5),
                "excl_cpu_s": round(self.cpu[path] - self.child_cpu.get(path, 0.0), 5),
                "median_ms": round(1e3 * st.median(ts), 5),
            }
        return rows


BR: PhaseBrackets | None = None


# ---------------------------------------------------------------------------------------------
# per-call census: calls, wall and main-thread CPU per (bracket path, ttnn op)
# ---------------------------------------------------------------------------------------------
SPIN_US = 0.0


def _spin(us):
    end = time.perf_counter() + us * 1e-6
    while time.perf_counter() < end:
        pass


class CallCensus:
    """Nest-guarded so one logical ttnn call is charged once, matching the 487 202 denominator.

    `synchronize_device` is excluded: the brackets above issue those, and charging a drain to a
    phase would report device wait as host cost, which is the exact error this file exists to
    avoid.
    """
    SKIP = {"synchronize_device", "graph", "get_device", "open_device", "close_device",
            "open_mesh_device", "close_mesh_device", "GetDefaultDevice", "SetDefaultDevice",
            "Tensor", "begin_trace_capture", "end_trace_capture", "execute_trace",
            "release_trace"}

    def __init__(self, ttnn, spin=False):
        self.ttnn = ttnn
        self.spin = spin
        self.rows: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"n": 0, "wall": 0.0, "cpu": 0.0})
        self.depth = 0
        self._orig = {}
        self.spin_cpu = 0.0

    def _install_ns(self, ns, prefix):
        n = 0
        for name in dir(ns):
            if name.startswith("_") or name in self.SKIP:
                continue
            obj = getattr(ns, name, None)
            if not callable(obj) or isinstance(obj, type):
                continue

            def mk(nm, f):
                def w(*a, **k):
                    if CEN.depth:
                        return f(*a, **k)
                    CEN.depth = 1
                    path = "/".join(BR.stack) if BR and BR.stack else "(glue)"
                    t0, c0 = time.perf_counter(), time.thread_time()
                    try:
                        return f(*a, **k)
                    finally:
                        if CEN.spin and SPIN_US:
                            _spin(SPIN_US)
                        r = CEN.rows[(path, nm)]
                        r["n"] += 1
                        r["wall"] += time.perf_counter() - t0
                        r["cpu"] += time.thread_time() - c0
                        CEN.depth = 0
                return w
            try:
                setattr(ns, name, mk(prefix + name, obj))
                self._orig[(ns, name)] = obj
                n += 1
            except Exception:
                pass
        return n

    def install(self):
        n = self._install_ns(self.ttnn, "ttnn.")
        for sub in ("experimental", "transformer", "operations"):
            ns = getattr(self.ttnn, sub, None)
            if ns is not None:
                n += self._install_ns(ns, f"ttnn.{sub}.")
        return n

    def remove(self):
        for (ns, name), obj in self._orig.items():
            setattr(ns, name, obj)
        self._orig = {}

    def totals(self):
        n = sum(r["n"] for r in self.rows.values())
        return {"n_calls": n,
                "wall_in_ttnn_s": round(sum(r["wall"] for r in self.rows.values()), 4),
                "cpu_in_ttnn_s": round(sum(r["cpu"] for r in self.rows.values()), 4),
                "us_per_call_cpu": round(1e6 * sum(r["cpu"] for r in self.rows.values())
                                         / max(n, 1), 3)}

    def by_phase(self):
        agg = defaultdict(lambda: {"n": 0, "wall": 0.0, "cpu": 0.0})
        for (p, _op), r in self.rows.items():
            for lvl in range(1, p.count("/") + 2):
                key = "/".join(p.split("/")[:lvl])
                d = agg[key]
                d["n"] += r["n"]
                d["wall"] += r["wall"]
                d["cpu"] += r["cpu"]
        return {k: {"calls": v["n"], "cpu_s": round(v["cpu"], 4),
                    "wall_in_ttnn_s": round(v["wall"], 4),
                    "us_per_call_cpu": round(1e6 * v["cpu"] / max(v["n"], 1), 3)}
                for k, v in sorted(agg.items())}

    def by_op(self, top=40):
        agg = defaultdict(lambda: {"n": 0, "cpu": 0.0})
        for (_p, op), r in self.rows.items():
            d = agg[op]
            d["n"] += r["n"]
            d["cpu"] += r["cpu"]
        rows = sorted(agg.items(), key=lambda kv: -kv[1]["cpu"])[:top]
        return [{"op": o, "n": v["n"], "cpu_s": round(v["cpu"], 4),
                 "us_per_call": round(1e6 * v["cpu"] / max(v["n"], 1), 3)} for o, v in rows]


CEN: CallCensus | None = None


# ---------------------------------------------------------------------------------------------
def controls(ttnn, dev):
    """Does main-thread CPU actually fall when the device is the constraint?"""
    import torch
    res = {}
    for tag, n_el, reps in (("device_bound_4096_matmul", 4096, 400),
                            ("device_bound_2048_matmul", 2048, 800),
                            ("host_bound_1tile_add", 32, 20000)):
        a = ttnn.from_torch(torch.randn(n_el, n_el), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev)
        b = ttnn.from_torch(torch.randn(n_el, n_el), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev)
        op = ttnn.matmul if n_el >= 512 else ttnn.add
        for _ in range(5):                                  # compile + warm
            op(a, b)
        ttnn.synchronize_device(dev)
        t0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(reps):
            op(a, b)
        t_issue, c_issue = time.perf_counter() - t0, time.thread_time() - c0
        ttnn.synchronize_device(dev)
        t_total = time.perf_counter() - t0
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        res[tag] = {
            "reps": reps, "issue_wall_s": round(t_issue, 4), "issue_cpu_s": round(c_issue, 4),
            "total_wall_s": round(t_total, 4),
            "cpu_over_issue_wall": round(c_issue / t_issue, 4),
            "cpu_over_total_wall": round(c_issue / t_total, 4),
            "drain_s": round(t_total - t_issue, 4),
            "us_per_call_cpu": round(1e6 * c_issue / reps, 3),
            "us_per_call_device": round(1e6 * t_total / reps, 3),
        }
        print(f"  {tag:28s} cpu/wall {res[tag]['cpu_over_total_wall']:.3f}  "
              f"host {res[tag]['us_per_call_cpu']:8.2f} us/call  "
              f"device {res[tag]['us_per_call_device']:8.2f} us/call  "
              f"drain {res[tag]['drain_s']:.3f} s", flush=True)
    return res


def block_device_floor(ttnn, T, grabbed, reps=20):
    """Trace-capture one shipped PairformerLayer call and replay it with no host in between."""
    layer, args, kwargs = grabbed["layer"], grabbed["args"], grabbed["kwargs"]
    out = {"note": "trace capture of one shipped PairformerLayer.__call__, replayed back-to-back"}
    dev = T.get_device()
    # host cost of the eager call, and its synced wall, for reference
    for _ in range(2):
        layer(*args, **kwargs)
    ttnn.synchronize_device(dev)
    walls, cpus = [], []
    for _ in range(5):
        t0, c0 = time.perf_counter(), time.thread_time()
        layer(*args, **kwargs)
        c = time.thread_time() - c0
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
        cpus.append(c)
    out["eager_synced_ms"] = round(1e3 * st.median(walls), 4)
    out["eager_host_cpu_ms"] = round(1e3 * st.median(cpus), 4)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    layer(*args, **kwargs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    meds = []
    for _ in range(5):
        t0, c0 = time.perf_counter(), time.thread_time()
        for _ in range(reps):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        c = time.thread_time() - c0
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
        out["replay_issue_cpu_us_per_block"] = round(1e6 * c / reps, 3)
    out["replay_device_ms_per_block"] = round(1e3 * st.median(meds), 4)
    out["replay_all_ms"] = [round(1e3 * m, 4) for m in meds]
    ttnn.release_trace(dev, tid)
    return out


# ---------------------------------------------------------------------------------------------
def main() -> int:
    global OUT_PATH, BR, CEN, SPIN_US
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="plain,table,census,slope,control,block")
    ap.add_argument("--spins", default="0,10,25")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--trace-region", action="store_true",
                    help="reserve 1 GiB so the block leg can capture a trace")
    a = ap.parse_args()
    OUT_PATH = a.out
    phases = [p for p in a.phases.split(",") if p]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    if a.trace_region:
        T.get_device(trace_region_size=1 << 30)       # must precede build_fold's own open

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    msa_dir = HERE / f".msa_{a.size}"

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "trace_region": bool(a.trace_region),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": loadavg(), "size": a.size, "phases": phases}
    dump()
    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa", "card_type")
                       if k in meta})
    dev = T.get_device()
    dump()

    print("=== cold fold (discarded) ===", flush=True)
    t, m = one_fold()
    OUT["cold_s"] = round(t, 3)
    dump()

    def plain(tag):
        t0 = time.perf_counter()
        c0 = time.thread_time()
        _t, m = one_fold()
        rec = {"wall_s": round(time.perf_counter() - t0, 4),
               "cpu_s": round(time.thread_time() - c0, 4),
               "plddt": m.get("plddt"), "loadavg": loadavg()}
        rec["cpu_pct"] = round(100 * rec["cpu_s"] / rec["wall_s"], 2)
        OUT.setdefault("plain", {})[tag] = rec
        print(f"  plain[{tag}] wall {rec['wall_s']:.3f} s  cpu {rec['cpu_s']:.3f} s "
              f"({rec['cpu_pct']:.1f} %)", flush=True)
        dump()
        return rec

    if "plain" in phases:
        print("=== plain folds (wall reference + A/A floor) ===", flush=True)
        plain("a0")
        plain("a1")

    grabbed = {}
    if "table" in phases or "census" in phases:
        print("=== bracketed fold: wall + main-thread CPU per phase ===", flush=True)
        br = PhaseBrackets(dev, ttnn, sync_depth=1)
        BR = br
        classes = br.install(T)
        t0, c0 = time.perf_counter(), time.thread_time()
        _t, m = one_fold()
        wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
        br.remove()
        tree = br.tree()
        top = {p: v for p, v in tree.items() if "/" not in p}
        OUT["table"] = {
            "fold_wall_s": round(wall, 4), "fold_cpu_s": round(cpu, 4),
            "fold_cpu_pct": round(100 * cpu / wall, 2),
            "classes_bracketed": len(classes), "plddt": m.get("plddt"),
            "top_level_wall_s": round(sum(v["incl_wall_s"] for v in top.values()), 4),
            "top_level_cpu_s": round(sum(v["incl_cpu_s"] for v in top.values()), 4),
            "loadavg": loadavg(),
            "tree": tree,
        }
        OUT["table"]["residual_wall_s"] = round(wall - OUT["table"]["top_level_wall_s"], 4)
        OUT["table"]["residual_cpu_s"] = round(cpu - OUT["table"]["top_level_cpu_s"], 4)
        BR = None
        for p, v in sorted(top.items(), key=lambda kv: -kv[1]["incl_wall_s"]):
            print(f"  {p:24s} wall {v['incl_wall_s']:8.3f} s  cpu {v['incl_cpu_s']:8.3f} s "
                  f"({100*v['incl_cpu_s']/max(v['incl_wall_s'],1e-9):5.1f} %)  "
                  f"calls {v['calls']}", flush=True)
        print(f"  {'(residual)':24s} wall {OUT['table']['residual_wall_s']:8.3f} s  "
              f"cpu {OUT['table']['residual_cpu_s']:8.3f} s", flush=True)
        dump()

    if "census" in phases:
        print("=== census fold: calls, wall and CPU per (phase, op) ===", flush=True)
        br = PhaseBrackets(dev, ttnn, sync_depth=1)
        BR = br
        br.install(T)
        cen = CallCensus(ttnn)
        CEN = cen
        n_wrapped = cen.install()
        # grab one settled PairformerLayer call's arguments for the block leg
        if "block" in phases:
            orig_call = T.PairformerLayer.__dict__["__call__"]

            def grab(self_obj, *args, **kw):
                if not grabbed and getattr(self_obj, "transform_s", False):
                    grabbed.update({"layer": self_obj,
                                    "args": tuple(ttnn.clone(x) if isinstance(x, ttnn.Tensor)
                                                  else x for x in args),
                                    "kwargs": {k: (ttnn.clone(v) if isinstance(v, ttnn.Tensor)
                                                   else v) for k, v in kw.items()}})
                return orig_call(self_obj, *args, **kw)
            T.PairformerLayer.__call__ = grab
        t0, c0 = time.perf_counter(), time.thread_time()
        _t, m = one_fold()
        wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
        cen.remove()
        if "block" in phases:
            T.PairformerLayer.__call__ = orig_call
        br.remove()
        tree = br.tree()
        OUT["census"] = {
            "fold_wall_s": round(wall, 4), "fold_cpu_s": round(cpu, 4),
            "n_ttnn_entrypoints_wrapped": n_wrapped, "plddt": m.get("plddt"),
            "totals": cen.totals(), "by_phase": cen.by_phase(), "by_op": cen.by_op(),
            "tree": tree, "loadavg": loadavg(),
            "grabbed_block": bool(grabbed),
        }
        BR = CEN = None
        print("  " + json.dumps(cen.totals()), flush=True)
        dump()

    if "block" in phases and grabbed:
        print("=== block: device floor for one pairformer block by trace replay ===", flush=True)
        try:
            OUT["block"] = block_device_floor(ttnn, T, grabbed)
            print("  " + json.dumps(OUT["block"]), flush=True)
        except Exception as e:                                              # noqa: BLE001
            OUT["block"] = {"error": f"{type(e).__name__}: {e}"}
            print("  FAILED " + OUT["block"]["error"], flush=True)
        dump()
    elif "block" in phases:
        OUT["block"] = {"error": "no PairformerLayer call was grabbed (census phase not run?)"}
        dump()

    if "slope" in phases:
      try:
        print("=== slope: inject host CPU per ttnn call, read d(wall)/d(cpu) per phase ===",
              flush=True)
        rows = []
        for spin in [float(x) for x in a.spins.split(",")] + [0.0]:
            SPIN_US = spin
            br = PhaseBrackets(dev, ttnn, sync_depth=1)
            BR = br
            br.install(T)
            cen = CallCensus(ttnn, spin=True)
            CEN = cen
            cen.install()
            t0, c0 = time.perf_counter(), time.thread_time()
            _t, m = one_fold()
            wall, cpu = time.perf_counter() - t0, time.thread_time() - c0
            cen.remove()
            br.remove()
            tree = br.tree()
            top = {p: v for p, v in tree.items() if "/" not in p}
            rows.append({"spin_us": spin, "fold_wall_s": round(wall, 4),
                         "fold_cpu_s": round(cpu, 4), "plddt": m.get("plddt"),
                         "n_calls": cen.totals()["n_calls"],
                         "cpu_in_ttnn_s": cen.totals()["cpu_in_ttnn_s"],
                         "phases": {p: {"wall_s": v["incl_wall_s"], "cpu_s": v["incl_cpu_s"],
                                        "calls": v["calls"]} for p, v in top.items()},
                         "by_phase_calls": cen.by_phase(), "loadavg": loadavg()})
            BR = CEN = None
            SPIN_US = 0.0
            OUT["slope"] = rows
            dump()
            print(f"  spin {spin:5.1f} us/call -> fold wall {wall:8.3f} s  cpu {cpu:8.3f} s  "
                  f"calls {rows[-1]['n_calls']}", flush=True)
      except Exception as e:                                                # noqa: BLE001
        OUT["slope_error"] = f"{type(e).__name__}: {e}"
        BR = CEN = None
        SPIN_US = 0.0
        print("  SLOPE FAILED " + OUT["slope_error"], flush=True)
        dump()

    if "control" in phases:
      try:
        print("=== controls: is main-thread CPU a real host/device discriminator? ===",
              flush=True)
        OUT["controls"] = controls(ttnn, dev)
        dump()
      except Exception as e:                                                # noqa: BLE001
        OUT["controls"] = {"error": f"{type(e).__name__}: {e}"}
        print("  CONTROL FAILED " + OUT["controls"]["error"], flush=True)
        dump()

    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
