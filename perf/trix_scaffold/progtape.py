#!/usr/bin/env python3
"""Program-level attribution of TriangleMultiplication, measured IN THE FOLD.

The census's op-class labels are not evidence here: `k10-p1-trimul-critpath` showed one
`GenericOp` row is seven distinct programs. So this instrument attributes at the level the
device actually dispatches -- (op, operand signature, program config) -- and records, per
program, its call count, its wall, and the chain of `tt_bio/tenstorrent.py` frames that issued
it. Program identity is not asserted from the key: the cold fold reads
`device.num_program_cache_entries()` on both sides of every call, so a key that creates a cache
entry is a program and a key that does not is a repeat of one.

Four folds in one process, one device context:
  cold   tape on, timings discarded, program-cache deltas recorded -- this is the program census
  A      tape off, module body timer only            -> the module wall every share is a share of
  B      tape on, sync both sides of every op        -> the per-program walls
  C      tape off                                    -> the A/A floor on the module wall

AICLK is pinned by the caller and sampled from sysfs at 5 Hz DURING every fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
ROWS: dict = {}
STATE = {"dev": None, "tape": False, "cold": False, "depth": 0}


# --------------------------------------------------------------------------- clock
class ClockSampler(threading.Thread):
    """Card AICLK and power from sysfs at 5 Hz. Opens no device, so it is not contention."""

    ROOT = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.aiclk: list[int] = []
        self.power: list[int] = []
        self._clk = self.ROOT / f"tenstorrent!{card}" / "tt_aiclk"
        if not self._clk.exists():
            cand = next(iter(self.ROOT.glob(f"*{card}/tt_aiclk")), None)
            self._clk = cand if cand else self._clk
        self._pw = next(iter(self._clk.parent.glob("device/hwmon/hwmon*/power1_input")), None)

    def run(self):
        while not self.stop.wait(0.2):
            try:
                v = int(self._clk.read_text().strip())
                if v < 3000:
                    self.aiclk.append(v)
            except (OSError, ValueError):
                pass
            if self._pw is not None:
                try:
                    self.power.append(int(self._pw.read_text().strip()))
                except (OSError, ValueError):
                    pass

    def take(self):
        a, w = self.aiclk[:], self.power[:]
        if not a:
            return {"aiclk_n": 0}
        a.sort()
        return {"aiclk_min": a[0], "aiclk_max": a[-1], "aiclk_median": a[len(a) // 2],
                "aiclk_mean": round(sum(a) / len(a), 1), "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


# --------------------------------------------------------------------------- signatures
def _short(s: str, n: int = 44) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 7] + "~" + hashlib.md5(s.encode()).hexdigest()[:6]


def tsig(t) -> str:
    import ttnn
    try:
        shp = "x".join(str(int(d)) for d in t.shape)
    except Exception:                                                          # noqa: BLE001
        return "?"
    try:
        mc = t.memory_config()
        bt = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
        ml = str(mc.memory_layout).rsplit(".", 1)[-1]
        ml = "" if ml.startswith("INTERLEAVED") else ":" + ml
    except Exception:                                                          # noqa: BLE001
        bt, ml = "?", ""
    try:
        dt = str(t.dtype).rsplit(".", 1)[-1]
    except Exception:                                                          # noqa: BLE001
        dt = "?"
    return f"{shp}|{dt}|{bt}{ml}"


def argsig(a) -> str:
    import ttnn
    if isinstance(a, ttnn.Tensor):
        return tsig(a)
    if isinstance(a, (list, tuple)):
        return "[" + ",".join(argsig(x) for x in a[:3]) + (",..." if len(a) > 3 else "") + "]"
    return _short(repr(a), 28)


def kwsig(kw) -> str:
    import ttnn
    out = []
    for k in sorted(kw):
        v = kw[k]
        if v is None or k in ("bias", "bias_tensor") and v is None:
            continue
        if k == "memory_config":
            try:
                bt = "L1" if v.buffer_type == ttnn.BufferType.L1 else "DRAM"
            except Exception:                                                  # noqa: BLE001
                bt = "?"
            out.append(f"mc={bt}")
        elif k == "compute_kernel_config":
            out.append("ckc=" + _short(v, 30))
        elif k == "program_config":
            out.append("pc=" + type(v).__name__.replace("ProgramConfig", "") + "/" +
                       _short(v, 60))
        elif isinstance(v, ttnn.Tensor):
            out.append(f"{k}={tsig(v)}")
        else:
            out.append(f"{k}={_short(v, 28)}")
    return ",".join(out)


SKIP_FRAMES = {"_tape", "f", "wrapper"}


def sites(limit: int = 5) -> str:
    """The chain of tt_bio frames that issued this dispatch, innermost first."""
    f = sys._getframe(2)
    chain = []
    while f is not None and len(chain) < limit:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn:
            nm = f.f_code.co_name
            if nm not in SKIP_FRAMES:
                chain.append(f"{Path(fn).name.replace('.py', '')}.{nm}:{f.f_lineno}")
                if nm == "__call__" and type(f.f_locals.get("self")).__name__ == \
                        "TriangleMultiplication":
                    break
        f = f.f_back
    return " <- ".join(chain) if chain else "?"


def genop_kernels(pd) -> str:
    try:
        ks = []
        for k in pd.kernels:
            s = str(getattr(k, "kernel_source", "?"))
            ks.append(Path(s).name if "/" in s else _short(s, 20))
        return "+".join(ks)
    except Exception:                                                          # noqa: BLE001
        return "?"


# --------------------------------------------------------------------------- the tape
def install(ttnn, TT):
    def tape(name, sigf):
        def deco(orig):
            def f(*a, **kw):
                if not (STATE["tape"] and STATE["depth"]):
                    return orig(*a, **kw)
                key = (name, sigf(a, kw))
                site = sites()
                dev = STATE["dev"]
                if STATE["cold"]:
                    n0 = dev.num_program_cache_entries()
                    out = orig(*a, **kw)
                    ttnn.synchronize_device(dev)
                    d = dev.num_program_cache_entries() - n0
                    r = ROWS.setdefault(key, {"op": name, "sig": key[1], "n": 0, "s": 0.0,
                                              "cold_n": 0, "new_programs": 0, "sites": {}})
                    r["cold_n"] += 1
                    r["new_programs"] += d
                    r["sites"][site] = r["sites"].get(site, 0) + 1
                    return out
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                r = ROWS.setdefault(key, {"op": name, "sig": key[1], "n": 0, "s": 0.0,
                                          "cold_n": 0, "new_programs": 0, "sites": {}})
                r["n"] += 1
                r["s"] += dt
                r["sites"][site] = r["sites"].get(site, 0) + 1
                return out
            f.__wrapped__ = orig
            return f
        return deco

    def gen(a, kw):
        return f"in={argsig(a[0]) if a else '?'}|{genop_kernels(a[1]) if len(a) > 1 else '?'}"

    def plain(a, kw):
        return "(" + ",".join(argsig(x) for x in a) + ")" + ("|" + kwsig(kw) if kw else "")

    targets = [
        (ttnn, "matmul", plain), (ttnn, "linear", plain), (ttnn, "layer_norm", plain),
        (ttnn, "multiply_", plain), (ttnn, "multiply", plain), (ttnn, "add", plain),
        (ttnn, "chunk", plain), (ttnn, "concat", plain), (ttnn, "clone", plain),
        (ttnn, "permute", plain), (ttnn, "transpose", plain), (ttnn, "reallocate", plain),
        (ttnn, "slice", plain), (ttnn, "unsqueeze", plain), (ttnn, "reshape", plain),
        (ttnn, "to_memory_config", plain), (ttnn, "typecast", plain), (ttnn, "to_layout", plain),
        (ttnn, "sigmoid", plain), (ttnn, "pad", plain), (ttnn, "copy", plain),
        (ttnn, "allocate_tensor_on_device", plain), (ttnn, "from_torch", plain),
        (ttnn, "to_torch", plain), (ttnn, "generic_op", gen),
        (ttnn.experimental, "minimal_matmul", plain),
    ]
    for op in ("reblock_permute", "reblock_permute_back", "reblock_permute_gated"):
        if hasattr(ttnn.experimental, op):
            targets.append((ttnn.experimental, op, plain))
    for mod, name, sigf in targets:
        if hasattr(mod, name):
            setattr(mod, name, tape(name, sigf)(getattr(mod, name)))


def install_body(ttnn, TT):
    """Sync-both-sides wall for the module, and the depth flag the tape gates on."""
    orig = TT.TriangleMultiplication.__call__

    def body(self, *a, **kw):
        dev = STATE["dev"]
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        STATE["depth"] += 1
        try:
            out = orig(self, *a, **kw)
        finally:
            STATE["depth"] -= 1
        ttnn.synchronize_device(dev)
        w = WALL["body:TriangleMultiplication"]
        w["n"] += 1
        w["s"] += time.perf_counter() - t0
        k = f"shape:{'x'.join(str(int(d)) for d in a[0].shape)}|{'end' if self.ending else 'start'}"
        s = WALL[k]
        s["n"] += 1
        s["s"] += time.perf_counter() - t0
        return out
    TT.TriangleMultiplication.__call__ = body


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", default="cold,A,B,B2,C")
    ap.add_argument("--arm", default="shipped", choices=("shipped", "rootcause"),
                    help="rootcause = the module configuration state/trimul-bottleneck-rootcause.md "
                         "decomposed, i.e. the one the 65.4 %% is a share of: in-projection group 1, "
                         "no gated move, no back kernel, no fused tail, no fused g_out.")
    a = ap.parse_args()

    import ttnn
    import tt_bio
    import tt_bio.tenstorrent as TT
    import tt_baseline as B
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {tt_bio.__file__}, not this worktree")

    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_trix_{a.model}_{a.size}", tgt, a3m)
    STATE["dev"] = TT.get_device()
    g = STATE["dev"].compute_with_storage_grid_size()

    import tt_bio.reblock_permute as RB
    if a.arm == "rootcause":
        TT._TRIMUL_INPROJ_GROUP = 1
        TT._TRIMUL_MASK_AFTER_MOVE = False
        TT._TRIMUL_TAIL_F1 = False
        TT._TRIMUL_FUSED_GOUT = False
        RB.set_enabled_gated(False)
        RB.set_enabled_back(False)
        RB.set_enabled(False)

    install_body(ttnn, TT)
    install(ttnn, TT)

    res = {
        "host": socket.gethostname(), "model": a.model, "size": a.size, "card": a.card,
        "grid": [g.x, g.y],
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "env": {k: os.environ.get(k) for k in
                ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER")},
        "protocol": {"recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
        "arm": a.arm,
        "flags": {"reblock_gated": RB._ENABLED_GATED, "reblock_back": RB._ENABLED_BACK,
                  "reblock_fwd": RB._ENABLED,
                  "trimul_inproj_group": TT._TRIMUL_INPROJ_GROUP,
                  "mask_after_move": TT._TRIMUL_MASK_AFTER_MOVE,
                  "fused_gout": TT._TRIMUL_FUSED_GOUT,
                  "tail_f1": TT._TRIMUL_TAIL_F1,
                  "inproj_rowblock": TT._TRIMUL_INPROJ_ROWBLOCK,
                  "mm_transpose": TT._TRIMUL_MM_TRANSPOSE},
        "folds": [],
    }

    def run(tag, tape, cold):
        WALL.clear()
        STATE["tape"], STATE["cold"] = tape, cold
        cs = ClockSampler(a.card)
        cs.start()
        t0 = time.perf_counter()
        fold_s, m = one_fold()
        cs.stop.set()
        rec = {"tag": tag, "tape": tape, "cold": cold,
               "fold_s": round(fold_s, 3), "wall_s": round(time.perf_counter() - t0, 3),
               "clock": cs.take(), "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
               "loadavg1": round(os.getloadavg()[0], 2),
               "module": {k: {"calls": v["n"], "s": round(v["s"], 4)}
                          for k, v in sorted(WALL.items())}}
        res["folds"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        b = rec["module"].get("body:TriangleMultiplication", {})
        print(f"  {tag}: fold {fold_s:.2f}s  trimul {b.get('s')}s over {b.get('calls')} calls "
              f"clk {rec['clock'].get('aiclk_min')}-{rec['clock'].get('aiclk_max')} "
              f"plddt {m.get('plddt')}", flush=True)
        return rec

    plan = {"cold": (True, True), "A": (False, False), "B": (True, False),
            "B2": (True, False), "C": (False, False)}
    for tag in a.folds.split(","):
        tape, cold = plan[tag]
        run(tag, tape, cold)
        if tag.startswith("B"):
            res.setdefault("taped_folds", {})[tag] = {
                f"{k[0]}|{k[1]}": {"n": v["n"], "ms": round(v["s"] * 1e3, 3)}
                for k, v in ROWS.items() if v["n"]}
            for v in ROWS.values():
                v["n"], v["s"] = 0, 0.0
        if tag == "cold":
            res["programs_cold"] = {f"{k[0]}|{k[1]}": v for k, v in ROWS.items()}
            res["cold_cache_entries"] = STATE["dev"].num_program_cache_entries()
            for v in ROWS.values():
                v["n"] = 0
                v["s"] = 0.0
                v["sites"] = {}
        a.out.write_text(json.dumps(res, indent=1))

    tf = res.get("taped_folds", {})
    ref = tf.get("B", {})
    rep = tf.get("B2", {})
    res["programs"] = sorted(
        ({"op": v["op"], "sig": v["sig"],
          "n": ref.get(f"{k[0]}|{k[1]}", {}).get("n", 0),
          "ms": ref.get(f"{k[0]}|{k[1]}", {}).get("ms", 0.0),
          "ms_repeat": rep.get(f"{k[0]}|{k[1]}", {}).get("ms", 0.0),
          "cold_n": v["cold_n"], "new_programs": v["new_programs"],
          "sites": dict(sorted(v["sites"].items(), key=lambda kv: -kv[1]))}
         for k, v in ROWS.items()),
        key=lambda r: -r["ms"])
    res["taped_sum_s"] = round(sum(r["ms"] for r in res["programs"]) / 1e3, 4)
    res["taped_sum_repeat_s"] = round(sum(r["ms_repeat"] for r in res["programs"]) / 1e3, 4)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"taped sum {res['taped_sum_s']:.3f}s over {len(res['programs'])} program keys; "
          f"{sum(r['new_programs'] for r in res['programs'])} program-cache entries created",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
