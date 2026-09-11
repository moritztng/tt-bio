#!/usr/bin/env python3
"""Boltz-2 512 aa on Blackhole: the published cell re-measured, and the fold attributed.

Everything runs in ONE process, because a p300c chip wedges inside ``ttnn.open_device`` on its
fourth open since that chip's last reset. Phases are selected with ``--phases`` and each writes
its result as soon as it has one, so a turn that runs out of time still lands what it has.

  baseline  the published protocol -- cdk2x2_512.yaml + its fixed 35-row a3m, 3 recycles, 200
            sampling steps, 1 sample, seed 0, templates off -- through the production
            ``_WorkerState.predict_one`` (host featurisation + fold + CIF write), cold fold
            discarded, arms alternated plain/instrumented so the instrument is proved not to
            move the wall it measures. The instrumented arm carries ``tt_baseline.Instrument``,
            which is the bracket the published host/device split was taken with.
  attrib    one fold with every ``tt_bio.tenstorrent`` Module and TorchWrapper bracketed:
            inclusive wall per unit with a device sync on both sides, and a ``ttnn.graph``
            capture of one settled call per (unit, input shape). Nested units push a marker
            into the capture, so the capture of a pairformer block resolves per sub-unit and
            per ttnn op instead of being one 12.17 GB number.
  census    a second fold with every ttnn entry point wrapped in a byte model (DRAM inputs read
            once, DRAM outputs written once) charged to the bracket stack. This is what prices
            the glue -- the ops that belong to no module -- which no capture can reach, since a
            capture of a region big enough to contain the glue is also big enough to contain
            the whole trunk.
  control   one cdk2x2_298 fold, the monomeric control the RMSD bar is read on.

The attrib and census folds serialise host and device at every bracket, so their fold wall is
NOT a baseline number. The baseline number comes from the plain arm of phase ``baseline``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump() -> None:
    if OUT_PATH is not None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# the bracket
# --------------------------------------------------------------------------------------------
class Brackets:
    """Wall and DRAM traffic per named module, for one fold.

    Every ``tt_bio.tenstorrent`` class that defines its own ``__call__`` (the TT modules) and
    every ``TorchWrapper`` that defines its own ``forward`` is wrapped. A call records its path
    down the bracket stack, so a unit's exclusive time is its inclusive time minus its
    children's, and the fold closes by construction rather than by subtraction.

    ``capture_max_ms`` keeps captures bounded: a settled call is captured only if its first
    calls were shorter than that, because capturing the trunk as one region would hold several
    hundred thousand nodes in host RAM and dedupe a weight read across 264 blocks into one.
    """

    def __init__(self, dev, ttnn, capture_call=3, capture_max_ms=250.0, sync=True):
        self.dev = dev
        self.ttnn = ttnn
        self.capture_call = capture_call
        self.capture_max_ms = capture_max_ms
        self.sync = sync
        self.stack: list[str] = []
        self.times: dict[str, list[float]] = defaultdict(list)   # path -> inclusive seconds
        self.child_s: dict[str, float] = defaultdict(float)      # path -> children inclusive
        self.sig_times: dict[str, list[float]] = defaultdict(list)
        self.captures: dict[str, list] = {}                      # sig -> raw graph
        self.capturing = False
        self._orig: list = []

    # -- naming ------------------------------------------------------------------------------
    @staticmethod
    def shape_sig(args) -> str:
        parts = []
        for x in args:
            sh = getattr(x, "shape", None)
            if sh is not None:
                try:
                    parts.append("x".join(str(int(d)) for d in sh))
                except TypeError:
                    pass
        return ",".join(parts[:2])

    def _sync(self):
        if self.sync:
            self.ttnn.synchronize_device(self.dev)

    def call(self, name, fn, self_obj, a, k):
        sig = f"{name}|{self.shape_sig(a)}"
        path = "/".join(self.stack + [name])
        n_before = len(self.sig_times[sig])
        want = (
            not self.capturing
            and n_before >= self.capture_call >= 0
            and sig not in self.captures
            and (not self.sig_times[sig]
                 or 1e3 * st.median(self.sig_times[sig]) <= self.capture_max_ms)
        )
        marker = self.capturing and not want
        if marker:
            self.ttnn.graph.track_function_start(f"unit::{name}")
        if want:
            self.capturing = True
            self._sync()
            self.ttnn.graph.begin_graph_capture(self.ttnn.graph.RunMode.NORMAL)
            self.ttnn.graph.track_function_start(f"unit::{name}")
        self.stack.append(name)
        self._sync()
        t0 = time.perf_counter()
        try:
            return fn(self_obj, *a, **k) if self_obj is not None else fn(*a, **k)
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self.stack.pop()
            if want:
                self.ttnn.graph.track_function_end()
                self.captures[sig] = self.ttnn.graph.end_graph_capture()
                self.capturing = False
            elif marker:
                self.ttnn.graph.track_function_end()
            self.times[path].append(dt)
            self.sig_times[sig].append(dt)
            if self.stack:
                self.child_s["/".join(self.stack)] += dt

    # -- installation ------------------------------------------------------------------------
    def install(self, T) -> list[str]:
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

    # -- results -----------------------------------------------------------------------------
    def tree(self) -> dict:
        rows = {}
        for path, ts in sorted(self.times.items()):
            rows[path] = {
                "calls": len(ts),
                "incl_s": round(sum(ts), 5),
                "excl_s": round(sum(ts) - self.child_s.get(path, 0.0), 5),
                "median_ms": round(1e3 * st.median(ts), 5),
            }
        return rows

    def sigs(self) -> dict:
        return {s: {"calls": len(ts), "median_ms": round(1e3 * st.median(ts), 5),
                    "total_ms": round(1e3 * sum(ts), 3)}
                for s, ts in sorted(self.sig_times.items())}


BR: Brackets | None = None


# --------------------------------------------------------------------------------------------
# the ttnn byte model, for the ops that belong to no module
# --------------------------------------------------------------------------------------------
_ELEM_BYTES = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625,
               "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0}


def tensor_bytes(t) -> tuple[float, bool]:
    """(bytes, is_dram) for one ttnn tensor, tile-padded, from its own spec."""
    try:
        shp = list(t.padded_shape) if hasattr(t, "padded_shape") else list(t.shape)
    except Exception:
        return 0.0, False
    vol = 1
    for d in shp:
        vol *= int(d)
    dt = str(getattr(t, "dtype", "")).split(".")[-1].upper()
    b = vol * _ELEM_BYTES.get(dt, 2.0)
    dram = False
    try:
        dram = "DRAM" in str(t.memory_config().buffer_type)
    except Exception:
        pass          # a host tensor has no memory config, and moves no DRAM traffic
    return b, dram


class Census:
    """Every ttnn call charged to the bracket stack, with a floor byte model.

    A floor, not a measurement: DRAM inputs read once and DRAM outputs written once counts no
    intermediate and no multi-pass re-read, so it under-counts exactly where a graph capture
    over-counts nothing. Its use is the ratio -- captured / modelled for the units that have
    both -- which is what lets the glue's modelled bytes be read as traffic.
    """

    def __init__(self, ttnn):
        self.ttnn = ttnn
        self.rows: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"n": 0, "in_b": 0.0, "out_b": 0.0})
        self._orig = {}

    def charge(self, op, args, out):
        path = "/".join(BR.stack) if BR and BR.stack else "(glue)"
        r = self.rows[(path, op)]
        r["n"] += 1
        seen = set()
        for x in args:
            if isinstance(x, self.ttnn.Tensor):
                if id(x) in seen:
                    continue
                seen.add(id(x))
                b, dram = tensor_bytes(x)
                if dram:
                    r["in_b"] += b
        outs = out if isinstance(out, (list, tuple)) else [out]
        for x in outs:
            if isinstance(x, self.ttnn.Tensor):
                b, dram = tensor_bytes(x)
                if dram:
                    r["out_b"] += b

    # Ops that move no traffic of their own, or whose argument is not a read.
    SKIP = {"synchronize_device", "Tensor", "graph", "deallocate", "get_device",
            "open_device", "close_device", "open_mesh_device", "close_mesh_device",
            "dump_tensor", "load_tensor", "begin_trace_capture", "end_trace_capture",
            "execute_trace", "release_trace", "GetDefaultDevice", "SetDefaultDevice"}

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
                    out = f(*a, **k)
                    try:
                        CEN.charge(nm, a, out)
                    except Exception:
                        pass
                    return out
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

    def table(self) -> list[dict]:
        return [{"path": p, "op": o, "n": r["n"],
                 "in_GB": round(r["in_b"] / 1e9, 6), "out_GB": round(r["out_b"] / 1e9, 6)}
                for (p, o), r in sorted(self.rows.items(), key=lambda kv: -kv[1]["in_b"])]


CEN: Census | None = None


# --------------------------------------------------------------------------------------------
def main() -> int:
    global OUT_PATH, BR, CEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--capdir", type=Path, default=None, help="where raw captures are written")
    ap.add_argument("--phases", default="baseline,control,attrib,census")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--capture-max-ms", type=float, default=250.0)
    ap.add_argument("--cifdir", type=Path, required=True)
    args = ap.parse_args()
    OUT_PATH = args.out
    phases = [p for p in args.phases.split(",") if p]
    args.cifdir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg           # injects conf_kwargs, as the cell did
    sys.path[:] = snap
    patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt = fix / f"cdk2x2_{args.size}.yaml"
    a3m = fix / f"cdk2x2_{args.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{args.size}"

    import importlib.metadata as im
    OUT["env"] = {
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "ttnn": im.version("ttnn"), "torch": torch.__version__,
        "fast": bool(args.fast),
        "protocol": {"fixture": f"perf/size512/fixtures/cdk2x2_{args.size}.yaml + its a3m",
                     "recycling_steps": B.RECYCLING_STEPS,
                     "sampling_steps": B.SAMPLING_STEPS,
                     "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": B.SEED,
                     "timed_region": "predict_one (featurise + fold + CIF write)"},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phases_requested": phases,
    }
    dump()

    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m, fast=args.fast)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa",
                                            "card_type", "aiclk_mhz") if k in meta})
    dev = T.get_device()
    struct_dir = Path(meta["struct_dir"])
    dump()

    def keep_cifs(tag: str) -> dict:
        got = {}
        for f in sorted(struct_dir.glob("*.cif")):
            dst = args.cifdir / f"{tag}_{f.name}"
            shutil.copyfile(f, dst)
            got[f.name] = sha256_file(dst)
        return got

    def loadavg():
        return open("/proc/loadavg").read().split()[:3]

    # ---- cold fold, discarded ---------------------------------------------------------------
    print("=== cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa") or meta.get("n_msa"), "fold ran without an MSA"
    OUT["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.3f}s plddt={cold_m.get('plddt')}", flush=True)
    dump()

    # ---- phase: baseline --------------------------------------------------------------------
    if "baseline" in phases:
        inst = B.Instrument("boltz2", state)
        runs = []
        for i in range(args.reps):
            for arm in ("plain", "instr"):
                if arm == "instr":
                    inst.on()
                try:
                    fold_s, m = one_fold()
                finally:
                    inst.off()
                rec = {"arm": arm, "rep": i, "fold_s": round(fold_s, 3),
                       "plddt": m.get("plddt", m.get("complex_plddt")),
                       "n_tokens": m.get("n_tokens"),
                       "cifs": keep_cifs(f"base_{arm}{i}"), "loadavg": loadavg()}
                if arm == "instr":
                    rec["phase"] = inst.row(fold_s)
                runs.append(rec)
                print(f"  {arm:5s} rep{i} {fold_s:8.3f}s plddt={rec['plddt']} "
                      f"{rec.get('phase', {}).get('host', '')} "
                      f"{list(rec['cifs'].values())[0][:16]}", flush=True)
                OUT["baseline"] = runs
                dump()

        plain = [r["fold_s"] for r in runs if r["arm"] == "plain"]
        instr = [r["fold_s"] for r in runs if r["arm"] == "instr"]
        rows = [r["phase"] for r in runs if r["arm"] == "instr"]
        digests = sorted({d for r in runs for d in r["cifs"].values()})
        OUT["baseline_summary"] = {
            "n": len(plain),
            "plain_s": plain, "instr_s": instr,
            "plain_median_s": round(st.median(plain), 3),
            "instr_median_s": round(st.median(instr), 3),
            "aa_floor_s": round(max(plain) - min(plain), 3),
            "aa_floor_pct": round(100 * (max(plain) - min(plain)) / st.median(plain), 3),
            "instr_delta_pct": round(100 * (st.median(instr) - st.median(plain))
                                     / st.median(plain), 3),
            "host_s": round(st.median([r["host"] for r in rows]), 4),
            "device_s": round(st.median([r["device"] for r in rows]), 4),
            "transfer_s": round(st.median([r["transfer"] for r in rows]), 4),
            "residual_s": round(st.median([r["residual"] for r in rows]), 4),
            "plddt": sorted({r["plddt"] for r in runs}),
            "cif_sha256": digests,
            "cif_sha256_16": sorted({d[:16] for d in digests}),
            "bit_identical_across_folds": len(digests) == 1,
        }
        dump()
        print("[baseline]", json.dumps(OUT["baseline_summary"], indent=1), flush=True)

    # ---- phase: control (cdk2x2_298) --------------------------------------------------------
    if "control" in phases:
      try:
        c_tgt, c_a3m = fix / "cdk2x2_298.yaml", fix / "cdk2x2_298.a3m"
        n_msa = B.seed_msa_cache(c_tgt, c_a3m, msa_dir)
        job = dict(meta["job_cfg"])
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        m, _b, _f = state.predict_one(c_tgt, job)
        dt = time.perf_counter() - t0
        OUT["control_298"] = {"fold_s": round(dt, 3), "n_msa": n_msa,
                              "plddt": m.get("plddt", m.get("complex_plddt")),
                              "n_tokens": m.get("n_tokens"),
                              "cifs": keep_cifs("ctl298")}
        dump()
        print("[control]", json.dumps(OUT["control_298"]), flush=True)
        # restore the 512 MSA the timed folds use
        B.seed_msa_cache(tgt, a3m, msa_dir)
      except Exception:
        import traceback
        OUT["control_error"] = traceback.format_exc()
        print("[control] FAILED\n" + OUT["control_error"], flush=True)
        dump()

    # ---- phase: attrib ----------------------------------------------------------------------
    if "attrib" in phases:
      try:
        capdir = args.capdir or args.out.parent / "captures"
        capdir.mkdir(parents=True, exist_ok=True)
        BR = Brackets(dev, ttnn, capture_max_ms=args.capture_max_ms)
        wrapped = BR.install(T)
        print(f"[attrib] bracketed {len(wrapped)} classes: {','.join(wrapped)}", flush=True)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        m, _b, _f = state.predict_one(tgt, dict(meta["job_cfg"]))
        wall = time.perf_counter() - t0
        BR.remove()
        caps = {}
        for sig, g in BR.captures.items():
            fn = capdir / ("cap_" + sig.replace("/", "_").replace("|", "__") + ".json.gz")
            with gzip.open(fn, "wt") as fh:
                json.dump(g, fh)
            caps[sig] = {"file": str(fn.relative_to(ROOT)), "nodes": len(g)}
        OUT["attrib"] = {
            "instrumented_fold_s": round(wall, 3),
            "plddt": m.get("plddt", m.get("complex_plddt")),
            "cifs": keep_cifs("attrib"),
            "n_classes": len(wrapped), "classes": wrapped,
            "tree": BR.tree(), "sigs": BR.sigs(), "captures": caps,
            "note": ("per-call device syncs serialise host and device; this fold's wall is a "
                     "diagnostic, not a baseline"),
        }
        dump()
        print(f"[attrib] fold {wall:.3f}s, {len(caps)} captures", flush=True)
      except Exception as e:
        import traceback
        OUT["attrib_error"] = traceback.format_exc()
        print("[attrib] FAILED\n" + OUT["attrib_error"], flush=True)
      finally:
        if BR is not None:
            BR.remove()
        BR = None
        dump()

    # ---- phase: census ----------------------------------------------------------------------
    if "census" in phases:
      try:
        BR = Brackets(dev, ttnn, capture_call=-1, sync=False)   # stack only, no syncs
        wrapped = BR.install(T)
        CEN = Census(ttnn)
        n_wrapped = CEN.install()
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        m, _b, _f = state.predict_one(tgt, dict(meta["job_cfg"]))
        wall = time.perf_counter() - t0
        CEN.remove()
        BR.remove()
        tbl = CEN.table()
        OUT["census"] = {
            "instrumented_fold_s": round(wall, 3),
            "plddt": m.get("plddt", m.get("complex_plddt")),
            "cifs": keep_cifs("census"),
            "ttnn_names_wrapped": n_wrapped,
            "tree": BR.tree(),
            "rows": tbl,
            "total_in_GB": round(sum(r["in_GB"] for r in tbl), 4),
            "total_out_GB": round(sum(r["out_GB"] for r in tbl), 4),
            "note": "floor byte model: DRAM inputs read once, DRAM outputs written once",
        }
        dump()
        print(f"[census] fold {wall:.3f}s, {len(tbl)} (path,op) rows, "
              f"{OUT['census']['total_in_GB']:.1f}+{OUT['census']['total_out_GB']:.1f} GB",
              flush=True)
      except Exception as e:
        import traceback
        OUT["census_error"] = traceback.format_exc()
        print("[census] FAILED\n" + OUT["census_error"], flush=True)
      finally:
        if CEN is not None:
            CEN.remove()
        if BR is not None:
            BR.remove()
        BR = None
        dump()

    OUT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    print("DONE " + str(OUT_PATH), flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
