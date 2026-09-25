#!/usr/bin/env python3
"""of3t-infab: does wk/of3t change inference? Digest, census and dispatched op trace, per model.

    infab.py --before TREE --after TREE --repo GIT --card 1 --workdir W --out INFAB.json

Copied from perf/land_standing/d264/infab.py (origin/wk/land-d264) and widened from four models to
the catalog. Per model, folds interleaved BEFORE, AFTER, BEFORE, AFTER on one card: the two BEFORE
folds are the A/A floor, BEFORE vs AFTER the A/B. Every fold carries three instruments, all inside
the folding process:

  census   every `tt_bio.*` module the process imported (intersected with the files the diff
           touches, which is what decides whether the model is in scope), whether
           `tt_bio.autograd` is present, and its exact softmax / layer norm counters read without
           importing it. Rewritten to $INFAB_CENSUS_DIR/<pid>.json on each watched import and
           once a second, because `tt_bio.main predict` terminates its spawned worker and an
           atexit hook never fires there (D236).
  trace    every ttnn operation called through `FastOperation.__call__` (which is every ttnn op,
           operator overload and `generic_op` custom kernel in fast-runtime mode), one line per
           call: nesting depth, op name, and each argument's signature -- shape, dtype, layout and
           memory config for a tensor, the kernel sources and compile-time args for a program
           descriptor, the repr of everything else (program configs, compute kernel configs) with
           pointers and the tree root removed. Same op sequence with same signatures means the
           same programs dispatched, so identical device work on any host load.
  digest   sha256 of every output structure/array (CIF/PDB bytes, npz array contents).

AICLK is sampled from tt-smi every 3 s DURING each fold and stamped beside its wall time.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

SMI = "/home/ttuser/.local/bin/tt-smi"
SITE = r'''import atexit, json, os, re, sys, threading
_SEEN = set()
_BUILT = [0]
def _dump():
    out = {"pid": os.getpid()}
    mod = sys.modules.get("tt_bio.autograd")
    out["autograd_imported"] = mod is not None
    for n in ("EXACT_SOFTMAX_STATS", "EXACT_LAYER_NORM_STATS"):
        out[n] = dict(getattr(mod, n)) if mod is not None and hasattr(mod, n) else None
    out["taped_ttnn_imported"] = "tt_bio.taped_ttnn" in sys.modules
    out["train_modules"] = sorted(m for m in sys.modules if m.startswith("tt_bio.train"))
    out["tt_bio_modules"] = sorted(m for m in list(sys.modules) if m == "tt_bio" or m.startswith("tt_bio."))
    out["input_atom_encoder_built"] = _BUILT[0]
    out["tt_bio_file"] = getattr(sys.modules.get("tt_bio"), "__file__", None)
    out["argv"] = sys.argv[:3]
    out["imports_seen"] = sorted(_SEEN)
    out["ops_traced"] = _N[0]
    return out
def _write():
    d = os.environ.get("INFAB_CENSUS_DIR")
    if d:
        p = os.path.join(d, f"{os.getpid()}.json")
        tmp = p + f".{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(_dump(), f)
        os.replace(tmp, p)
atexit.register(_write)
import importlib.abc, importlib.machinery
_WATCH = ("tt_bio.train", "tt_bio.autograd", "tt_bio.taped_ttnn", "tt_bio.")
class _Seen(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.startswith(_WATCH) and name not in _SEEN:
            _SEEN.add(name)
            _write()
        return None
# --- op trace
_N = [0]
_TRD = os.environ.get("INFAB_TRACE_DIR")
_ROOT = os.environ.get("INFAB_TREE", "\0")
_ADDR = re.compile(r"0x[0-9a-fA-F]+|tensor_id=\d+|id=\d+")
_LOCK = threading.Lock()
_TL = threading.local()
_FH = [None, None]
def _norm(s):
    return _ADDR.sub("", s.replace(_ROOT, "ROOT"))
def _sig(v, d=0):
    if d > 4:
        return "..."
    t = type(v)
    mn, tn = t.__module__ or "", t.__name__
    if tn == "Tensor" and mn.startswith("ttnn"):
        s = []
        for f in ("shape", "dtype", "layout"):
            try:
                a = getattr(v, f)
                s.append(str(tuple(a)) if f == "shape" else str(a))
            except Exception:
                s.append("?")
        try:
            st = str(v.storage_type())
            s.append(st)
            if "DEVICE" in st and v.is_allocated():
                s.append(_norm(str(v.memory_config())))
        except Exception:
            pass
        return "T(" + ",".join(s) + ")"
    if tn == "Tensor" and mn.startswith("torch"):
        return f"torch{tuple(v.shape)}{v.dtype}"
    if isinstance(v, (list, tuple)):
        return ("[" if isinstance(v, list) else "(") + ",".join(_sig(x, d + 1) for x in v[:64]) + (
            f",+{len(v) - 64}" if len(v) > 64 else "") + ("]" if isinstance(v, list) else ")")
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_sig(x, d + 1)}" for k, x in list(v.items())[:64]) + "}"
    if hasattr(v, "kernels") and hasattr(v, "cbs"):   # a ProgramDescriptor (generic_op)
        try:
            ks = [(_norm(str(getattr(k, "kernel_source", ""))),
                   list(getattr(k, "compile_time_args", []) or []),
                   _norm(repr(getattr(k, "defines", ""))),
                   _norm(repr(getattr(k, "core_ranges", ""))),
                   _norm(repr(getattr(k, "config", ""))))
                  for k in v.kernels]
            return "PD" + repr(ks) + "|cbs=" + str(len(v.cbs))
        except Exception as e:
            return "PD?" + type(e).__name__
    try:
        return _norm(repr(v))[:600]
    except Exception:
        return tn
def _patch(m):
    FO = m.FastOperation
    call = FO.__call__
    def traced(self, *a, **k):
        dep = getattr(_TL, "d", 0)
        try:
            line = f"{dep}\t{self.python_fully_qualified_name}\t{_sig(a)}\t{_sig(k)}\n"
        except Exception as e:
            line = f"{dep}\t{self.python_fully_qualified_name}\tSIGFAIL {type(e).__name__}\n"
        with _LOCK:
            if _FH[1] != os.getpid():   # first op, or a forked child that inherited the handle
                _FH[0] = open(os.path.join(_TRD, f"{os.getpid()}.trace"), "w", buffering=1)   # line-buffered: the CLI SIGTERMs its worker
                _FH[1] = os.getpid()
            _FH[0].write(line)
            _N[0] += 1
        _TL.d = dep + 1
        try:
            return call(self, *a, **k)
        finally:
            _TL.d = dep
    FO.__call__ = traced
def _flush():
    if _FH[0] is not None and _FH[1] == os.getpid():
        with _LOCK:
            _FH[0].flush()
def _child():
    global _LOCK
    _LOCK = threading.Lock()
    _FH[0] = _FH[1] = None
os.register_at_fork(before=_flush, after_in_child=_child)
class _Hook(importlib.abc.MetaPathFinder):
    """Wraps a module's exec: counts InputAtomEncoder builds, patches ttnn's op dispatcher."""
    def find_spec(self, name, path, target=None):
        if name not in ("tt_bio.openfold3", "ttnn.decorators"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module
        def exec_module(m):
            run(m)
            if name == "ttnn.decorators":
                if _TRD:
                    _patch(m)
                return
            if hasattr(m, "InputAtomEncoder"):
                init = m.InputAtomEncoder.__init__
                def counted(self, *a, **k):
                    _BUILT[0] += 1
                    _write()
                    init(self, *a, **k)
                m.InputAtomEncoder.__init__ = counted
            _write()
        spec.loader.exec_module = exec_module
        return spec
sys.meta_path.insert(0, _Hook())
sys.meta_path.insert(0, _Seen())
import time
def _snap():
    while os.environ.get("INFAB_CENSUS_DIR"):
        _flush()
        _write()
        time.sleep(1)
threading.Thread(target=_snap, daemon=True).start()
atexit.register(_flush)
'''


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(p: Path) -> str:
    """Bytes for a structure; array contents for an npz (the zip container carries mtimes)."""
    if p.suffix == ".npz":
        import numpy as np
        h = hashlib.sha256()
        with np.load(p, allow_pickle=False) as z:
            for k in sorted(z.files):
                a = z[k]
                h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
                h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    return sha(p)


def clock(card: int, stop: threading.Event, out: list):
    while not stop.is_set():
        try:
            d = json.loads(subprocess.run([SMI, "-s"], capture_output=True, text=True,
                                          timeout=20).stdout)
            out.append(int(d["device_info"][card]["telemetry"]["aiclk"].strip()))
        except Exception:
            pass
        stop.wait(3)


FX = "perf/size512/fixtures/cdk2x2_128.yaml"
PRED = ("--single_sequence", "--sampling_steps", "6", "--diffusion_samples", "1", "--seed", "0")
# key = the module whose import marks a process as the one that folds this model.
MODELS = {
    "openfold3":   {"key": "tt_bio.openfold3", "argv": ["predict", FX, "--model", "openfold3", *PRED]},
    "boltz2":      {"key": "tt_bio.boltz2", "argv": ["predict", FX, "--model", "boltz2", *PRED]},
    "protenix-v2": {"key": "tt_bio.protenix", "argv": ["predict", FX, "--model", "protenix-v2", *PRED]},
    "esmfold2":    {"key": "tt_bio.esmfold2", "argv": ["predict", FX, "--model", "esmfold2", *PRED]},
    "af2ig":       {"key": "tt_bio.af2", "script": "scripts/af2_port/fold_timing.py",
                    "argv": ["--pdb", "scripts/af2_port/parity_artifacts/designpop_bg119/binder_complex.pdb",
                             "--reps", "2", "--params",
                             "/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz"]},
    "boltzgen":    {"key": "tt_bio.boltzgen", "argv": ["design", "tests/fixtures/boltzgen/bg400.yaml",
                                                       "--model", "boltzgen", "--num_designs", "1",
                                                       "--steps", "design"]},
    "openbind":    {"key": "tt_bio.openfold3", "argv": ["predict", FX, "--model", "openbind", *PRED]},
    "protenix-v1": {"key": "tt_bio.protenix", "argv": ["predict", FX, "--model", "protenix-v1", *PRED]},
    "opendde":     {"key": "tt_bio.opendde", "argv": ["predict", FX, "--model", "opendde", *PRED]},
    "rf3":         {"key": "tt_bio.rf3", "argv": ["predict", FX, "--model", "rf3", *PRED]},
    "nesso1":      {"key": "tt_bio.nesso1", "argv": ["affinity", "perf/nesso1/inputs/ladder/aa128/cdk2_128.yaml",
                                                     "--model", "nesso1", "--trunk", "bf16",
                                                     "--recycling_steps", "5", "--tokens_budget", "256"]},
    "esmc-300m":   {"key": "tt_bio.esmc", "argv": ["embed", "perf/of3t_infab/fixtures/cdk2_128.fasta",
                                                  "--model", "esmc-300m"]},
    "saprot-35m":  {"key": "tt_bio.saprot", "argv": ["saprot", "perf/of3t_infab/fixtures/cdk2_128.fasta",
                                                    "--model", "saprot-35m"]},
    "rfd3":        {"key": "tt_bio.rfd3", "argv": ["design",
                                                  "scripts/rfd3_port/parity_artifacts/iai_protein/iai_inputs.yaml",
                                                  "--model", "rfd3", "--num_designs", "1", "--seed", "0",
                                                  "--num_timesteps", "20", "--from_pdb"]},
    "pxdesign":    {"key": "tt_bio.pxdesign", "argv": ["design", "tests/fixtures/pxdesign/PDL1.yaml",
                                                      "--model", "pxdesign", "--num_designs", "1", "--seed", "0",
                                                      "--n_step", "50"]},
}
OUT_FLAG = {"af2ig": "--out"}


def trace_summary(tdir: Path, census: list, key: str) -> dict:
    """Per folding process: op count, sha of the whole normalized trace, and op-name counts."""
    procs = []
    for c in census:
        if key not in c.get("tt_bio_modules", []):
            continue
        p = tdir / f"{c['pid']}.trace"
        if not p.exists():
            procs.append({"pid": c["pid"], "n_ops": 0, "sha": None})
            continue
        b = p.read_bytes()
        names = collections.Counter(ln.split("\t", 2)[1] for ln in b.decode().splitlines() if ln)
        procs.append({"pid": c["pid"], "n_ops": sum(names.values()),
                      "sha": hashlib.sha256(b).hexdigest(), "path": str(p), "names": dict(names)})
    return {"processes": procs}


def fold(py, tree: Path, inputs: Path, name: str, out: Path, card: str, holder: str):
    spec = MODELS[name]
    subprocess.run(["rm", "-rf", str(out)], check=False)
    out.mkdir(parents=True)
    sc = out.parent / ("_sc_" + out.name)
    subprocess.run(["rm", "-rf", str(sc)], check=False)
    (sc / "census").mkdir(parents=True)
    (sc / "trace").mkdir()
    (sc / "sitecustomize.py").write_text(SITE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_BIO_")}
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": holder, "OMP_NUM_THREADS": "8",
                "PYTHONPATH": str(sc) + os.pathsep + str(tree),
                "INFAB_CENSUS_DIR": str(sc / "census"), "INFAB_TRACE_DIR": str(sc / "trace"),
                "INFAB_TREE": str(tree)})
    # Inputs are read from ONE tree (--inputs) so both arms fold the same bytes.
    argv = [str(inputs / a) if (inputs / a).is_file() and not a.startswith("/") else a
            for a in spec["argv"]]
    if "script" in spec:
        cmd = [py, str(tree / spec["script"]), *argv, OUT_FLAG[name], str(out / "af2.json")]
    else:
        cmd = [py, "-m", "tt_bio.main", *argv, "--out_dir", str(out)]
    samples, stop = [], threading.Event()
    th = threading.Thread(target=clock, args=(int(card), stop, samples), daemon=True)
    th.start()
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True, timeout=2400)
        rc, text = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", str(e.stdout or "")[-2000:]
    secs = time.time() - t
    stop.set(); th.join()
    census = [json.loads(p.read_text()) for p in sorted((sc / "census").glob("*.json"))]
    if "script" in spec:
        rep = json.loads((out / "af2.json").read_text()) if (out / "af2.json").exists() else {}
        digests = {"structure_sha16_all": rep.get("structure_sha16_all")}
    else:
        digests = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
                   if p.suffix in (".cif", ".pdb", ".npz", ".npy")}
    runtime = None
    for p in out.rglob("results.json"):
        try:
            ts = [x.get("runtime_s") for x in json.loads(p.read_text()) if isinstance(x, dict)]
            ts = [x for x in ts if x is not None]
            runtime = max(ts) if ts else None
        except Exception:
            pass
    s = sorted(samples)
    return {"rc": rc, "seconds": round(secs, 2), "runtime_s": runtime, "digests": digests,
            "census": census, "trace": trace_summary(sc / "trace", census, spec["key"]),
            "aiclk_mhz_sampled_DURING": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                          "max": s[-1]} if s else None),
            "cmd": [c.replace(str(tree), "TREE") for c in cmd],
            "tail": text[-3000:] if rc != 0 else ""}


def trace_diff(ra: dict, rb: dict, limit: int = 80) -> dict:
    """Op-by-op comparison of two folds' folding processes, in pid order."""
    pa, pb = ra["trace"]["processes"], rb["trace"]["processes"]
    same = [p["sha"] for p in pa] == [p["sha"] for p in pb] and all(p["sha"] for p in pa)
    out = {"identical": bool(same) and len(pa) > 0, "n_ops": [[p["n_ops"] for p in pa], [p["n_ops"] for p in pb]]}
    if not out["identical"] and len(pa) == len(pb):
        diffs = []
        for x, y in zip(pa, pb):
            if x["sha"] == y["sha"]:
                continue
            nx = collections.Counter(x.get("names", {})); ny = collections.Counter(y.get("names", {}))
            la = Path(x["path"]).read_text().splitlines() if x.get("path") else []
            lb = Path(y["path"]).read_text().splitlines() if y.get("path") else []
            first = next((i for i, (u, v) in enumerate(zip(la, lb)) if u != v), min(len(la), len(lb)))
            ud = list(difflib.unified_diff(la, lb, "a", "b", n=2, lineterm=""))
            diffs.append({"first_differing_op": first, "n_diff_lines": len(ud),
                          "name_count_delta": {k: ny[k] - nx[k] for k in set(nx) | set(ny) if ny[k] != nx[k]},
                          "diff_head": ud[:limit]})
        out["diffs"] = diffs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--before-commit", required=True)
    ap.add_argument("--after-commit", required=True)
    ap.add_argument("--repo", required=True, help="a git checkout that holds both commits")
    ap.add_argument("--inputs", required=True, help="where fixtures are read from, for both arms")
    ap.add_argument("--card", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    ap.add_argument("--holder", default="worker:of3t-infab")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trees = {"before": Path(a.before), "after": Path(a.after)}
    changed = subprocess.run(["git", "-C", a.repo, "diff", "--name-only", a.before_commit,
                              a.after_commit, "--", "tt_bio"], capture_output=True, text=True,
                             check=True).stdout.split()
    changed_mods = sorted({p[:-3].replace("/", ".").removesuffix(".__init__")
                           for p in changed if p.endswith(".py")})
    outp = Path(a.out)
    report = json.loads(outp.read_text()) if outp.exists() else {}
    report.update({"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
                   "card": int(a.card),
                   "commits": {"before": a.before_commit, "after": a.after_commit},
                   "changed_modules": changed_mods})
    report.setdefault("models", {})
    for m in a.models.split(","):
        prev = report["models"].get(m)
        if prev and prev.get("all_folds_succeeded"):
            print(f"INFAB {m} already complete, skipping", flush=True)
            continue
        runs = []
        for i, side in enumerate(("before", "after", "before", "after")):
            r = fold(a.python, trees[side], Path(a.inputs), m, Path(a.workdir) / m / f"{side}{i}",
                     a.card, a.holder)
            r["side"] = side
            runs.append(r)
            print(f"INFAB {m} {side}{i} rc={r['rc']} {r['seconds']}s runtime_s={r['runtime_s']} "
                  f"clk={r['aiclk_mhz_sampled_DURING']} ops={[p['n_ops'] for p in r['trace']['processes']]} "
                  f"{r['digests']}", flush=True)
            if r["rc"] != 0:
                print(r["tail"][-1500:], flush=True)
                break
        ok = len(runs) == 4 and all(r["rc"] == 0 for r in runs)
        key = MODELS[m]["key"]
        fold_procs = [[p for p in r["census"] if key in p.get("tt_bio_modules", [])] for r in runs]
        d = [json.dumps(r["digests"], sort_keys=True) for r in runs]
        fp = [p for ps in fold_procs for p in ps]
        imported_changed = sorted({x for p in fp for x in p.get("tt_bio_modules", [])} & set(changed_mods))
        fired = [(p.get("EXACT_SOFTMAX_STATS") or {}, p.get("EXACT_LAYER_NORM_STATS") or {}) for p in fp]
        report["models"][m] = {
            "argv": runs[0]["cmd"], "all_folds_succeeded": ok,
            "census_saw_every_fold": all(fold_procs) and len(fold_procs) == 4,
            "tree_loaded_per_fold": [sorted({str(p.get("tt_bio_file")) for p in ps}) for ps in fold_procs],
            "imports_changed_modules": imported_changed,
            "digest_nonempty": ok and all(v for v in runs[0]["digests"].values()) and bool(runs[0]["digests"]),
            "AA_before_identical": ok and d[0] == d[2],
            "AA_after_identical": ok and d[1] == d[3],
            "AB_identical": ok and d[0] == d[1] == d[2] == d[3],
            "digests": runs[0]["digests"],
            "autograd_imported_any": any(p.get("autograd_imported") for r in runs for p in r["census"]),
            "taped_ttnn_imported_any": any(p.get("taped_ttnn_imported") for r in runs for p in r["census"]),
            "train_modules_imported_any": sorted({x for r in runs for p in r["census"]
                                                  for x in p.get("train_modules", [])}),
            "exact_counters_all_zero": all(not any(s.values()) and not any(l.values()) for s, l in fired),
            "input_atom_encoder_built_total": sum(p.get("input_atom_encoder_built", 0) for p in fp),
            "trace": ({"AA_before": trace_diff(runs[0], runs[2]), "AA_after": trace_diff(runs[1], runs[3]),
                       "AB": trace_diff(runs[0], runs[1]), "AB2": trace_diff(runs[2], runs[3])}
                      if ok else None),
            "seconds": {r["side"] + str(i): r["seconds"] for i, r in enumerate(runs)},
            "runtime_s": {r["side"] + str(i): r["runtime_s"] for i, r in enumerate(runs)},
            "aiclk": {r["side"] + str(i): r["aiclk_mhz_sampled_DURING"] for i, r in enumerate(runs)},
            "runs": [{k: v for k, v in r.items() if k != "census"} | {"census": r["census"]} for r in runs],
        }
        t = report["models"][m]["trace"]
        print(f"INFAB {m} DONE ok={ok} AA={report['models'][m]['AA_before_identical']},"
              f"{report['models'][m]['AA_after_identical']} AB={report['models'][m]['AB_identical']} "
              f"trace={ {k: v['identical'] for k, v in t.items()} if t else None} "
              f"changed={len(imported_changed)} autograd={report['models'][m]['autograd_imported_any']}",
              flush=True)
        outp.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
