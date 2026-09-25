#!/usr/bin/env python3
"""of3t-confpfe: inference before/after the D266 fix, from firing and digest (of3t-stackship infab.py).

    infab.py --before TREE --after TREE --card 1 --workdir W --out INFAB.json

Per model, folds interleaved BEFORE, AFTER, BEFORE, AFTER on one card: the two BEFORE folds are
the A/A floor, BEFORE vs AFTER the A/B. of3t-verbinstall's `infaa.py` census, taken inside the
folding process at exit (D236), extended to this row's counters:

  autograd_imported        whether `tt_bio.autograd` is in the folding process at all
  EXACT_*_STATS            the exact softmax / layer norm counters, read WITHOUT importing it
  raw_ops                  which callable `ttnn.softmax` and `ttnn.layer_norm` are at exit

  train_modules            every `tt_bio.train.*` module in the process
  input_atom_encoder_built how many `openfold3.InputAtomEncoder` (D263's training encoder) were built

`tt_bio.main predict` folds in a spawned worker that the CLI terminates at shutdown, so an atexit
hook never fires there and a stdout census reads only the CLI process, which never imports the
model (of3t-ieatom found stackship's census was that). Every process therefore also rewrites its
census to `$INFAB_CENSUS_DIR/<pid>.json` once a second, and the report requires a process that
imported `tt_bio.openfold3`, so a census that missed the folding process cannot pass.

The non-zero control for the same counters is `STACK_SHIP_SHIP.json`, the training arm.
AICLK is sampled from tt-smi every 3 s DURING each fold and stamped beside its wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

SMI = "/home/ttuser/.local/bin/tt-smi"
SITE = '''import atexit, json, sys
def _dump():
    out = {"pid": __import__("os").getpid()}
    mod = sys.modules.get("tt_bio.autograd")
    out["autograd_imported"] = mod is not None
    for n in ("EXACT_SOFTMAX_STATS", "EXACT_LAYER_NORM_STATS"):
        out[n] = dict(getattr(mod, n)) if mod is not None and hasattr(mod, n) else None
    t = sys.modules.get("ttnn")
    out["raw_ops"] = ({n: getattr(getattr(t, n, None), "__qualname__", repr(getattr(t, n, None)))
                       for n in ("softmax", "layer_norm")} if t is not None else None)
    out["taped_ttnn_imported"] = "tt_bio.taped_ttnn" in sys.modules
    out["train_modules"] = sorted(m for m in sys.modules if m.startswith("tt_bio.train"))
    out["input_atom_encoder_built"] = _BUILT[0]
    out["openfold3_imported"] = "tt_bio.openfold3" in sys.modules
    out["tensor_getitem"] = (getattr(t.Tensor.__getitem__, "__qualname__", None)
                             if t is not None else None)
    out["argv"] = sys.argv[:3]
    out["imports_seen"] = sorted(_SEEN)
    return out
def _print():
    print("STACKSHIP_CENSUS " + json.dumps(_dump()), flush=True)
atexit.register(_print)
import importlib.abc, importlib.machinery, os
_SEEN = set()
def _write():
    d = os.environ.get("INFAB_CENSUS_DIR")
    if d:
        p = os.path.join(d, f"{os.getpid()}.json")
        with open(p + f".{__import__('threading').get_ident()}.tmp", "w") as f:
            json.dump(_dump(), f)
            tmp = f.name
        os.replace(tmp, p)
# Every import of a watched module is recorded and written at once, so the census does not depend
# on the process exiting cleanly or on a poll getting the GIL.
_WATCH = ("tt_bio.train", "tt_bio.autograd", "tt_bio.taped_ttnn", "tt_bio.openfold3")
class _Seen(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.startswith(_WATCH) and name not in _SEEN:
            _SEEN.add(name)
            _write()
        return None
# Count InputAtomEncoder constructions: wrap __init__ as tt_bio.openfold3 finishes executing.
_BUILT = [0]
class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "tt_bio.openfold3":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        run = spec.loader.exec_module
        def exec_module(m):
            run(m)
            if not hasattr(m, "InputAtomEncoder"):
                return
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
import threading, time
def _snap():
    while os.environ.get("INFAB_CENSUS_DIR"):
        _write()
        time.sleep(1)
threading.Thread(target=_snap, daemon=True).start()
'''


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clock(card: int, stop: threading.Event, out: list):
    while not stop.is_set():
        try:
            d = json.loads(subprocess.run([SMI, "-s"], capture_output=True, text=True,
                                          timeout=20).stdout)
            out.append(int(d["device_info"][card]["telemetry"]["aiclk"].strip()))
        except Exception:
            pass
        stop.wait(3)


def fold(py, tree: Path, spec, out: Path, card: str, holder: str = "worker:of3t-stackship"):
    subprocess.run(["rm", "-rf", str(out)], check=False)
    out.mkdir(parents=True)
    sc = out.parent / ("_sc_" + out.name)
    sc.mkdir(parents=True, exist_ok=True)
    (sc / "sitecustomize.py").write_text(SITE)
    cdir = sc / "census"
    subprocess.run(["rm", "-rf", str(cdir)], check=False)
    cdir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_BIO_")}
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": holder, "OMP_NUM_THREADS": "8",
                "PYTHONPATH": str(sc) + os.pathsep + str(tree), "INFAB_CENSUS_DIR": str(cdir)})
    if spec["kind"] == "predict":
        cmd = [py, "-m", "tt_bio.main", "predict", str(tree / spec["fixture"]), "--model",
               spec["model"], "--single_sequence", "--sampling_steps", "6",
               "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out)]
    else:
        cmd = [py, str(tree / "scripts/af2_port/fold_timing.py"), "--pdb",
               str(tree / spec["fixture"]), "--reps", "2", "--params", spec["params"],
               "--out", str(out / "af2.json")]
    samples, stop = [], threading.Event()
    th = threading.Thread(target=clock, args=(int(card), stop, samples), daemon=True)
    th.start()
    t = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    secs = time.time() - t
    stop.set(); th.join()
    census = [json.loads(p.read_text()) for p in sorted(cdir.glob("*.json"))]
    if spec["kind"] == "predict":
        digests = {c.name: sha(c) for c in sorted(out.rglob("*.cif"))}
    else:
        rep = json.loads((out / "af2.json").read_text()) if (out / "af2.json").exists() else {}
        digests = {"structure_sha16_all": rep.get("structure_sha16_all")}
    s = sorted(samples)
    return {"rc": r.returncode, "seconds": round(secs, 2), "digests": digests,
            "census": census,
            "aiclk_mhz_sampled_DURING": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                          "max": s[-1]} if s else None),
            "tail": (r.stdout + r.stderr)[-2000:] if r.returncode else ""}


MODELS = {
    "openfold3": {"kind": "predict", "model": "openfold3",
                  "fixture": "perf/size512/fixtures/cdk2x2_128.yaml"},
    "protenix-v2": {"kind": "predict", "model": "protenix-v2",
                    "fixture": "perf/size512/fixtures/cdk2x2_128.yaml"},
    "af2-ig": {"kind": "af2",
               "fixture": "scripts/af2_port/parity_artifacts/designpop_bg119/binder_complex.pdb",
               "params": "/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    ap.add_argument("--quiet", default="")
    ap.add_argument("--holder", default="worker:of3t-stackship")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trees = {"before": Path(a.before), "after": Path(a.after)}
    commits = {k: subprocess.run(["git", "-C", str(v), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip() for k, v in trees.items()}
    report = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
              "card": int(a.card), "commits": commits, "host_quiet": a.quiet, "models": {}}
    for m in a.models.split(","):
        runs = []
        for i, side in enumerate(("before", "after", "before", "after")):
            r = fold(a.python, trees[side], MODELS[m], Path(a.workdir) / m / f"{side}{i}", a.card,
                     a.holder)
            r["side"] = side
            runs.append(r)
            print(f"INFAB {m} {side} rc={r['rc']} {r['seconds']}s clk={r['aiclk_mhz_sampled_DURING']} "
                  f"{r['digests']}", flush=True)
        ok = all(r["rc"] == 0 for r in runs)
        saw = all(any("tt_bio.openfold3" in p.get("imports_seen", []) for p in r["census"])
                  for r in runs)
        d = [json.dumps(r["digests"], sort_keys=True) for r in runs]
        cen = [p for r in runs for p in r["census"]]
        fired = [(p.get("EXACT_SOFTMAX_STATS") or {}, p.get("EXACT_LAYER_NORM_STATS") or {})
                 for p in cen]
        report["models"][m] = {
            "fixture": MODELS[m]["fixture"], "all_folds_succeeded": ok,
            "AA_before_identical": ok and d[0] == d[2],
            "AA_after_identical": ok and d[1] == d[3],
            "AB_identical": ok and d[0] == d[1] == d[2] == d[3],
            "digests": runs[0]["digests"],
            "autograd_imported_any": any(p.get("autograd_imported") for p in cen),
            "watched_imports_seen": sorted({x for p in cen for x in p.get("imports_seen", [])}),
            "train_modules_imported_any": sorted({x for p in cen for x in p.get("train_modules", [])}),
            "input_atom_encoder_built_total": sum(p.get("input_atom_encoder_built", 0) for p in cen),
            "census_folding_processes": sum("tt_bio.openfold3" in p.get("imports_seen", []) for p in cen),
            "census_saw_every_fold": saw,
            "exact_counters_all_zero": all(not any(s.values()) and not any(l.values())
                                           for s, l in fired),
            "raw_ops_at_exit": sorted({json.dumps(p.get("raw_ops")) for p in cen}),
            "taped_ttnn_imported_any": any(p.get("taped_ttnn_imported") for p in cen),
            "tensor_getitem_at_exit": sorted({str(p.get("tensor_getitem")) for p in cen}),
            "census_processes": len(cen),
            "seconds": {r["side"] + str(i): r["seconds"] for i, r in enumerate(runs)},
            "aiclk": {r["side"] + str(i): r["aiclk_mhz_sampled_DURING"] for i, r in enumerate(runs)},
            "runs": runs,
        }
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
