#!/usr/bin/env python3
"""In-process A/B of one ESMFold2 lever on the GAP-table workload (cdk2x2_512, shipped steps, 1 sample).

Builds the model once through tt_baseline.build_fold (the path perf/of3_4xpd/xmodel_ab.py times), then
alternates a module global between folds: one cold fold per arm first (discarded), then --repeat rounds
of arm A, arm B interleaved, so both arms see the same clock and host load. Every fold records its wall,
pLDDT, CIF digest, the opened node's AICLK median during the fold, and loadavg.

    python perf/spd_gap_esmf/ab.py --model esmfold2-fast --alt tt_bio.esmfold2.DIFFUSION_TRACE --out r.json
"""
import argparse, hashlib, importlib, json, os, statistics, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="esmfold2", choices=("esmfold2", "esmfold2-fast"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--alt", required=True, help="module.ATTR flipped between arms; arm A = True, arm B = False")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

import tt_bio.tenstorrent as T
import tt_baseline as B
from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                         _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd: os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
modname, attr = a.alt.rsplit(".", 1)
mod = importlib.import_module(modname)

fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta = B.build_fold(a.model, ROOT / f".msa_ab_{a.model}_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
                              fix / f"cdk2x2_{a.size}.a3m", fast=a.fast)[:2]
sdir = Path(meta["struct_dir"])
nodes = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.realpath(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")} or {-1})
clk = []
def sampler():
    while True:
        try: clk.append((time.monotonic(), int(Path(f"/sys/class/tenstorrent/tenstorrent!{nodes[0]}/tt_aiclk").read_text().split()[0])))
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=sampler, daemon=True).start()

res = {"model": a.model, "fast": a.fast, "alt": a.alt, "nodes": nodes, "trace_region": T.trace_region_size(),
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "folds": []}
def fold(arm, tag):
    setattr(mod, attr, arm == "A")
    t0 = time.monotonic(); s, m = one_fold(); t1 = time.monotonic()
    c = sorted(v for t, v in clk if t0 <= t <= t1)
    rec = {"arm": arm, "tag": tag, "fold_s": round(s, 3), "plddt": m.get("plddt"),
           "cif": [hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(sdir.glob("*.cif"))],
           "aiclk_median": c[len(c) // 2] if c else None, "aiclk_min": c[0] if c else None,
           "loadavg": open("/proc/loadavg").read().split()[0]}
    res["folds"].append(rec); print(json.dumps(rec), flush=True)
    a.out.write_text(json.dumps(res, indent=1))
fold("A", "cold"); fold("B", "cold")
for i in range(a.repeat):
    fold("A", f"warm{i}"); fold("B", f"warm{i}")
for arm in "AB":
    w = sorted(f["fold_s"] for f in res["folds"] if f["arm"] == arm and f["tag"] != "cold")
    res[f"median_{arm}"], res[f"spread_{arm}"] = statistics.median(w), round(w[-1] - w[0], 3)
for name in ("DIFFUSION_TRACE_STATS",):
    import tt_bio.esmfold2 as E
    res[name] = list(getattr(E, name, []))
a.out.write_text(json.dumps(res, indent=1))
print(f"A {res['median_A']:.3f}  B {res['median_B']:.3f}  delta {res['median_B'] - res['median_A']:+.3f} s", flush=True)
T.cleanup()
