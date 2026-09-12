#!/usr/bin/env python3
"""A 512 aa Boltz-2 fold with its start and end timestamps written out, so a sysfs clock trace can
be sliced to the fold rather than to the process.

The campaign's two roofs (429.9 GB/s streaming, 85.96 TFLOP/s dense bf16) were fitted from short
microbenchmarks. If the part boosts for a microbenchmark and drops back under a 24-second fold, both
roofs are overstated by that ratio and part of the "2.34x deficit" is arithmetic against a clock the
chip never held. Nothing here measures the clock itself; it records when the fold ran.
"""
import hashlib, json, os, shutil, statistics as st, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as TT                                               # noqa: E402

FIX = ROOT / "perf" / "size512" / "fixtures"
RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0
REPS = int(os.environ.get("B2Z_REPS", "3"))
OUT_PATH = Path(sys.argv[1])
FIXTURE = sys.argv[2] if len(sys.argv) > 2 else "cdk2x2_512"

sys.path.insert(0, str(ROOT / "perf" / "b2x-flag-levers"))
from ab_flag_levers import _seed_msa, build_cfg                                # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts                # noqa: E402

work = Path(tempfile.mkdtemp(prefix="b2z-stackcfg-"))
struct_dir = work / "out"; struct_dir.mkdir(parents=True)
msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
_seed_msa(FIX / f"{FIXTURE}.yaml", (FIX / f"{FIXTURE}.a3m").read_text(), msa_dir)
cfg = build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)

OUT = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "fixture": FIXTURE, "folds": []}

t0 = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z-bh-stack-config", cfg)
OUT["model_load_s"] = round(time.perf_counter() - t0, 3)

dev = TT.get_device()

# Price the core axis. The extra Tensix column that ETH dispatch would hand back cannot be tested
# directly on this wheel, so measure the same axis downward instead: hold everything else fixed and
# give the model fewer columns. A fold whose time is linear in core count says the column is worth
# taking; a flat fold says the whole dispatch-core chase is dead.
if os.environ.get("B2Z_GRID_X"):
    gx = int(os.environ["B2Z_GRID_X"])
    TT.CORE_GRID_MAIN = ttnn.CoreGrid(y=TT.COMPUTE_GRID_Y, x=gx)
    TT.COMPUTE_GRID_MAIN = (gx, TT.COMPUTE_GRID_Y)
    OUT["grid_forced_x"] = gx

g = dev.compute_with_storage_grid_size()
OUT["compute_grid"] = [g.x, g.y]
OUT["compute_grid_main"] = list(TT.COMPUTE_GRID_MAIN)
OUT["compute_grid_measured"] = bool(TT.COMPUTE_GRID_MEASURED)
OUT["cores"] = TT.COMPUTE_GRID_MAIN[0] * TT.COMPUTE_GRID_MAIN[1]


def one(tag):
    for p in struct_dir.glob("*"):
        p.unlink() if p.is_file() else shutil.rmtree(p)
    ttnn.synchronize_device(dev)
    w0, e0 = time.perf_counter(), time.time()
    state.predict_one(FIX / f"{FIXTURE}.yaml", cfg)
    ttnn.synchronize_device(dev)
    w1, e1 = time.perf_counter(), time.time()
    cifs = sorted(struct_dir.glob("*.cif"))
    assert cifs, "no CIF written"
    return {"tag": tag, "wall_s": round(w1 - w0, 4), "epoch_start": e0, "epoch_end": e1,
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]}


OUT["folds"].append(one("cold"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(OUT, indent=1))
for i in range(REPS):
    OUT["folds"].append(one(f"warm{i}"))
    OUT_PATH.write_text(json.dumps(OUT, indent=1))

warm = [f["wall_s"] for f in OUT["folds"] if f["tag"] != "cold"]
OUT["warm_median_s"] = round(st.median(warm), 4)
OUT_PATH.write_text(json.dumps(OUT, indent=1))
print("RESULT " + json.dumps({k: v for k, v in OUT.items() if k != "folds"}))
for f in OUT["folds"]:
    print(f"  {f['tag']:6s} {f['wall_s']:8.4f} s  cif {f['cif_sha256']}")
