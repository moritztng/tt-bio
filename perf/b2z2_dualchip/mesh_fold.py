"""The full Boltz-2 512 aa cell fold, run on BOTH Blackhole chips of the p300c.

Before a fold can be SHARDED across the pair, the whole model has to RUN on the pair. This is that
step, and on its own it is the plumbing the sharded fold will be a delta against:

  mesh    a 1x2 MeshDevice over one p300c board, every tensor REPLICATED. Both chips execute the
          identical fold in lockstep, so the wall clock is what one chip takes and the CIF must be
          bit-identical to the single-chip control. Anything else means the mesh changed the math.
  single  the ordinary one-chip path, same protocol, as that control.

The device is injected at `_open_device_locked` and that injection reimplements the function
faithfully rather than replacing it with a bare open: it does three things, not one, and two of them
matter. `_configure_active_compute_grid` sets the CORE_GRID_MAIN the model is tuned against, and
`enable_program_cache()` is what stops every op recompiling on every call.

`tt_baseline.build_fold` cannot be used: its cfg omits `conf_kwargs`, so `load_model` dies with a
KeyError on boltz2. The cfg below is the one `perf/b2z2_compose/ab_compose.py` folds with.
"""

import hashlib
import json
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "mesh"
REPS = int(os.environ.get("FOLD_REPS", "3"))
# Trace the diffusion loop. On ONE chip b2z2-diffusion-loop-attack measured this at 0.9948x,
# i.e. no help, because the loop was already 93.8 % device-bound and the dispatch it removes
# was already overlapped. On a MESH the same dispatch costs more: the fold-level mesh tax is
# 1.0538x against a block-level 1.0044x, and the difference is per-program dispatch over ~250k
# programs, of which the diffusion step is 1066 x 200 = 213k. So trace should pay here even
# though it did not there, and that is worth a measurement rather than an assumption.
TRACE = os.environ.get("FOLD_TRACE", "0") == "1"
ROOT = Path("/home/ttuser/.coworker/wt/b2z2-dual-chip-fold")
OUT_PATH = Path(os.environ.get("FOLD_OUT", f"/tmp/b2z2_fold_{MODE}.json"))
sys.path.insert(0, str(ROOT))

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0

import torch  # noqa: E402,F401
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


if MODE == "mesh":
    def _mesh_open(device_id, kwargs):
        with T._device_init_lock():
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            log(f"opening 1x2 mesh, kwargs={kwargs}")
            dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **kwargs)
            T._configure_active_compute_grid(dev)
            dev.enable_program_cache()
            return dev
    T._open_device_locked = _mesh_open


def build_cfg(msa_dir, struct_dir):
    return dict(
        model="boltz2", fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
        conf_kwargs=dict(
            predict_args={"recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                          "diffusion_samples": DIFFUSION_SAMPLES, "max_parallel_samples": 5},
            diffusion_process_args={
                "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                "alignment_reverse_diff": True, "synchronize_sigmas": True},
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": False, "num_subsampled_msa": 1024,
                      "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                      "msa_dropout": 0.15, "z_dropout": 0.25, "pairwise_head_width": 32,
                      "pairwise_num_heads": 4, "activation_checkpointing": True},
            steering_args={"fk_steering": False, "physical_guidance_update": False,
                           "contact_guidance_update": True, "num_particles": 3,
                           "fk_lambda": 4.0, "fk_resampling_interval": 3, "num_gd_steps": 20},
            use_kernels=True, use_tenstorrent=True, trace=False, diffusion_trace=TRACE,
        ),
    )


from tt_bio.main import _read_bio_chains  # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts  # noqa: E402

fix = ROOT / "perf" / "size512" / "fixtures"
tgt, a3m = fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m"
msa_dir = Path(f"/tmp/b2z2_msa512_{MODE}")
struct_dir = Path(f"/tmp/b2z2_struct_{MODE}")
struct_dir.mkdir(parents=True, exist_ok=True)

seq = _read_bio_chains(tgt)[0][1]
rows = a3m.read_text().split("\n")
assert rows[1] == seq, "a3m query row does not match the target sequence"
msa_dir.mkdir(parents=True, exist_ok=True)
(msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m.read_text())

OUT = {"mode": MODE, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "protocol": {"fixture": "perf/size512/fixtures/cdk2x2_512.yaml + its a3m",
                    "recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                    "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED},
       "diffusion_trace": None,  # set below
       "benchlocked": True,
       }


def dump():
    OUT_PATH.write_text(json.dumps(OUT, indent=1))


cfg = build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)
t0 = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z2-dual-chip-fold", cfg)
OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
dev = T.get_device()
OUT["device"] = str(dev)
OUT["n_devices"] = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
OUT["core_grid_main"] = str(T.CORE_GRID_MAIN)
OUT["diffusion_trace"] = TRACE
OUT["trace_region_size"] = T.trace_region_size()
log(f"device={OUT['device']} n_devices={OUT['n_devices']} grid={OUT['core_grid_main']} "
    f"load={OUT['model_load_s']}s")
dump()


def fold_once():
    for p in struct_dir.glob("*"):
        p.unlink() if p.is_file() else shutil.rmtree(p)
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    metrics, _b, _f = state.predict_one(tgt, cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t
    cifs = sorted(hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(struct_dir.glob("*.cif")))
    return wall, metrics, cifs


w, m, c = fold_once()
OUT["cold_s"] = round(w, 3)
log(f"cold {w:.3f}s plddt={m.get('plddt')} cif={c[0][:16] if c else 'NONE'} (discarded)")
dump()

rows_out = []
for i in range(REPS):
    w, m, c = fold_once()
    rows_out.append({"fold_s": round(w, 4), "plddt": m.get("plddt"), "cif": c})
    log(f"rep {i+1}/{REPS}  {w:.4f}s  plddt={m.get('plddt')}  cif={c[0][:16] if c else 'NONE'}")
    OUT["reps_done"] = rows_out
    dump()

times = [r["fold_s"] for r in rows_out]
digests = sorted({d for r in rows_out for d in r["cif"]})
OUT["summary"] = {
    "median_s": round(st.median(times), 4), "min_s": min(times), "max_s": max(times),
    "plddt": sorted({r["plddt"] for r in rows_out}),
    "cif_sha256": digests, "cif_sha256_16": sorted({d[:16] for d in digests}),
    "bit_identical_across_reps": len(digests) == 1,
}
dump()
log(json.dumps(OUT["summary"], indent=1))
log(f"wrote {OUT_PATH}")
os._exit(0)
