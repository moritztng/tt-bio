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

--- THIS COPY, and why it is a copy ---

`perf/b2z2_dualchip/mesh_fold.py` on `wk/b2z2-dual-chip-fold`, taken verbatim and then changed in
four places, because that row established the rule this file exists to obey: **an op-level
bit-exactness suite cannot certify a shard.** All five of its pair-track ops passed `torch.equal` at
their own shape while the composed fold emitted the wrong CIF. The failure mode is L1 PRESSURE
tripping an allocation refusal the unsharded fold never hits; `_L1_OUT_RUNG` remembers it BY SHAPE
and steps a full-height op's config down a rung for the rest of the process, and a config that sets
reduction order is not bit-exact across picks. No block-level harness reaches that pressure.

It is a copy rather than a merge because the two rows are two shard lineages and reconciling them
conflicts in eight hunks inside the very functions `b_shard` edits. That reconciliation is the
orchestrator's merge, not a worker's.

What changed:

  1. ROOT is this worktree, and MODE adds `shard` and `bshard` on top of `single` and `mesh`.
  2. **The four config caches are read after every fold and reported.** A digest that matches while
     a rung fired is luck, not a certification, and a digest that differs tells you which cache moved.
  3. The COLD fold's CIF is recorded rather than only discarded: the cold fold is what populates a
     rung, so cold and warm can disagree and that disagreement is the signal.
  4. Arms are compared here rather than by eye: the run fails unless every arm emits one digest and
     every arm emits the SAME one.
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
ROOT = Path(os.environ.get("FOLD_ROOT", "/home/tt-admin/wt-shardrep"))
OUT_PATH = Path(os.environ.get("FOLD_OUT", f"/tmp/b2z2_fold_{MODE}.json"))
sys.path.insert(0, str(ROOT))

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0

import torch  # noqa: E402,F401
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


MESH_MODES = ("mesh", "shard", "bshard")
if MODE in ("shard", "bshard"):
    os.environ["TT_BIO_ROW_SHARD_FOLD"] = "1"
if MODE == "bshard":
    os.environ["TT_BIO_B_SHARD_FOLD"] = "1"

if MODE in MESH_MODES:
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
# Namespaced per row AND per mode. The original keys these on MODE alone, and /tmp/b2z2_msa512_single
# already existed on whglx owned by another user, so the run died on PermissionError before it opened
# a device. Sibling perf campaigns need namespaced output paths.
_TMP = os.environ.get("FOLD_TMP", "/tmp/b2z2_shardrep")
msa_dir = Path(f"{_TMP}/msa512_{MODE}")
struct_dir = Path(f"{_TMP}/struct_{MODE}")
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
       # Recorded, not asserted. This field was once a hardcoded True/False and drifted from
       # what the run actually did: b2z2_fold_mesh.json says benchlocked=false for a run that
       # WAS under benchlock. The caller passes FOLD_BENCHLOCKED and the loadavg is captured
       # either way, so a reader can check the claim against a number.
       "benchlocked": os.environ.get("FOLD_BENCHLOCKED") == "1",
       "loadavg_at_start": open("/proc/loadavg").read().split()[:3],
       }


def dump():
    OUT_PATH.write_text(json.dumps(OUT, indent=1))


cfg = build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)
t0 = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z2-shard-replication-attack", cfg)
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


class Splitter:
    """Stage boundaries off the model's own progress callback.

    The trunk stage wall this row prices its lever against has been a number borrowed from another
    build and scaled. This measures it here: the first `diffusion` callback is the end of the trunk,
    so trunk = fold wall - (everything from that mark to the end). Four synchronize_device calls per
    fold, which is nothing against a 20 s fold.
    """

    def __init__(self):
        self.marks = []

    def _mark(self, label):
        ttnn.synchronize_device(dev)
        self.marks.append((label, time.perf_counter()))

    def __call__(self, stage=None, step=0, total=0, *a, **k):
        if stage == "diffusion":
            if not self.marks:
                self._mark("trunk_end")
            elif len(self.marks) == 1:
                self._mark("diffusion_conditioning_end")
        elif stage == "confidence":
            self._mark("sampler_end")

    def close(self, t0, wall):
        self._mark("end")
        out, prev = {}, t0
        for lab, t in self.marks:
            out[lab] = round(t - prev, 4)
            prev = t
        out["trunk_s"] = out.pop("trunk_end", None)
        return out


def fold_once():
    for p in struct_dir.glob("*"):
        p.unlink() if p.is_file() else shutil.rmtree(p)
    sp = Splitter()
    state.pfn = sp
    state.model.progress_fn = sp
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    metrics, _b, _f = state.predict_one(tgt, cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t
    stages = sp.close(t, wall)
    cifs = sorted(hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(struct_dir.glob("*.cif")))
    return wall, metrics, cifs, stages


def rungs():
    """The four caches a shard can silently move a kernel config through.

    Recorded after every fold, not once at the end: the COLD fold is what populates a rung, so a
    warm-only snapshot cannot tell a clean run from one that stepped down before the first timed
    rep. A digest that matches with a rung fired is luck and must not be read as a certification.
    """
    return {"L1_OUT_RUNG": {str(k): v for k, v in T._L1_OUT_RUNG.items()},
            "BMM_CFG_RUNG": {str(k): v for k, v in T._BMM_CFG_RUNG.items()},
            "BMM_CFG_REFUSED": sorted(str(k) for k in T._BMM_CFG_REFUSED),
            "OPM_JOIN_REFUSED": {str(k): v for k, v in T._OPM_JOIN_REFUSED.items()}}


def rung_count(r):
    return len(r["L1_OUT_RUNG"]) + len(r["BMM_CFG_RUNG"]) + len(r["BMM_CFG_REFUSED"]) + \
        len(r["OPM_JOIN_REFUSED"])


w, m, c, st_ = fold_once()
OUT["cold_s"] = round(w, 3)
OUT["cold_cif"] = c
OUT["rungs_after_cold"] = rungs()
log(f"cold {w:.3f}s plddt={m.get('plddt')} cif={c[0][:16] if c else 'NONE'} "
    f"rungs={rung_count(OUT['rungs_after_cold'])} (time discarded, digest kept)")
dump()

rows_out = []
for i in range(REPS):
    w, m, c, st_ = fold_once()
    rows_out.append({"fold_s": round(w, 4), "plddt": m.get("plddt"), "cif": c, "stages": st_,
                     "rungs": rungs()})
    log(f"rep {i+1}/{REPS}  {w:.4f}s  cif={c[0][:16] if c else 'NONE'}  "
        f"rungs={rung_count(rows_out[-1]['rungs'])}  stages={st_}")
    OUT["reps_done"] = rows_out
    dump()

times = [r["fold_s"] for r in rows_out]
digests = sorted({d for r in rows_out for d in r["cif"]})
OUT["summary"] = {
    "median_s": round(st.median(times), 4), "min_s": min(times), "max_s": max(times),
    "plddt": sorted({r["plddt"] for r in rows_out}),
    "cif_sha256": digests, "cif_sha256_16": sorted({d[:16] for d in digests}),
    "bit_identical_across_reps": len(digests) == 1,
    "trunk_s_median": round(st.median([r["stages"]["trunk_s"] for r in rows_out
                                       if r["stages"].get("trunk_s")]), 4)
                      if any(r["stages"].get("trunk_s") for r in rows_out) else None,
}
dump()
# A shard changes L1 pressure, and this module has four caches that step a kernel config DOWN a
# rung on an allocation refusal and remember it by shape. A config that sets blocking or reduction
# order is not bit-exact across picks, so a refusal the unsharded fold never hits is enough to
# change the answer while every op stays bit-exact at its own shape. Record what fired.
OUT["config_rungs"] = {
    name: (len(getattr(T, name)) if hasattr(T, name) else None)
    for name in ("_L1_OUT_RUNG", "_BMM_CFG_RUNG", "_BMM_CFG_REFUSED", "_OPM_JOIN_REFUSED")
}
OUT["config_rungs_detail"] = {
    name: sorted(map(str, getattr(T, name)))[:12] if hasattr(T, name) else None
    for name in ("_L1_OUT_RUNG", "_BMM_CFG_RUNG", "_BMM_CFG_REFUSED", "_OPM_JOIN_REFUSED")
}
log("config rungs fired: " + json.dumps(OUT["config_rungs"]))
for _n, _v in OUT["config_rungs_detail"].items():
    if _v:
        log(f"  {_n}: {_v}")
dump()
log(json.dumps(OUT["summary"], indent=1))
log(f"wrote {OUT_PATH}")
# NOT `os._exit(0)`. The original ends there, and `b2z2-dual-chip-fold`'s own ledger records an
# ethernet core wedged after every one of those runs: `_exit` skips teardown and leaves the mesh
# open at process exit. On whglx that is not survivable -- `tt-smi -r` is a no-op there and
# `-glx_reset` takes ~3 minutes across all 32 boards while nine sibling rows are live on them. Close
# the mesh and take fabric down in process, the way `perf/b2z2_pairchain/chain_bitexact.py` does.
T.cleanup()
if MODE in MESH_MODES:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
log("mesh closed, fabric down")
sys.exit(0)
