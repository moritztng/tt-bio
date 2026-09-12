"""The three fold arms paired and interleaved in ONE process, with the floor of their own statistic.

`b2z2-shard-replication-attack` ran one arm per process, n=1, and was right to claim no timing from
it: 27.51 / 22.14 / 21.97 s of trunk are three numbers from three sessions on a shared box. This
file interleaves instead.

The parent could not interleave because `_ROW_SHARD_FOLD` and `_B_SHARD_FOLD` are read at import,
which is the trap that cost it a whole run: setting the env var after `import tt_bio` folded every
arm unsharded under a sharded name and produced three matching digests that certified nothing. The
flags are module globals, so the arm is selected by assigning them BETWEEN folds -- which is both
the fix and what makes pairing possible. `ROW_SHARD_CALLS` is read per fold and asserted per fold,
so an arm that did not engage fails the rep rather than the run.

  mesh     1x2 MeshDevice, every tensor replicated, no shard. Every chip computes the identical
           fold, so this is the one-processor denominator PLUS the mesh tax, and it is the
           denominator both ratios below are taken against. The genuinely single-chip number needs
           its own process and is NOT comparable rep-for-rep; run `perf/b2z2_shardrep/fold_digest.py
           single` for it and quote it as a separate session.
  shard    row shard on the i axis.
  bshard   the same, with the `b` role of both triangle products split too.

A/A: `mesh` is folded TWICE per rep, at the two ends of the rep, and the ratio of those two is the
floor of exactly the statistic the arms are quoted in -- a median of paired folds in one session.
A lever ratio inside that band is not a measurement. Quote the floor of the statistic you report.

    FOLD_REPS=7 TT_VISIBLE_DEVICES=28,29 TT_BIO_LEASE_CARDS=28,29 \
    TT_BIO_LEASE_HOLDER=worker:b2z2-bshard-timing AB_OUT=... \
    PYTHONPATH=$PWD python3 perf/b2z2_bshardtime/fold_ab_interleaved.py
"""

import hashlib
import json
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

REPS = int(os.environ.get("FOLD_REPS", "7"))
ROOT = Path(os.environ.get("FOLD_ROOT", "/home/tt-admin/wt-bshardtime"))
OUT_PATH = Path(os.environ.get("AB_OUT", "/tmp/b2z2_bshard_ab.json"))
MESH_N = int(os.environ.get("MESH_N", "2"))
sys.path.insert(0, str(ROOT))

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0

import torch  # noqa: E402,F401
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _mesh_open(device_id, kwargs):
    with T._device_init_lock():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, MESH_N), **kwargs)
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
            use_kernels=True, use_tenstorrent=True, trace=False, diffusion_trace=False,
        ),
    )


from tt_bio.main import _read_bio_chains  # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts  # noqa: E402

fix = ROOT / "perf" / "size512" / "fixtures"
tgt, a3m = fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m"
# Namespaced to this row. Sibling perf campaigns that share a /tmp prefix collide on ownership.
_TMP = os.environ.get("FOLD_TMP", "/tmp/b2z2_bshardtime")
msa_dir, struct_dir = Path(f"{_TMP}/msa512"), Path(f"{_TMP}/struct")
struct_dir.mkdir(parents=True, exist_ok=True)
msa_dir.mkdir(parents=True, exist_ok=True)
seq = _read_bio_chains(tgt)[0][1]
assert a3m.read_text().split("\n")[1] == seq, "a3m query row does not match the target sequence"
(msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m.read_text())

OUT = {"reps": REPS, "mesh_n": MESH_N, "cards": os.environ.get("TT_VISIBLE_DEVICES"),
       "protocol": {"fixture": "perf/size512/fixtures/cdk2x2_512.yaml + its a3m",
                    "recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                    "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED},
       "benchlocked": False,
       "benchlock_note": "whglx is shared with four other live b2z2 rows and has no host-exclusive "
                         "benchlock. The defence is pairing and interleaving inside one process; "
                         "loadavg is recorded per fold and the A/A floor is the guard.",
       "loadavg_at_start": open("/proc/loadavg").read().split()[:3], "folds": []}


def dump():
    OUT_PATH.write_text(json.dumps(OUT, indent=1))


cfg = build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)
t0 = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z2-bshard-timing", cfg)
OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
dev = T.get_device()
OUT["device"], OUT["n_devices"] = str(dev), int(dev.get_num_devices())
assert OUT["n_devices"] == MESH_N, f"opened {OUT['n_devices']} devices, wanted {MESH_N}"
log(f"device={OUT['device']} grid={T.CORE_GRID_MAIN} load={OUT['model_load_s']}s")
dump()


class Splitter:
    """Stage boundaries off the model's own progress callback; the first `diffusion` ends the trunk."""

    def __init__(self):
        self.marks = []

    def _mark(self, label):
        ttnn.synchronize_device(dev)
        self.marks.append((label, time.perf_counter()))

    def __call__(self, stage=None, step=0, total=0, *a, **k):
        if stage == "diffusion" and len(self.marks) < 2:
            self._mark("trunk_end" if not self.marks else "diffusion_conditioning_end")
        elif stage == "confidence":
            self._mark("sampler_end")

    def close(self, t0):
        self._mark("end")
        out, prev = {}, t0
        for lab, t in self.marks:
            out[lab] = round(t - prev, 4)
            prev = t
        out["trunk_s"] = out.pop("trunk_end", None)
        return out


ARMS = {"mesh": (False, False), "shard": (True, False), "bshard": (True, True)}


def rungs():
    return {n: {str(k): v for k, v in getattr(T, n).items()} if isinstance(getattr(T, n), dict)
            else sorted(map(str, getattr(T, n)))
            for n in ("_L1_OUT_RUNG", "_BMM_CFG_RUNG", "_BMM_CFG_REFUSED", "_OPM_JOIN_REFUSED")}


def fold_once(arm):
    """One fold under `arm`, with the engagement counter checked for THIS fold, not for the run."""
    T._ROW_SHARD_FOLD, T._B_SHARD_FOLD = ARMS[arm]
    before = dict(T.ROW_SHARD_CALLS)
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
    stages = sp.close(t)
    cifs = sorted(hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(struct_dir.glob("*.cif")))
    chain = T.ROW_SHARD_CALLS["chain"] - before["chain"]
    bsh = T.ROW_SHARD_CALLS["b_shard"] - before["b_shard"]
    # An absent shard produces a perfect digest and a plausible wall. Fail the fold, not the run.
    if arm == "mesh":
        assert chain == 0 and bsh == 0, f"mesh arm entered the sharded chain: {chain}/{bsh}"
    else:
        assert chain > 0, f"{arm} never entered the sharded chain -- this fold certifies nothing"
        assert bsh == (chain if arm == "bshard" else 0), \
            f"{arm} b_shard count {bsh} against {chain} chain calls"
    rec = {"arm": arm, "fold_s": round(wall, 4), "trunk_s": stages.get("trunk_s"),
           "stages": stages, "plddt": metrics.get("plddt"), "cif": cifs,
           "chain": chain, "b_shard": bsh, "loadavg": os.getloadavg()[0], "rungs": rungs()}
    OUT["folds"].append(rec)
    dump()
    log(f"{arm:6s} fold {wall:7.3f}s trunk {rec['trunk_s']} chain={chain} b={bsh} "
        f"load={rec['loadavg']:.1f} cif={cifs[0][:16] if cifs else 'NONE'}")
    return rec


# Cold fold per arm: it is what populates a rung and what compiles the arm's programs, and its
# time is discarded. Its DIGEST is kept, because cold and warm disagreeing is the signal.
OUT["cold"] = {}
for _a in ARMS:
    _r = fold_once(_a)
    OUT["cold"][_a] = {k: _r[k] for k in ("fold_s", "cif", "rungs")}
dump()

for rep in range(REPS):
    fold_once("mesh")                                    # A/A first half
    for a in (("shard", "bshard") if rep % 2 == 0 else ("bshard", "shard")):
        fold_once(a)
    fold_once("mesh")                                    # A/A second half
    log(f"--- rep {rep+1}/{REPS} done")

warm = OUT["folds"][len(ARMS):]


def series(arm, key):
    return [f[key] for f in warm if f["arm"] == arm and f[key] is not None]


aa = [warm[i * 4 + 3][k] / warm[i * 4][k] for i in range(REPS) for k in ("fold_s",)]
aa_trunk = [warm[i * 4 + 3]["trunk_s"] / warm[i * 4]["trunk_s"] for i in range(REPS)]
digests = sorted({d for f in OUT["folds"] for d in f["cif"]})
OUT["summary"] = {
    "median": {a: {k: round(st.median(series(a, k)), 4) for k in ("fold_s", "trunk_s")}
               for a in ARMS},
    "ratios_vs_mesh": {a: {k: round(st.median(series("mesh", k)) / st.median(series(a, k)), 5)
                           for k in ("fold_s", "trunk_s")} for a in ("shard", "bshard")},
    "bshard_vs_shard": {k: round(st.median(series("shard", k)) / st.median(series(a, k)), 5)
                        for a in ("bshard",) for k in ("fold_s", "trunk_s")},
    # The A/A floor of the SAME statistic: mesh-against-mesh, paired, same session.
    "aa_floor": {"fold_s": {"median": round(st.median(aa), 5), "max": round(max(aa), 5),
                            "min": round(min(aa), 5), "pairs": aa},
                 "trunk_s": {"median": round(st.median(aa_trunk), 5),
                             "max": round(max(aa_trunk), 5), "min": round(min(aa_trunk), 5)}},
    "cif_sha256_16": sorted({d[:16] for d in digests}),
    "bit_identical_across_all_arms_and_reps": len(digests) == 1,
    "loadavg_range": [min(f["loadavg"] for f in warm), max(f["loadavg"] for f in warm)],
}
OUT["row_shard_calls_total"] = dict(T.ROW_SHARD_CALLS)
dump()
log(json.dumps(OUT["summary"], indent=1))
assert len(digests) == 1, f"arms disagree on the fold: {sorted(d[:16] for d in digests)}"
T.cleanup()
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
log(f"mesh closed, fabric down. wrote {OUT_PATH}")
