"""The 512 aa fold on BOTH Blackhole processors of one p300c, with the trunk shard and the
atom-axis shard switched on and off INSIDE ONE PROCESS, arms interleaved.

Two rows measured the halves and neither composed them. The trunk shard
(`b2z2-dual-chip-fold`) is bit-exact on the cell's own silicon; the atom-axis shard
(`b2z2-atom-axis-shard`) is bit-exact on Wormhole and was never wired into a fold. This runs
the union.

Why one process, and why interleaved. A mesh fold and a one-chip fold cannot share a process --
tt_bio opens one device context -- so the honest A/B for a SHARD is sharded-vs-replicated on the
SAME mesh, in the same session, arms alternating, which is what this does. The card-level
one-processor number comes from a separate `--arms base --mesh 1` run and is quoted as what it
is. Interleaving is not cosmetic: qb2 carries sibling load all evening, and an arm block measured
before another arm block charges that drift to the lever.

Every rep records its CIF sha256 AND the four kernel-config rung caches. An op-level bit-exactness
suite cannot certify a shard: all five trunk ops passed `torch.equal` at their own shape while the
composed fold wrote a different CIF, because an op picks its kernel configs from the shape it is
given and a shard changes L1 pressure enough to step a config down a rung. The digest is the gate.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0

ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True, help="comma list, e.g. base,trunk,base,trunk")
ap.add_argument("--mesh", type=int, default=2)
ap.add_argument("--trace", type=int, default=1)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()
ARMS = [a for a in args.arms.split(",") if a]
assert set(ARMS) <= {"base", "trunk", "atom", "both"}, ARMS

import torch  # noqa: E402,F401
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

assert Path(T.__file__).resolve().parents[1] == ROOT, f"wrong tt_bio: {T.__file__}"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


MESH = args.mesh
if MESH > 1:
    def _mesh_open(device_id, kwargs):
        with T._device_init_lock():
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            dev = ttnn.open_mesh_device(ttnn.MeshShape(1, MESH), **kwargs)
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
            use_kernels=True, use_tenstorrent=True, trace=False,
            diffusion_trace=bool(args.trace),
        ),
    )


from tt_bio.main import _read_bio_chains  # noqa: E402
from tt_bio.worker import _WorkerState, _ensure_local_artifacts  # noqa: E402

fix = ROOT / "perf" / "size512" / "fixtures"
tgt, a3m = fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m"
sfx = args.tag or "u"
msa_dir = Path(f"/tmp/b2z2_bs_msa_{sfx}")
struct_dir = Path(f"/tmp/b2z2_bs_struct_{sfx}")
struct_dir.mkdir(parents=True, exist_ok=True)
msa_dir.mkdir(parents=True, exist_ok=True)
seq = _read_bio_chains(tgt)[0][1]
rows = a3m.read_text().split("\n")
assert rows[1] == seq, "a3m query row does not match the target sequence"
(msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m.read_text())

OUT = {"tag": args.tag, "arms_requested": ARMS, "mesh": MESH, "trace": bool(args.trace),
       "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
       "protocol": {"fixture": "perf/size512/fixtures/cdk2x2_512.yaml + its a3m",
                    "recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                    "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED},
       "benchlocked": os.environ.get("FOLD_BENCHLOCKED") == "1",
       "loadavg_start": open("/proc/loadavg").read().split()[:3],
       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reps": []}
args.out.parent.mkdir(parents=True, exist_ok=True)


def dump():
    args.out.write_text(json.dumps(OUT, indent=1))


cfg = build_cfg(msa_dir, struct_dir)
_ensure_local_artifacts(cfg)
t0 = time.perf_counter()
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("b2z2-p300c-both-shards", cfg)
OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
dev = T.get_device()
OUT["n_devices"] = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
OUT["core_grid_main"] = str(T.CORE_GRID_MAIN)
OUT["arch"] = T.arch_name()
assert OUT["n_devices"] == MESH, f"opened {OUT['n_devices']} devices, wanted {MESH}"
log(f"n_devices={OUT['n_devices']} arch={OUT['arch']} grid={OUT['core_grid_main']} "
    f"load={OUT['model_load_s']}s trace={bool(args.trace)}")
dump()

# Both shards are module-level gates in tt_bio.tenstorrent, read at the call site rather than
# captured at import, so an arm is one assignment and no arm needs its own process.
assert hasattr(T, "_ATOM_ROW_SHARD"), "this tree has no atom-axis shard gate"
assert hasattr(T, "_ROW_SHARD_FOLD"), "this tree has no trunk row-shard gate"


def set_arm(arm):
    T._ROW_SHARD_FOLD = arm in ("trunk", "both")
    T._ATOM_ROW_SHARD = arm in ("atom", "both")


class Splitter:
    """Stage boundaries off the model's own progress callback: the first `diffusion` callback
    ends the trunk, the first `confidence` one ends the sampler."""

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

    def close(self, t0):
        self._mark("end")
        out, prev = {}, t0
        for lab, t in self.marks:
            out[lab] = round(t - prev, 4)
            prev = t
        out["trunk_s"] = out.pop("trunk_end", None)
        out["sampler_s"] = out.pop("sampler_end", None)
        return out


def rungs():
    return {n: (len(getattr(T, n)) if hasattr(T, n) else None)
            for n in ("_L1_OUT_RUNG", "_BMM_CFG_RUNG", "_BMM_CFG_REFUSED", "_OPM_JOIN_REFUSED")}


def fold_once(arm):
    for p in struct_dir.glob("*"):
        p.unlink() if p.is_file() else shutil.rmtree(p)
    set_arm(arm)
    sp = Splitter()
    state.pfn = sp
    state.model.progress_fn = sp
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    metrics, _b, _f = state.predict_one(tgt, cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t
    stages = sp.close(t)
    cifs = sorted(hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted(struct_dir.glob("*.cif")))
    return {"arm": arm, "fold_s": round(wall, 4), "plddt": metrics.get("plddt"),
            "cif": [c[:16] for c in cifs], "stages": stages, "rungs": rungs(),
            "loadavg": open("/proc/loadavg").read().split()[:3]}


# A cold fold per DISTINCT arm, all discarded: the first fold of an arm compiles its programs and
# populates the config-rung caches, and a cold fold in the timed set is a lever-shaped artifact.
OUT["cold"] = []
for arm in dict.fromkeys(ARMS):
    r = fold_once(arm)
    OUT["cold"].append(r)
    log(f"cold[{arm}] {r['fold_s']:.3f}s cif={r['cif']} rungs={r['rungs']}")
    dump()

for i, arm in enumerate(ARMS):
    r = fold_once(arm)
    OUT["reps"].append(r)
    log(f"rep {i+1}/{len(ARMS)} [{arm}] {r['fold_s']:.4f}s trunk={r['stages'].get('trunk_s')} "
        f"sampler={r['stages'].get('sampler_s')} cif={r['cif']} load={r['loadavg'][0]}")
    dump()


def med(xs):
    return round(st.median(xs), 4) if xs else None


summ = {}
for arm in dict.fromkeys(ARMS):
    rs = [r for r in OUT["reps"] if r["arm"] == arm]
    summ[arm] = {
        "n": len(rs), "fold_median_s": med([r["fold_s"] for r in rs]),
        "fold_min_s": min([r["fold_s"] for r in rs]), "fold_max_s": max([r["fold_s"] for r in rs]),
        "trunk_median_s": med([r["stages"]["trunk_s"] for r in rs if r["stages"].get("trunk_s")]),
        "sampler_median_s": med([r["stages"]["sampler_s"] for r in rs
                                 if r["stages"].get("sampler_s")]),
        "cif": sorted({c for r in rs for c in r["cif"]}),
        "plddt": sorted({r["plddt"] for r in rs}),
    }
OUT["summary"] = summ
OUT["all_cif"] = sorted({c for r in OUT["reps"] for c in r["cif"]})
OUT["bit_identical_across_all_arms"] = len(OUT["all_cif"]) == 1
OUT["config_rungs_end"] = rungs()
OUT["config_rungs_detail"] = {
    n: sorted(map(str, getattr(T, n)))[:12] if hasattr(T, n) else None
    for n in ("_L1_OUT_RUNG", "_BMM_CFG_RUNG", "_BMM_CFG_REFUSED", "_OPM_JOIN_REFUSED")}
if "base" in summ:
    b = summ["base"]["fold_median_s"]
    OUT["ratios_vs_base"] = {a: round(b / summ[a]["fold_median_s"], 5) for a in summ}
    OUT["trunk_stage_ratios_vs_base"] = {
        a: (round(summ["base"]["trunk_median_s"] / summ[a]["trunk_median_s"], 5)
            if summ[a]["trunk_median_s"] and summ["base"]["trunk_median_s"] else None)
        for a in summ}
    OUT["sampler_stage_ratios_vs_base"] = {
        a: (round(summ["base"]["sampler_median_s"] / summ[a]["sampler_median_s"], 5)
            if summ[a]["sampler_median_s"] and summ["base"]["sampler_median_s"] else None)
        for a in summ}
OUT["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
dump()
log(json.dumps({k: OUT[k] for k in ("summary", "all_cif", "bit_identical_across_all_arms",
                                    "config_rungs_end") if k in OUT}, indent=1))
if "ratios_vs_base" in OUT:
    log("ratios vs base: " + json.dumps(OUT["ratios_vs_base"]))
log(f"wrote {args.out}")
os._exit(0)
