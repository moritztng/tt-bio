"""A/B of two layer-level levers on the stepped gated rollout, one process, arms interleaved.

base : the scatter divisor recomputed in every layer (the path before the lever)
inv  : layers over the same cut share it (head)
g2   : inv, and the fp32 row gather carries 2 bf16 slices (16 mantissa bits) instead of 3
g1   : inv, and the gather carries 1 bf16 slice

Per arm: one warm rollout, then timed rollouts at the same seed. Poses are compared with base:
inv must be bit-identical, g2/g1 report per-pose RMSD in Angstrom.
"""
import argparse, json, statistics, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--edge-tp", default="tile")
ap.add_argument("--ligand", type=int, default=9)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--arms", default="base,inv,g2,g1")
ap.add_argument("--workload", default="/home/ttuser/scratch/dfd/wl/workload")
ap.add_argument("--out", required=True)
a = ap.parse_args()
import torch
import ttnn
from tt_bio.diffdock import device as ddev, esm, input as dfd_input, network, sampling
from tt_bio.diffdock import run as dock_run
from tt_bio.diffdock.job import DockOptions, read_csv_jobs
o = DockOptions(device="tt", edge_tp=a.edge_tp)
model = network.load(o)
job = read_csv_jobs(Path(a.workload) / "screen72.csv")[a.ligand]
out = Path(a.out)
rec = dock_run._receptor_structure(job, out.with_suffix("") / job.name)
spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(rec), ligand_description=job.ligand)
emb = esm.receptor_embeddings(dfd_input.chain_sequences_for_embedding(rec), on_device=False)
g = dfd_input.build_complex_graph(spec, seed=o.seed, receptor_path=rec, lm_embeddings=emb,
                                  all_atoms=model.config.all_atoms)
cfg = dock_run._sampling_config(o)
b = sampling.SigmaBounds.from_config(model.config)
re_ = g.ligand["edge_index"].T[g.ligand["edge_mask"]]


def roll():
    n = sampling.Noise(seed=o.seed)
    s = sampling.randomize_position(g.ligand["pos"], receptor_pos=g.receptor["pos"],
        n_samples=o.diffusion_samples, cfg=cfg, bounds=b, noise=n, edge_index=re_,
        mask_rotate=g.ligand["mask_rotate"], no_torsion=model.config.no_torsion, centre=None)
    inner, per = model.score_fn(g), []
    def sc(st):
        t = time.perf_counter(); r = inner(st); per.append(time.perf_counter() - t); return r
    t0 = time.perf_counter()
    r = sampling.sample(sc, s, cfg=cfg, bounds=b, noise=n, edge_index=re_,
                        mask_rotate=g.ligand["mask_rotate"], no_torsion=model.config.no_torsion,
                        crop_beyond_a=model.config.crop_beyond)
    return time.perf_counter() - t0, per, r.poses.double()


Layer = ddev.DeviceConvLayer
head_call, head_gather = Layer.__call__, ddev.gather_rows


def base_call(self, *args, inv_degree=None, **kw):
    return head_call(self, *args, **kw)


def gather_k(k):
    bf, rm = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    def gather(table, idx, n):
        if table.dtype == bf:
            return head_gather(table, idx, n)
        acc, rest = None, table
        for j in range(k):
            part = ttnn.typecast(rest, bf)
            if j < k - 1:
                rest = ttnn.subtract(rest, ttnn.typecast(part, ttnn.float32))
            got = ttnn.embedding(idx, ttnn.to_layout(part, rm), layout=ttnn.TILE_LAYOUT)
            got = ttnn.typecast(ttnn.reshape(got, (n, table.shape[-1])), ttnn.float32)
            acc = got if acc is None else ttnn.add(acc, got)
        return acc
    return gather


def arm(name):
    Layer.__call__ = base_call if name == "base" else head_call
    ddev.gather_rows = {"g2": gather_k(2), "g1": gather_k(1)}.get(name, head_gather)


arms = a.arms.split(",")
res = {"edge_tp": a.edge_tp, "name": job.name, "poses": o.diffusion_samples, "steps": cfg.steps,
       "arms": {k: {"rollout_s": [], "median_step_s": []} for k in arms}}
poses = {}
for k in arms:                            # warm: every arm's programs compiled
    arm(k); roll()
for rep in range(a.reps):
    for k in arms:
        arm(k)
        t, per, p = roll()
        res["arms"][k]["rollout_s"].append(round(t, 3))
        res["arms"][k]["median_step_s"].append(round(statistics.median(per), 4))
        if k in poses:
            res["arms"][k]["rerun_bit_identical"] = bool(torch.equal(poses[k], p))
        poses[k] = p
        print(k, rep, round(t, 3), flush=True)
ref = poses[arms[0]]
for k in arms[1:]:
    rmsd = (poses[k] - ref).pow(2).sum(-1).mean(-1).sqrt()
    res["arms"][k]["vs_" + arms[0]] = {
        "bit_identical": bool(torch.equal(poses[k], ref)),
        "rmsd_a_per_pose": [round(float(x), 4) for x in rmsd],
        "rmsd_a_median": round(float(rmsd.median()), 4), "rmsd_a_max": round(float(rmsd.max()), 4),
        "max_atom_dev_a": round(float((poses[k] - ref).norm(dim=-1).max()), 4)}
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res), flush=True)
