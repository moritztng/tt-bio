"""Where one WORKLOAD-S rollout spends its seconds, score_fn form against the resident rollout.

Docks one ligand of the gated screen onto 5SAK_ZRY three ways with the model loaded once:
`sampling.sample` over `score_fn` (what `tt-bio dock` runs), the device-resident rollout issued
eagerly (`trace_body`), and one captured trace of it replayed. Each is run more than once so the
last is warm, the score_fn form is profiled, and the resident poses are compared with the
score_fn poses at the same seed (``--seed``; an unseeded rollout draws its own noise, and two of
those differ by the seed floor whatever the form). The clock is sampled by the caller.
"""
import argparse, cProfile, io, json, pstats, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--edge-tp", default="tile", choices=["gemm", "tile"])
ap.add_argument("--ligand", type=int, default=9, help="row of screen72.csv")
ap.add_argument("--workload", default="/home/ttuser/scratch/dfd/wl/workload")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import torch  # noqa: E402,F401
from tt_bio.diffdock import esm, input as dfd_input, network, sampling  # noqa: E402
from tt_bio.diffdock import run as dock_run  # noqa: E402
from tt_bio.diffdock.job import DockOptions, read_csv_jobs  # noqa: E402

out_path = Path(args.out)
res = {"edge_tp": args.edge_tp, "seed": args.seed}


def save():
    out_path.write_text(json.dumps(res, indent=1))


options = DockOptions(device="tt", edge_tp=args.edge_tp, seed=args.seed)
t = time.time()
model = network.load(options)
res["network_load_s"] = round(time.time() - t, 3)
job = read_csv_jobs(Path(args.workload) / "screen72.csv")[args.ligand]
scratch = out_path.with_suffix("") / job.name
receptor = dock_run._receptor_structure(job, scratch)
spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(receptor),
                             ligand_description=job.ligand)
emb = esm.receptor_embeddings(dfd_input.chain_sequences_for_embedding(receptor), on_device=False)
graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                      lm_embeddings=emb, all_atoms=model.config.all_atoms)
cfg = dock_run._sampling_config(options)
bounds = sampling.SigmaBounds.from_config(model.config)
rot_edges = graph.ligand["edge_index"].T[graph.ligand["edge_mask"]]
res.update(name=job.name, n_lig=int(graph.ligand["pos"].shape[0]),
           n_rec=int(graph.receptor["pos"].shape[0]), steps=cfg.steps,
           poses=options.diffusion_samples)


def start_and_noise():
    noise = sampling.Noise(seed=options.seed)
    start = sampling.randomize_position(
        graph.ligand["pos"], receptor_pos=graph.receptor["pos"],
        n_samples=options.diffusion_samples, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=rot_edges, mask_rotate=graph.ligand["mask_rotate"],
        no_torsion=model.config.no_torsion, centre=None)
    return start, noise


def stepped(profile=False):
    start, noise = start_and_noise()
    inner, per_step = model.score_fn(graph), []

    def score(step):
        t0 = time.perf_counter()
        r = inner(step)
        per_step.append(round(time.perf_counter() - t0, 4))
        return r

    dev0 = sum(m.secs for m in model.convs.values())
    prof = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    r = sampling.sample(score, start, cfg=cfg, bounds=bounds, noise=noise, edge_index=rot_edges,
                        mask_rotate=graph.ligand["mask_rotate"],
                        no_torsion=model.config.no_torsion,
                        crop_beyond_a=model.config.crop_beyond)
    if prof:
        prof.disable()
    row = {"rollout_s": round(time.perf_counter() - t0, 3), "per_step_s": per_step,
           "in_device_layers_s": round(sum(m.secs for m in model.convs.values()) - dev0, 3),
           "profiled": profile}
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(45)
        out_path.with_suffix(".prof.txt").write_text(s.getvalue())
    return row, r.poses


def deviation(a, b):
    d = (a - b).norm(dim=-1)
    rmsd = d.pow(2).mean(1).sqrt()
    return {"per_pose_rmsd_a": [round(x, 3) for x in rmsd.tolist()],
            "median_rmsd_a": round(rmsd.median().item(), 3), "max_atom_a": round(d.max().item(), 3)}


res["score_fn"], ref = [], None
for i in range(3):
    prev = ref
    row, ref = stepped(profile=(i == 2))
    if prev is not None:
        row["vs_previous_run"] = deviation(ref, prev)
    res["score_fn"].append(row)
    save()

import ttnn  # noqa: E402

res["resident_eager"] = []
for i in range(2):
    start, noise = start_and_noise()
    t0 = time.perf_counter()
    body, read = model.trace_body(graph, start, cfg, bounds, noise)
    built = time.perf_counter() - t0
    per_step = []
    for s in range(cfg.steps):
        t1 = time.perf_counter()
        body(s)
        ttnn.synchronize_device(model.device)
        per_step.append(round(time.perf_counter() - t1, 4))
    t1 = time.perf_counter()
    poses = read()
    res["resident_eager"].append({"build_s": round(built, 3), "per_step_s": per_step,
                                  "steps_s": round(sum(per_step), 3),
                                  "read_s": round(time.perf_counter() - t1, 3),
                                  "vs_score_fn": deviation(poses, ref)})
    save()

from tt_bio.diffdock.capture import capture_loop  # noqa: E402

start, noise = start_and_noise()
body, read = model.trace_body(graph, start, cfg, bounds, noise)
t0 = time.perf_counter()
tid, _ = capture_loop(model.device, body, cfg.steps)
row = {"capture_s": round(time.perf_counter() - t0, 3), "replay_s": []}
for i in range(3):
    t0 = time.perf_counter()
    ttnn.execute_trace(model.device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(model.device)
    row["replay_s"].append(round(time.perf_counter() - t0, 3))
row["vs_eager"] = deviation(read(), poses)
res["resident_traced"] = row
save()
print("rollout_prof done", flush=True)
