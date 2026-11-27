"""Per-ligand stage split of the shipped dock on the GATED workload (dfd-orchestrator, 17:00Z).

PERF.md section 10's table is the gemm form on 5S8I_2LY. wlt2 reads 25.46 s/complex on the
tile form docking 5SAK_ZRY against the WORKLOAD-S ligands, and nothing says where those
seconds are. This times the stages `tt_bio.diffdock.run._sample_one` runs, in its order, on
the workload's own receptor and its own ligands, with the model loaded once as a screen loads
it, so rollout-on-this-complex is separated from everything around it.
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--edge-tp", default="tile", choices=["gemm", "tile"])
ap.add_argument("--name", default="wlsplit")
ap.add_argument("--n", type=int, default=4, help="timed ligands, after one discarded")
args = ap.parse_args()

T0 = time.time()
import torch  # noqa: F401
from tt_bio.diffdock import (checkpoints, network, output, chunking, esm,
                             input as dfd_input, neighbour_cap, sampling)
from tt_bio.diffdock.job import DockOptions, read_csv_jobs
from tt_bio.diffdock import run as dock_run
t_import = time.time() - T0

WL = Path("/home/ttuser/scratch/dfd/wl/workload")
OUT = Path("/home/ttuser/scratch/dfd/perf") / args.name
OUT.mkdir(parents=True, exist_ok=True)
RESULT = Path(f"/home/ttuser/scratch/dfd/perf/{args.name}.json")

options = DockOptions(device="tt", **({"edge_tp": "tile"} if args.edge_tp == "tile" else {}))
jobs = read_csv_jobs(WL / "screen72.csv")
picked = jobs[8:8 + args.n + 1]          # one discarded, then args.n timed

t = time.time()
checkpoints.resolve("score", options.checkpoint, None)
checkpoints.resolve("confidence", options.confidence_checkpoint, None)
t_ckpt = time.time() - t
t = time.time()
model = network.load(options)
t_load = time.time() - t

cfg = dock_run._sampling_config(options)
bounds = sampling.SigmaBounds.from_config(model.config)
rows = []


def one(job):
    m, t = {}, time.time()
    receptor = dock_run._receptor_structure(job, OUT / job.name)
    spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(receptor),
                                 ligand_description=job.ligand)
    chains = dfd_input.chain_sequences_for_embedding(receptor)
    m["receptor_structure_chains"] = round(time.time() - t, 3)

    t = time.time()
    emb = esm.receptor_embeddings(chains, on_device=getattr(model, "on_device", True))
    m["esm_receptor_embeddings"] = round(time.time() - t, 3)

    t = time.time()
    graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                          lm_embeddings=emb,
                                          all_atoms=model.config.all_atoms)
    neighbour_cap.check_ligand_graph(graph.ligand["pos"], model.config.lig_max_radius,
                                     label=job.name, on_input=True)
    m["build_complex_graph"] = round(time.time() - t, 3)

    t = time.time()
    noise = sampling.Noise(seed=options.seed)
    start = sampling.randomize_position(
        graph.ligand["pos"], receptor_pos=graph.receptor["pos"],
        n_samples=options.diffusion_samples, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
        mask_rotate=graph.ligand["mask_rotate"], no_torsion=model.config.no_torsion,
        centre=None)
    m["randomize_position"] = round(time.time() - t, 3)

    n_torsions = 0 if model.config.no_torsion else len(graph.ligand["mask_rotate"])
    score_fn = chunking.chunked_score_fn(model.score_fn(graph), options.max_parallel_samples,
                                         n_torsions=n_torsions,
                                         reset=getattr(model, "drop_trace", None))
    t = time.time()
    rollout = sampling.sample(score_fn, start, cfg=cfg, bounds=bounds, noise=noise,
                              edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
                              mask_rotate=graph.ligand["mask_rotate"],
                              no_torsion=model.config.no_torsion,
                              crop_beyond_a=model.config.crop_beyond, trajectory=False)
    m["rollout"] = round(time.time() - t, 3)

    t = time.time()
    conf = model.confidence(graph, rollout.poses)
    m["confidence"] = round(time.time() - t, 3)

    t = time.time()
    from rdkit import Chem
    mol = Chem.RemoveHs(Chem.Mol(graph.mol))
    poses = (rollout.poses + graph.original_center).numpy()
    c = torch.as_tensor(conf).flatten().float()
    order = torch.argsort(c, descending=True).tolist()
    output.write_poses(OUT / job.name, mol, poses[order], confidences=c[order].tolist())
    m["rank_write"] = round(time.time() - t, 3)

    m["name"] = job.name
    m["edges_lig"] = int(graph.ligand["edge_index"].shape[1])
    m["n_rec_nodes"] = int(graph.receptor["pos"].shape[0])
    m["n_lig_atoms"] = int(graph.ligand["pos"].shape[0])
    m["total"] = round(sum(v for k, v in m.items() if isinstance(v, float)), 3)
    return m


for i, job in enumerate(picked):
    r = one(job)
    r["discarded"] = (i == 0)
    print(json.dumps(r), flush=True)
    rows.append(r)
    timed = [x for x in rows if not x["discarded"]]
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    med = {k: round(sorted(x[k] for x in timed)[len(timed) // 2], 3) for k in keys} if timed else {}
    RESULT.write_text(json.dumps(
        {"edge_tp": args.edge_tp, "receptor": "5SAK_ZRY", "steps": cfg.steps,
         "poses": options.diffusion_samples, "ligands": rows,
         "median_timed_stage_s": med,
         "cold_lines": {"import_s": round(t_import, 3), "checkpoints_s": round(t_ckpt, 3),
                        "network_load_s": round(t_load, 3)}}, indent=1))
print("split done", flush=True)
