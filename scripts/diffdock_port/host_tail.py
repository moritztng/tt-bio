"""How much of a rollout step is host work that no kernel can remove.

`dock_stages` put the rollout at 22.81 s against the 17.3 s this row measured on the score
network alone. The difference is the sampler's own per-step work -- the noise draw, the
rigid update, the torsion update, the chunk reassembly -- and it rides on the host whatever
the card does. This runs the real `sampling.sample` over the real graph with the score
function replaced by a zero stub of the right shape, so what is left IS the tail. Card-free
on purpose: the tail never touches a chip.
"""
import time, json
import torch

from tt_bio.diffdock import chunking, input as dfd_input, sampling
from tt_bio.diffdock import run as dock_run
from tt_bio.diffdock.job import DockJob, DockOptions
from tt_bio.diffdock import config as dfd_config, checkpoints, score_model

PB = "/home/ttuser/scratch/dfd/pb"
job = DockJob(name="5S8I_2LY", receptor_path=f"{PB}/5S8I_2LY_protein.pdb",
              ligand=f"{PB}/5S8I_2LY_ligand.sdf")
options = DockOptions(device="cpu")

ckpt, params = checkpoints.resolve("score", options.checkpoint, None)
cfg_model = dfd_config.load(params.parent, old_model=False, name="diffdock")

from pathlib import Path
out = Path("/home/ttuser/scratch/dfd/perf/host_tail_out"); out.mkdir(parents=True, exist_ok=True)
receptor = dock_run._receptor_structure(job, out)
spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(receptor),
                             ligand_description=job.ligand)
graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                      lm_embeddings=None, all_atoms=cfg_model.all_atoms)

cfg = dock_run._sampling_config(options)
bounds = sampling.SigmaBounds.from_config(cfg_model)
noise = sampling.Noise(seed=options.seed)
P = options.diffusion_samples
start = sampling.randomize_position(
    graph.ligand["pos"], receptor_pos=graph.receptor["pos"], n_samples=P, cfg=cfg,
    bounds=bounds, noise=noise,
    edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
    mask_rotate=graph.ligand["mask_rotate"], no_torsion=cfg_model.no_torsion)

n_tor = 0 if cfg_model.no_torsion else len(graph.ligand["mask_rotate"])
calls = {"n": 0}

def stub(step):
    calls["n"] += 1
    p = step.lig_pos.shape[0]
    return (torch.zeros(p, 3), torch.zeros(p, 3), torch.zeros(p, n_tor))

score_fn = chunking.chunked_score_fn(stub, options.max_parallel_samples, n_torsions=n_tor)

times = []
for rep in range(3):
    noise = sampling.Noise(seed=options.seed)
    t = time.time()
    sampling.sample(score_fn, start, cfg=cfg, bounds=bounds, noise=noise,
                    edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
                    mask_rotate=graph.ligand["mask_rotate"], no_torsion=cfg_model.no_torsion,
                    crop_beyond_a=cfg_model.crop_beyond, trajectory=False)
    times.append(time.time() - t)
    print(f"rep {rep}: {times[-1]:.3f}s host tail over {cfg.steps} steps x {P} poses", flush=True)

best = min(times)
print(json.dumps({"complex": "5S8I_2LY", "steps": cfg.steps, "poses": P,
                  "torsions": n_tor, "score_calls": calls["n"],
                  "host_tail_s": round(best, 3),
                  "host_tail_s_per_step": round(best / cfg.steps, 4),
                  "reps": [round(t, 3) for t in times],
                  "rollout_on_card_s": 22.81,
                  "tail_fraction_of_rollout": round(best / 22.81, 4)}, indent=1))
