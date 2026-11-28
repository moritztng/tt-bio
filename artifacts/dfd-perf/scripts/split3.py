"""Per-OP device seconds inside the conv layers of one stepped rollout on a gated ligand.

split2 gave seconds per layer (0.27 us per edge row, the same for every main-stack layer). This
names what inside a layer costs it: every ttnn call a layer makes is wrapped with a device
synchronise on both sides and booked to (op, shapes). The synchronisation is a cost, so the total
is not a timing of the shipped path; only the split is.
"""
import argparse, collections, json, statistics, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--edge-tp", default="tile")
ap.add_argument("--ligand", type=int, default=9)
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
dev = next(iter(model.convs.values())).dev.device
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
    sampling.sample(sc, s, cfg=cfg, bounds=b, noise=n, edge_index=re_,
                    mask_rotate=g.ligand["mask_rotate"], no_torsion=model.config.no_torsion,
                    crop_beyond_a=model.config.crop_beyond)
    return time.perf_counter() - t0, per


roll()                                   # warm: every program compiled
t_plain, per_plain = roll()              # the shipped path, unsynchronised

book = collections.defaultdict(lambda: [0.0, 0])


def shape_of(x):
    return tuple(x.shape) if hasattr(x, "shape") and not isinstance(x, (int, float)) else None


def wrap(owner, name, label):
    fn = getattr(owner, name)
    def timed(*args, **kw):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        r = fn(*args, **kw)
        ttnn.synchronize_device(dev)
        key = (label, str([s for s in map(shape_of, args) if s is not None]))
        book[key][0] += time.perf_counter() - t
        book[key][1] += 1
        return r
    setattr(owner, name, timed)


OPS = ("matmul", "linear", "embedding", "concat", "multiply", "add", "subtract", "typecast",
       "reciprocal", "clamp", "relu", "to_layout", "reshape", "pad", "from_torch", "to_torch")
for name in OPS:
    wrap(ttnn, name, name)
# the tensor product itself, whole: its inner ops are also booked above, under their own names
tp_types = {type(tp) for m in model.convs.values() for tp in m.dev.tps}
for T in tp_types:
    wrap(T, "__call__", "TP:" + T.__name__)

t_sync, per_sync = roll()
rows = sorted(({"op": k[0], "shapes": k[1], "s": round(v[0], 3), "calls": v[1],
                "ms_per_call": round(v[0] / v[1] * 1e3, 3)} for k, v in book.items()),
              key=lambda r: -r["s"])
by_op = collections.defaultdict(float)
for r in rows:
    by_op[r["op"]] += r["s"]
res = {"edge_tp": a.edge_tp, "name": job.name, "poses": o.diffusion_samples, "steps": cfg.steps,
       "n_lig": int(g.ligand["pos"].shape[0]), "n_rec": int(g.receptor["pos"].shape[0]),
       "rollout_plain_s": round(t_plain, 3), "median_step_plain_s": round(statistics.median(per_plain), 4),
       "rollout_synced_s": round(t_sync, 3),
       "note": "TP:* rows contain the ops booked under their own names inside the tensor product; "
               "synchronised per op, so only the split is meaningful",
       "by_op_s": dict(sorted(((k, round(v, 3)) for k, v in by_op.items()), key=lambda kv: -kv[1])),
       "top": rows[:60]}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "top"}), flush=True)
