"""Card-free: profile write_design on a synthetic batch (engine-shaped output: residue types legal per token, atom37 mask
from the decoder's own table, coordinates = input coords + noise), the same call perf/rfd4/fair/run.py times as write_s."""
import sys, time, cProfile, pstats, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
from tt_bio import rfd4
from tt_bio.rfd4.engine import DesignOutput
from tt_bio.rfd4 import ae_fold as AF, network as N
from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_ae_masks
spec, B, mode = sys.argv[2], int(sys.argv[3]), sys.argv[4]
out = Path(tempfile.mkdtemp(dir="/tmp/gap224prof"))
ex = rfd4.prepare_examples(spec, out / "inputs")[0]
f, aa = rfd4.featurize(ex.cif, seed=1000)
n = f["padding_mask"].shape[1]
tab = AF.tables()
b = N._broadcast_batch(dict(f), B)
_, emit = compute_ae_masks(b)
g = torch.Generator().manual_seed(0)
cs = b[AF.C_SEQ].long()
rt = torch.where((cs >= 1) & (cs <= 20), cs, torch.randint(1, 21, cs.shape, generator=g))
am = tab["atom_mask"][rt] & emit[..., None]
am[..., 1] |= b["padding_mask"].bool()
coords = (b["coords"].float() / 10 + 0.05 * torch.randn(*b["coords"].shape, generator=g)) * am[..., None]
rt = torch.where(emit, rt, torch.full_like(rt, -1))
o = DesignOutput(coords_nm=coords, residue_type=rt, atom_mask=am, insertion_logits=torch.randn(B, n, n, generator=g))
print("tokens", n, "B", B)
def run(tag):
    t0 = time.perf_counter()
    for k in range(B):
        rfd4.write_design(o, f, aa, out / tag / f"d{k}.cif", row=k)
    return time.perf_counter() - t0
run("warm")
ts = [run(f"r{i}") for i in range(3)]
print("write_s per batch", [round(t, 3) for t in ts], "per design ms", round(1e3 * min(ts) / B, 1))
if mode == "prof":
    pr = cProfile.Profile(); pr.enable(); run("prof"); pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(45)
