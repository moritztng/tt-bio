"""p45 — which RFD3 kernel gates are still open at the page fixture.

Every fused kernel in the RFD3 path gates itself on shape and falls back silently, so a
CIF cannot tell "served" from "declined" (`rfd3_bias.stats_line` exists for exactly that
reason). This runs a few real timesteps at a chosen rung and batch and reports, per gate,
how often it served, how often it declined, and with which shape.

    p45_gate_census.py --rung R4 --batch 2 --steps 4
"""
import argparse, collections, json, os, pathlib, statistics, sys, time
import torch

sys.path.insert(0, os.getcwd())
os.environ.setdefault("RFD3_BIAS_STATS", "1")

import ttnn  # noqa: F401
from tt_bio import rfd3_bias
from tt_bio.rfd3.design import build_token_initializer, build_diffusion_module
from tt_bio.rfd3.sampler import RFD3Sampler
from tt_bio.rfd3.input import InputSpecification
from tt_bio.rfd3.featurize import featurize

ap = argparse.ArgumentParser()
ap.add_argument("--rung", default="R4")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="perf/p45/gate_census.jsonl")
a = ap.parse_args()

# Log every eligible_shape call with its shape, not just the pass/fail counter, because the
# gate that matters is `n_keys % 32` and the counter alone cannot show which n_keys was seen.
CALLS = collections.Counter()
_orig_eligible = rfd3_bias.eligible_shape


def eligible_shape(batch, n_heads, length, n_keys, dtype):
    ok = _orig_eligible(batch, n_heads, length, n_keys, dtype)
    CALLS[(int(batch), int(n_heads), int(length), int(n_keys), str(dtype), bool(ok))] += 1
    return ok


rfd3_bias.eligible_shape = eligible_shape
import tt_bio.rfd3.model as rmodel
rmodel.rfd3_bias.eligible_shape = eligible_shape

CKPT = pathlib.Path("/home/ttuser/.boltz/rfd3/weights")
dm = build_diffusion_module(torch.load(CKPT / "diffusion_module.real_weights.pt",
                                       map_location="cpu", weights_only=True))
ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                        map_location="cpu", weights_only=True))

specs = json.loads(pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % a.rung).read_text())
sid, sdict = next(iter(specs.items()))
spec = InputSpecification.from_dict(sdict)
f = featurize(spec.input, spec)
with torch.no_grad():
    init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
L = init["Q_L_init"].shape[0]
I = int(init["Q_L_init"].shape[0])
is_motif = f["is_motif_atom_with_fixed_coord"]
coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
print("[p45] rung=%s atoms=%d batch=%d steps=%d" % (a.rung, L, a.batch, a.steps), flush=True)

STEPS = []
_orig_call = RFD3Sampler.__call__ if hasattr(RFD3Sampler, "__call__") else None

sampler = RFD3Sampler(num_timesteps=a.steps)
gens = [torch.Generator().manual_seed(a.seed + i) for i in range(a.batch)]
t0 = time.perf_counter()
with torch.no_grad():
    X, _ = sampler.sample(dm, a.batch, L, coord0, f, init, is_motif, generator=gens)
wall = time.perf_counter() - t0
print("[p45] wall %.3f s for %d steps at batch %d" % (wall, a.steps, a.batch), flush=True)
print("[p45] %s" % rfd3_bias.stats_line(), flush=True)
for k, n in sorted(CALLS.items()):
    b, h, ln, nk, dt, ok = k
    print("[p45] eligible_shape(batch=%d n_heads=%d L=%d n_keys=%d %s) -> %s   x%d"
          % (b, h, ln, nk, dt, ok, n), flush=True)

out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("a") as fh:
    fh.write(json.dumps({
        "rung": a.rung, "atoms": L, "batch": a.batch, "steps": a.steps,
        "wall_s": wall,
        "sparse_bias_served": rfd3_bias.STATS[0], "sparse_bias_declined": rfd3_bias.STATS[1],
        "fused_served": rfd3_bias.FSTATS[0],
        "rejects": {str(k): v for k, v in rfd3_bias.REJECTS.items()},
        "eligible_calls": [{"batch": k[0], "n_heads": k[1], "L": k[2], "n_keys": k[3],
                            "dtype": k[4], "ok": k[5], "n": n} for k, n in sorted(CALLS.items())],
    }) + "\n")
print("[p45] wrote %s" % out, flush=True)
