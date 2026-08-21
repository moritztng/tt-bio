#!/usr/bin/env python3
"""Is an RF3 pass bit-exact run to run on one card? Measured, not assumed.

Three questions, three arms:

  in-process    two passes over freshly built device inputs in one process
  buffer reuse  a third pass that reuses one HostInputs across calls
  cross-process --dump twice into two files, then --compare them

`protenix-v2-blackhole-nondeterminism-256aa` records another model on this architecture
losing bit-exactness above 128 aa; whether RF3 does the same is a fact to write down,
not a bug to hide. Cross-process is the case that matters for a served model, and it is
the one an in-process probe cannot see.

With --full the compared tensors are a whole fold (coordinates, distogram, pLDDT logits)
rather than the trunk alone. The host RNG is re-seeded immediately before each fold, so
what is being measured is the device'"'"'s determinism and not two different noise draws.
"""
import argparse, json, os, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
SEL = {"cyclic": dict(cyclic_chains=["A"])}

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", default="glke")
ap.add_argument("--ckpt", default="/home/ttuser/rf3_ref_work/rf3_latest.ckpt")
ap.add_argument("--full", action="store_true",
                help="compare a whole fold (X_L, distogram, pLDDT) not just the trunk")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--dump", default=None,
                help="write this run's tensors here; run twice and --compare for the "
                     "cross-process answer")
ap.add_argument("--compare", nargs=2, default=None,
                help="compare two --dump files and exit; needs no device")
args = ap.parse_args()


def cmp_tensors(x, y):
    dmax = float((x - y).abs().max())
    return {"bit_exact": dmax == 0.0, "maxabs": dmax,
            "rel_rms": (float((x - y).pow(2).mean().sqrt() / y.std())
                        if y.std() > 0 else 0.0)}


if args.compare:
    a_d, b_d = (torch.load(f, weights_only=False) for f in args.compare)
    assert a_d["meta"] == b_d["meta"], (a_d["meta"], b_d["meta"])
    print(json.dumps({"arm": "cross_process", **a_d["meta"],
                      "comparisons": [{"tensor": k, **cmp_tensors(a_d["t"][k],
                                                                 b_d["t"][k])}
                                      for k in a_d["t"]]}, indent=2))
    sys.exit(0)

import ttnn
from tt_bio.rf3 import model as rf3_model
from tt_bio.rf3.featurize import featurize
from tt_bio.rf3.host import HostInputs
from tt_bio.tenstorrent import get_device

d = next((r / args.fixture for r in
          (REPO / "scripts/rf3_port/parity_artifacts",
           REPO / "scripts/rf3_port/size_ladder")
          if (r / args.fixture).is_dir()), None)
if d is None:
    raise SystemExit(f"{args.fixture}: no such fixture")
prev = os.getcwd(); os.chdir(d)
try:
    out = featurize("input.json", n_recycles=1, diffusion_batch_size=1, seed=42,
                    **SEL.get(args.fixture, {}))[0]
finally:
    os.chdir(prev)
f = out["feats"]
out_full = out

dev = get_device()
cfg = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True, packer_l1_acc=True)
tt = rf3_model.load(args.ckpt, cfg, num_timesteps=args.steps,
                    with_confidence=args.full)
NAMES = ["S_inputs", "S_trunk", "Z_trunk"]
if args.full:
    NAMES += ["X_L", "distogram", "plddt_logits"]


def trunk_once(rebuild_host):
    host = HostInputs.build(f, dev) if rebuild_host else HOST
    s_inputs, s, z = tt.trunk(host, 1)
    got = [torch.Tensor(ttnn.to_torch(x)).float().clone() for x in (s_inputs, s, z)]
    if not args.full:
        return got
    # Same noise both times: re-seed here, so a difference is the device's and not the
    # sampler's. rep_atom_idxs turns the confidence head on.
    torch.manual_seed(args.seed)
    out = tt.predict(f, n_recycles=1, diffusion_batch_size=1,
                     rep_atom_idxs=out_full.get("ground_truth", {}).get("rep_atom_idxs"))
    return got + [out["X_L"].float().clone(), out["distogram"].float().clone(),
                  out["plddt_logits"].float().clone()]

# Order matters. The reuse arm must run LAST: a run that reads a consumed buffer can
# leave the allocator in a state that contaminates whatever runs after it, and doing it
# in the middle is what made the first version of this probe unreadable.
HOST = HostInputs.build(f, dev)
a = trunk_once(True)
b = trunk_once(True)           # fresh host inputs both times: the determinism question
c = trunk_once(False)          # HOST reused across two calls: the buffer-lifetime question

meta = {"fixture": args.fixture, "atoms": HOST.n_atom, "tokens": HOST.n_token,
        "full": bool(args.full), "steps": args.steps, "seed": args.seed}
rows = [{"tensor": name, "fresh_inputs_twice": cmp_tensors(a[i], b[i]),
         "host_inputs_reused": cmp_tensors(a[i], c[i])}
        for i, name in enumerate(NAMES)]
print(json.dumps({"arm": "in_process", **meta, "comparisons": rows}, indent=2))
if args.dump:
    torch.save({"meta": meta, "t": dict(zip(NAMES, a))}, args.dump)
