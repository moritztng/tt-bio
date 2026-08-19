#!/usr/bin/env python3
"""Is an RF3 trunk pass bit-exact run to run on one card? Measured, not assumed.

Two passes over identical device inputs in one process, then the same again with the
whole model rebuilt. `protenix-v2-blackhole-nondeterminism-256aa` records another model
on this architecture losing bit-exactness above 128 aa; whether RF3 does the same is a
fact to write down, not a bug to hide.
"""
import argparse, json, os, sys
from pathlib import Path
import torch

REPO = Path("/home/ttuser/.coworker/wt/rf3-port-p1-parity")
sys.path.insert(0, str(REPO))
SEL = {"cyclic": dict(cyclic_chains=["A"])}

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", default="glke")
ap.add_argument("--ckpt", default="/home/ttuser/rf3_ref_work/rf3_latest.ckpt")
args = ap.parse_args()

import ttnn
from tt_bio.rf3 import model as rf3_model
from tt_bio.rf3.featurize import featurize
from tt_bio.rf3.host import HostInputs
from tt_bio.tenstorrent import get_device

d = REPO / "scripts/rf3_port/parity_artifacts" / args.fixture
prev = os.getcwd(); os.chdir(d)
try:
    out = featurize("input.json", n_recycles=1, diffusion_batch_size=1, seed=42,
                    **SEL.get(args.fixture, {}))[0]
finally:
    os.chdir(prev)
f = out["feats"]

dev = get_device()
cfg = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True, packer_l1_acc=True)
tt = rf3_model.load(args.ckpt, cfg, num_timesteps=8, with_confidence=False)

def trunk_once(rebuild_host):
    host = HostInputs.build(f, dev) if rebuild_host else HOST
    s_inputs, s, z = tt.trunk(host, 1)
    return [torch.Tensor(ttnn.to_torch(x)).float().clone() for x in (s_inputs, s, z)]

# Order matters. The reuse arm must run LAST: a run that reads a consumed buffer can
# leave the allocator in a state that contaminates whatever runs after it, and doing it
# in the middle is what made the first version of this probe unreadable.
HOST = HostInputs.build(f, dev)
a = trunk_once(True)
b = trunk_once(True)           # fresh host inputs both times: the determinism question
c = trunk_once(False)          # HOST reused across two calls: the buffer-lifetime question

rows = []
for name, i in (("S_inputs", 0), ("S_trunk", 1), ("Z_trunk", 2)):
    def cmp(x, y):
        dmax = float((x - y).abs().max())
        return {"bit_exact": dmax == 0.0, "maxabs": dmax,
                "rel_rms": (float((x - y).pow(2).mean().sqrt() / y.std())
                            if y.std() > 0 else 0.0)}
    rows.append({"tensor": name, "fresh_inputs_twice": cmp(a[i], b[i]),
                 "host_inputs_reused": cmp(a[i], c[i])})
print(json.dumps({"fixture": args.fixture, "atoms": HOST.n_atom,
                  "tokens": HOST.n_token, "comparisons": rows}, indent=2))
