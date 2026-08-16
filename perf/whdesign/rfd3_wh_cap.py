"""Where does RFD3 stop running on a 12 GB Wormhole chip?

R2 (318 target residues, 3844 atoms) runs and R3 (414, 4558) does not: it fails in the token
initializer, before any sampling, needing 111,241,216 B per DRAM bank against 227,338,464 B free
whose largest contiguous block is 103,054,624 B. Fragmentation, not capacity, and it reproduces in a
fresh process with nothing else ever loaded.

This bisects the cap on ONE target, R3's 9ma0, by varying only the contig length, so the two ends of
the bracket differ in size and in nothing else. R2 and R3 are different PDBs, which is fine for a
ladder and not fine for a cap.

One length per process, always: the first failure was seen in a load-once ladder carrying three
rungs' device state, and the whole point of this measurement is that the answer must not depend on
what ran before it. A short sample follows the token initializer, because surviving setup is not the
same as running.

    RFD3_CAP_LEN=380 TT_VISIBLE_DEVICES=<umd> PYTHONPATH=$PWD python3 perf/whdesign/rfd3_wh_cap.py
"""
import json
import os
import pathlib
import re
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3.design import build_token_initializer, build_diffusion_module   # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                                      # noqa: E402
from tt_bio.rfd3.input import InputSpecification                                 # noqa: E402
from tt_bio.rfd3.featurize import featurize                                      # noqa: E402
from tt_bio.rfd3.model import set_tune_matmul_for_atoms                          # noqa: E402

LEN = int(os.environ["RFD3_CAP_LEN"])
BINDER = os.environ.get("RFD3_CAP_BINDER", "100")
TARGET = os.environ.get("RFD3_CAP_TARGET", "perf/dsfix/targets/R3_9ma0_A.pdb")
STEPS = int(os.environ.get("RFD3_CAP_STEPS", "2"))
CKPT = pathlib.Path(os.environ.get("RFD3_CKPT", "/home/ttuser/.boltz/rfd3/weights"))
OUT = pathlib.Path(os.environ.get("RFD3_CAP_OUT", "perf/whdesign/results/rfd3_wh_cap.jsonl"))
HOST = os.environ.get("RFD3_HOST", "UF-EV-A13-GWH02")

rec = {"target": TARGET, "target_res": LEN, "binder": int(BINDER), "steps": STEPS,
       "host": HOST, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "atoms": None, "stage": "start", "ok": False, "error": None, "free_bytes": None}


def oom_numbers(msg):
    """Pull the allocator's own four numbers out of the throw, so the record carries them."""
    m = re.search(r"allocate (\d+) B DRAM buffer across (\d+) banks, where each bank needs to store "
                  r"(\d+) B, but bank size is (\d+) B \(allocated: (\d+) B, free: (\d+) B, "
                  r"largest free block: (\d+) B\)", msg)
    if not m:
        return None
    k = ["request_B", "banks", "per_bank_B", "bank_size_B", "allocated_B", "free_B", "largest_free_B"]
    d = {k[i]: int(m.group(i + 1)) for i in range(7)}
    d["free_over_need"] = round(d["free_B"] / d["per_bank_B"], 3)
    d["largest_over_need"] = round(d["largest_free_B"] / d["per_bank_B"], 3)
    return d


t0 = time.time()
try:
    dm = build_diffusion_module(torch.load(CKPT / "diffusion_module.real_weights.pt",
                                           map_location="cpu", weights_only=True))
    ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                            map_location="cpu", weights_only=True))
    spec = InputSpecification.from_dict(
        {"input": TARGET, "contig": "A1-%d,%s" % (LEN, BINDER), "length": BINDER})
    rec["stage"] = "featurize"
    f = featurize(spec.input, spec)
    rec["stage"] = "token_initializer"
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = int(init["Q_L_init"].shape[0])
    rec["atoms"] = L
    rec["tune_matmul"] = set_tune_matmul_for_atoms(L)
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    rec["stage"] = "sample"
    with torch.no_grad():
        X, _ = RFD3Sampler(num_timesteps=STEPS).sample(
            dm, 1, L, coord0, f, init, is_motif,
            generator=[torch.Generator().manual_seed(7)])
    rec["coords_finite"] = bool(torch.isfinite(X).all())
    rec["atoms_out"] = int(X.shape[1])
    rec["ok"] = rec["coords_finite"] and rec["atoms_out"] == L
    rec["stage"] = "done"
except Exception as e:                                    # noqa: BLE001
    msg = str(e)
    rec["error"] = {"type": type(e).__name__, "msg": msg[:400], "oom": oom_numbers(msg)}
rec["wall_s"] = round(time.time() - t0, 1)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("[cap] res=%d atoms=%s stage=%s ok=%s %s"
      % (LEN, rec["atoms"], rec["stage"], rec["ok"],
         json.dumps(rec["error"]["oom"]) if rec["error"] and rec["error"]["oom"] else
         (rec["error"]["type"] if rec["error"] else "")), flush=True)
