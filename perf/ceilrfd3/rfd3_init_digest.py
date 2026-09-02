"""Digest of everything the RFD3 token initializer returns, for one contig length.

The row-blocked P_LL section has to be bit-exact against the unblocked one, and "bit-exact"
here means the five host tensors the initializer hands the sampler are byte-identical, not
that a PCC clears a threshold. Run it once per checkout and diff the JSON.

    RFD3_CAP_LEN=100 TT_VISIBLE_DEVICES=<umd> PYTHONPATH=$PWD python3 perf/ceilrfd3/rfd3_init_digest.py
"""
import hashlib
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                                      # noqa: E402
from tt_bio.rfd3.design import build_token_initializer                           # noqa: E402
from tt_bio.rfd3.input import InputSpecification                                 # noqa: E402
from tt_bio.rfd3.featurize import featurize                                      # noqa: E402
from tt_bio.tenstorrent import get_device                                        # noqa: E402

LEN = int(os.environ["RFD3_CAP_LEN"])
BINDER = os.environ.get("RFD3_CAP_BINDER", "100")
TARGET = os.environ.get("RFD3_CAP_TARGET", "perf/dsfix/targets/R3_9ma0_A.pdb")
CKPT = pathlib.Path(os.environ.get("RFD3_CKPT", "/home/cust-team/.boltz/rfd3/weights"))
OUT = pathlib.Path(os.environ.get("RFD3_DIGEST_OUT", "perf/ceilrfd3/results/init_digest.jsonl"))
TAG = os.environ.get("RFD3_CAP_TAG", "")

t0 = time.time()
dev = get_device()
ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                        map_location="cpu", weights_only=True))
spec = InputSpecification.from_dict(
    {"input": TARGET, "contig": "A1-%d,%s" % (LEN, BINDER), "length": BINDER})
f = featurize(spec.input, spec)
with torch.no_grad():
    init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

rec = {"target_res": LEN, "binder": int(BINDER), "target": TARGET, "tag": TAG,
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "wall_s": round(time.time() - t0, 1),
       "atoms": int(init["Q_L_init"].shape[0])}
for k in sorted(init):
    v = init[k]
    rec[k] = {"shape": list(v.shape),
              "md5": hashlib.md5(v.contiguous().numpy().tobytes()).hexdigest()}
mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
rec["dram_free_per_bank_B"] = int(mv.total_bytes_free_per_bank)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(json.dumps(rec), flush=True)
