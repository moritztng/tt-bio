"""Where does RFD3 stop running on a 12 GiB Wormhole chip, and what is holding the DRAM?

Restored from `perf/whdesign/rfd3_wh_cap.py` (46eac51a, pruned from perf/ by 57669d90) with two
additions this campaign needs: a DRAM census taken at three points in the run, and an allocator
message parser that reads the numbers out of whatever wording ttnn throws.

One length per process, always. The first RFD3 out-of-memory was seen in a load-once ladder
carrying three rungs' device state, and the whole point of this measurement is that the answer
must not depend on what ran before it. A short sample follows the token initializer, because
surviving setup is not the same as running.

    RFD3_CAP_LEN=390 TT_VISIBLE_DEVICES=<umd> PYTHONPATH=$PWD python3 perf/ceilrfd3/rfd3_cap.py
"""
import json
import os
import pathlib
import re
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                                      # noqa: E402
from tt_bio.rfd3.design import build_token_initializer, build_diffusion_module   # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                                      # noqa: E402
from tt_bio.rfd3.input import InputSpecification                                 # noqa: E402
from tt_bio.rfd3.featurize import featurize                                      # noqa: E402
from tt_bio.rfd3.model import set_tune_matmul_for_atoms, ATOM_PAIR_BLOCK_STATS   # noqa: E402
from tt_bio.tenstorrent import get_device                                        # noqa: E402
import tt_bio                                                                    # noqa: E402

LEN = int(os.environ["RFD3_CAP_LEN"])
BINDER = os.environ.get("RFD3_CAP_BINDER", "100")
TARGET = os.environ.get("RFD3_CAP_TARGET", "perf/ceilrfd3/targets/laczc_1008.cif")
STEPS = int(os.environ.get("RFD3_CAP_STEPS", "2"))
CKPT = pathlib.Path(os.environ.get("RFD3_CKPT", "/home/cust-team/.boltz/rfd3/weights"))
OUT = pathlib.Path(os.environ.get("RFD3_CAP_OUT", "perf/ceilrfd3/results/rfd3_cap.jsonl"))
HOST = os.environ.get("RFD3_HOST", "UF-EV-A13-GWH02")
TAG = os.environ.get("RFD3_CAP_TAG", "")

rec = {"target": TARGET, "target_res": LEN, "binder": int(BINDER), "steps": STEPS,
       "total_res": LEN + int(BINDER),
       "host": HOST, "card": os.environ.get("TT_VISIBLE_DEVICES"), "tag": TAG,
       "atom_pair_budget_env": os.environ.get("TT_BIO_ATOM_PAIR_BUDGET_BYTES"),
       "atoms": None, "stage": "start", "ok": False, "error": None, "dram": {}}


def oom_numbers(msg):
    """Pull the allocator's own numbers out of the throw, whatever the wording.

    ttnn has phrased this message at least two ways across versions, so match on the labels
    that carry the numbers rather than on one full sentence.  A parse that silently returns
    None turns the one measurement that decides the mechanism into a bare exception type.
    """
    if "not enough space to allocate" not in msg:
        return None
    d = {}
    for key, pat in (("request_B", r"allocate (\d+) B"),
                     ("banks", r"across (\d+) banks"),
                     ("per_bank_B", r"bank needs? (?:to store )?(\d+) B"),
                     ("bank_size_B", r"bank size (?:is )?(\d+)"),
                     ("allocated_B", r"allocated:? (\d+)"),
                     ("free_B", r"free:? (\d+)"),
                     ("largest_free_B", r"largest free block:? (\d+)")):
        m = re.search(pat, msg)
        if m:
            d[key] = int(m.group(1))
    if "per_bank_B" in d and d["per_bank_B"]:
        if "free_B" in d:
            d["free_over_need"] = round(d["free_B"] / d["per_bank_B"], 3)
        if "largest_free_B" in d:
            d["largest_over_need"] = round(d["largest_free_B"] / d["per_bank_B"], 3)
    return d or None


def census(dev, where):
    """Bytes the DRAM allocator is holding right now, so 'fragmentation' is not asserted.

    `free_over_need` above compares free space to one request; this compares the standing
    allocation to the chip, which is the other half of the same question.
    """
    try:
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        lcf = mv.largest_contiguous_bytes_free_per_bank
        if isinstance(lcf, (list, tuple)):
            lcf = min(lcf)
        rec["dram"][where] = {"banks": int(mv.num_banks),
                              "total_per_bank_B": int(mv.total_bytes_per_bank),
                              "free_per_bank_B": int(mv.total_bytes_free_per_bank),
                              "largest_free_per_bank_B": int(lcf)}
    except Exception as e:                                # noqa: BLE001
        rec["dram"][where] = {"unavailable": type(e).__name__}


# Which tree is actually under test. The galaxy venv has tt_bio installed editable against
# the production checkout, so a run launched from the wrong cwd measures that tree and says
# nothing about the branch. Recorded in every row, not checked once at the top of a campaign.
rec["tt_bio"] = tt_bio.__file__
rec["host_rss_peak_kB"] = None

t0 = time.time()
try:
    dev = get_device()
    census(dev, "open")
    dm = build_diffusion_module(torch.load(CKPT / "diffusion_module.real_weights.pt",
                                           map_location="cpu", weights_only=True))
    ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                            map_location="cpu", weights_only=True))
    census(dev, "weights")
    spec = InputSpecification.from_dict(
        {"input": TARGET, "contig": "A1-%d,%s" % (LEN, BINDER), "length": BINDER})
    rec["stage"] = "featurize"
    f = featurize(spec.input, spec)
    rec["stage"] = "token_initializer"
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    census(dev, "token_init")
    L = int(init["Q_L_init"].shape[0])
    rec["atoms"] = L
    rec["tune_matmul"] = set_tune_matmul_for_atoms(L)
    rec["atom_pair_rows"], rec["atom_pair_blocks"] = ATOM_PAIR_BLOCK_STATS
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    rec["stage"] = "sample"
    with torch.no_grad():
        X, _ = RFD3Sampler(num_timesteps=STEPS).sample(
            dm, 1, L, coord0, f, init, is_motif,
            generator=[torch.Generator().manual_seed(7)])
    census(dev, "sample")
    rec["coords_finite"] = bool(torch.isfinite(X).all())
    rec["atoms_out"] = int(X.shape[1])
    rec["ok"] = rec["coords_finite"] and rec["atoms_out"] == L
    rec["stage"] = "done"
except Exception as e:                                    # noqa: BLE001
    msg = str(e)
    rec["error"] = {"type": type(e).__name__, "msg": msg[:600], "oom": oom_numbers(msg)}
rec["wall_s"] = round(time.time() - t0, 1)
try:
    # Host RAM has been this model's binding resource before, so the peak is measured
    # beside the DRAM census rather than argued about.
    rec["host_rss_peak_kB"] = int([l.split()[1] for l in open("/proc/self/status")
                                   if l.startswith("VmHWM")][0])
except Exception:                                         # noqa: BLE001
    pass
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("[cap] res=%d atoms=%s stage=%s ok=%s %s"
      % (LEN, rec["atoms"], rec["stage"], rec["ok"],
         json.dumps(rec["error"]["oom"]) if rec["error"] and rec["error"]["oom"] else
         (rec["error"]["type"] if rec["error"] else "")), flush=True)
print("[dram] " + json.dumps(rec["dram"]), flush=True)
