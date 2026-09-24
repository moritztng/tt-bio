"""af2ig's device output on one tree, as digests: device against device, no CPU reference arm.

Runs tap_gate.py's own device arm (bf16 trunk, 3 recycles, the laczc128_b80 fixture) from the
tree on PYTHONPATH and writes, per tap, a sha256 of its float32 bytes, plus the confidence
scalars and each tap's PCC against the captured JAX activations. Two trees agree bit-for-bit
iff every digest matches. AICLK and loadavg are sampled from sysfs every 2 s during the arm.

    PYTHONPATH=<tree> TT_VISIBLE_DEVICES=3 python3 perf/bcx_land/af2ig_digest.py --out x.json
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--params", default=os.path.expanduser("~/pxd_tool_weights/af2/params_model_1_ptm.npz"))
ap.add_argument("--npz", default=None, help="also write every tap as float32 here (~400 MB)")
ap.add_argument("--sysfs", default="/sys/class/tenstorrent/tenstorrent!0")
args = ap.parse_args()

tree = Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()
sys.path.insert(0, str(tree / "scripts" / "af2_port"))
sys.path.insert(0, str(tree))
import tap_gate as tg
from tt_bio.af2_confidence import confidence_scalars
from tt_bio.af2_weights import load_af2_state_dict

clock, stop = [], threading.Event()
def sample():
    while not stop.is_set():
        clock.append((round(time.time(), 1), Path(args.sysfs, "tt_aiclk").read_text().strip(),
                      float(Path("/proc/loadavg").read_text().split()[0])))
        stop.wait(2)

feats, prev = tg.load_inputs(tg.ARTIFACTS / "ref_inputs.npz")
with np.load(tg.ARTIFACTS / "ref_taps.npz", allow_pickle=False) as npz:
    ref = {k: npz[k] for k in npz.files}
bases = sorted({k.rsplit("/", 1)[0] for k in ref if k.endswith("/shape")})
state = load_af2_state_dict(args.params)

th = threading.Thread(target=sample, daemon=True); th.start()
t0, c0 = time.time(), time.process_time()
taps, last = tg.run_arm(state, feats, prev, template=True, dtype=torch.bfloat16, keep=set(bases),
                        recycles=tg.DEFAULT_RECYCLES, device=True)
wall, cpu = time.time() - t0, time.process_time() - c0
stop.set(); th.join()

digest = {k: hashlib.sha256(v.float().contiguous().numpy().tobytes()).hexdigest()[:16]
          for k, v in sorted(taps.values.items())}
whole = hashlib.sha256("".join(digest.values()).encode()).hexdigest()[:16]
pcc = {r["tap"]: r.get("pcc") for r in (tg.score_one(ref, b, taps.values[b]) for b in bases if b in taps.values)}
meta = json.loads(bytes(ref["_meta/json"]).decode())
binder = meta["fixture"]["binder_residues"] if meta["production"]["protocol"] == "binder" else None
scal = confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                          feats["seq_mask"], feats["asym_id"], binder_len=binder)
if args.npz:
    np.savez(args.npz, **{k: v.float().numpy() for k, v in taps.values.items()})
stamp = tree / ".commit"
commit = (stamp.read_text().strip() if stamp.exists() else
          subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
json.dump({"tree": str(tree), "commit": commit,
           "digest": whole, "taps": digest, "pcc_vs_jax": pcc, "scalars": scal,
           "triatt_fused_stats": tg._triatt_stats(True), "compute_grid": tg._compute_grid(True),
           "wall_s": wall, "cpu_s": cpu,
           "aiclk_mhz": sorted({c[1] for c in clock}), "aiclk_samples": [c[1] for c in clock],
           "loadavg": [c[2] for c in clock][:: max(1, len(clock) // 20)],
           "host": os.uname().nodename}, open(args.out, "w"), indent=1, default=float)
print("DIGEST", whole, "wall", round(wall, 1), "aiclk", sorted({c[1] for c in clock}))
