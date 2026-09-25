"""af2ig's device arm with the laczc128_b80 input padded at the tail to a bucket, graded on real residues.

Bucket 1 is the fixture as captured (208 residues, af2ig_digest.py's arm). Bucket 32 pads every
per-residue feature to 224 with zeros and seq/msa/extra-msa/template masks of 0, residue_index
continuing through the pad as BindCraft 2's `protein.py` does. Every tap is sliced back to the 208
real residues and scored against the captured JAX activations exactly as tap_gate scores them, and
the final CA trace is compared (Kabsch) to the bucket-1 device run and to JAX's own structure.

    PYTHONPATH=<tree> TT_VISIBLE_DEVICES=3 python3 perf/bcx_landmask/af2ig_bucket.py --bucket 32 \
        --b1 perf/bcx_landmask/af2ig/b1_c1.npz --out x.json
"""
import argparse, hashlib, json, os, sys, threading, time, traceback
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--bucket", type=int, default=32)
ap.add_argument("--b1", default=None, help="bucket-1 final positions (.pt) to compare against; written if absent")
ap.add_argument("--params", default=os.path.expanduser("~/pxd_tool_weights/af2/params_model_1_ptm.npz"))
ap.add_argument("--sysfs", default="/sys/class/tenstorrent/tenstorrent!0")
args = ap.parse_args()

tree = Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()
sys.path.insert(0, str(tree / "scripts" / "af2_port"))
sys.path.insert(0, str(tree))
import tap_gate as tg
from tt_bio.af2_confidence import confidence_scalars
from tt_bio.af2_weights import load_af2_state_dict

feats, prev = tg.load_inputs(tg.ARTIFACTS / "ref_inputs.npz")
n = int(feats["seq_mask"].shape[0])
N = -(-n // args.bucket) * args.bucket


def pad(t: torch.Tensor, key: str) -> torch.Tensor:
    for ax in range(t.dim()):
        if t.shape[ax] == n:
            widths = [0, 0] * (t.dim() - ax - 1) + [0, N - n]
            t = torch.nn.functional.pad(t, widths)
    return t


pf = {k: pad(v, k) for k, v in feats.items()}
pf["residue_index"][n:] = feats["residue_index"][-1] + 1 + torch.arange(N - n)
pp = {k: pad(v, k) for k, v in prev.items()}
assert float(pf["seq_mask"].sum()) == n and float(pf["msa_mask"][:, n:].abs().sum()) == 0


def unpad(t: torch.Tensor) -> torch.Tensor:
    for ax in range(t.dim()):
        if t.shape[ax] == N:
            t = t.narrow(ax, 0, n)
    return t


with np.load(tg.ARTIFACTS / "ref_taps.npz", allow_pickle=False) as npz:
    ref = {k: npz[k] for k in npz.files}
bases = sorted({k.rsplit("/", 1)[0] for k in ref if k.endswith("/shape")})
state = load_af2_state_dict(args.params)

clock, stop = [], threading.Event()
def sample():
    while not stop.is_set():
        clock.append((round(time.time(), 1), Path(args.sysfs, "tt_aiclk").read_text().strip(),
                      float(Path("/proc/loadavg").read_text().split()[0])))
        stop.wait(2)

th = threading.Thread(target=sample, daemon=True); th.start()
t0 = time.time()
blob = {"tree": str(tree), "n": n, "bucket": args.bucket, "padded_to": N, "host": os.uname().nodename}
try:
    taps, last = tg.run_arm(state, pf if N != n else feats, pp if N != n else prev, template=True,
                            dtype=torch.bfloat16, keep=set(bases), recycles=tg.DEFAULT_RECYCLES, device=True)
except Exception as e:  # main asserts the mask away; record how, not just that
    blob["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-600:]
    taps = None
wall = time.time() - t0
stop.set(); th.join()
blob |= {"wall_s": wall, "aiclk_mhz": sorted({c[1] for c in clock}), "aiclk_samples": [c[1] for c in clock],
         "loadavg": [c[2] for c in clock][:: max(1, len(clock) // 20)]}
if taps is not None:
    vals = {k: unpad(v) for k, v in taps.values.items()}
    rows = [tg.score_one(ref, b, vals[b]) for b in bases if b in vals]
    blob["pcc_vs_jax"] = {r["tap"]: r.get("pcc") for r in rows}
    blob["failing_taps_vs_jax"] = sorted(r["tap"] for r in rows if r["verdict"] != "PASS")
    blob["digest"] = hashlib.sha256("".join(
        hashlib.sha256(v.float().contiguous().numpy().tobytes()).hexdigest()[:16]
        for _, v in sorted(vals.items())).encode()).hexdigest()[:16]
    meta = json.loads(bytes(ref["_meta/json"]).decode())
    binder = meta["fixture"]["binder_residues"] if meta["production"]["protocol"] == "binder" else None
    blob["scalars"] = confidence_scalars(unpad(last["plddt_logits"]), unpad(last["pae_logits"]), last["pae_breaks"],
                                         feats["seq_mask"], feats["asym_id"], binder_len=binder)
    ca = unpad(last["structure"]["final_atom_positions"])[:, 1].double()
    blob["pad_rows_finite"] = bool(torch.isfinite(last["structure"]["final_atom_positions"]).all())
    blob["pcc_min_vs_jax"] = min(p for p in blob["pcc_vs_jax"].values() if p is not None)

    def kabsch(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        u, s, vt = torch.linalg.svd(a.T @ b)
        d = torch.sign(torch.det(u @ vt))
        r = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=a.dtype)) @ vt
        return float(((a @ r - b) ** 2).sum(1).mean().sqrt())
    jax = torch.from_numpy(ref["structure_module#3/final_atom_positions/full"].astype(np.float64)).reshape(-1, 37, 3)[:, 1]
    blob["ca_rmsd_vs_jax_A"] = kabsch(ca, jax)
    b1 = Path(args.b1) if args.b1 else None
    if b1 and b1.exists():
        blob["ca_rmsd_vs_b1_A"] = kabsch(ca, torch.load(b1).double())
    elif b1:
        torch.save(ca.float(), b1)
json.dump(blob, open(args.out, "w"), indent=1, default=float)
print({k: blob.get(k) for k in ("bucket", "padded_to", "digest", "ca_rmsd_vs_b1_A", "ca_rmsd_vs_jax_A", "pcc_min_vs_jax", "error", "wall_s", "aiclk_mhz")},
      "failing", len(blob.get("failing_taps_vs_jax", [])), "scalars", blob.get("scalars"))
