#!/usr/bin/env python3
"""of3t-angle job 1: the D242 cot_z correction ON THE frame384 FRAME, and nothing else.

A42 says compute the correction ONCE, on the arm that defines the boundary, and drive every
other arm from it. On the model frame that arm is `ref_f64_model_n384_corrected` and the file
is `cot_external.pt`, sha256 1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4.
THAT FILE IS A DIFFERENT FRAME. It is built from `of3t_modelframe/cot_model_n384.pt`
(4e66d1ef...) on `boundary_model_n384.pt` (583bcd7c...); the softmax ladder lives on
of3t-frame384`s `boundary_n384.pt` (8cb3a586...) with `block47_boundary.pt` (a55ef1c4...) as the
capture. A correction is d<cot_s, s_out>/d(z_out): it is a function of THIS boundary and THIS
cotangent, so the model frame`s cannot be carried across. This script computes the frame384 one
and every arm in this row is driven from its sha256.

It is not a second producer. `ref_grad.py`s own `build` and `ancestor_pairs` are imported, the
blocks before the last run under `no_grad` where `ref_grad --checkpoint` runs them under
recompute, and the last block runs eagerly in both. Recompute of an eval()-mode block is
deterministic, so the two must agree BIT for BIT -- and that identity is checked against the
full corrected f64 arm`s own `cot_z_correction` rather than assumed (job 2).
"""
from __future__ import annotations

import argparse, hashlib, json, os, resource, socket, subprocess, sys, time

import torch


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--producer", required=True, help="the repaired ref_grad.py")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--cap-last", required=True)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(a.producer)))
    sys.path.insert(0, a.tree)
    import ref_grad as R
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree on sys.path: {openfold3.__file__}")

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    sd = torch.load(R.CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mods, dims, load = R.build(sd, a.blocks, torch.float64)
    if load["missing"] or load["unexpected"]:
        raise SystemExit("strict load did not hold")

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s = b["s_in"].to(torch.float64).contiguous()
    z = b["z_in"].to(torch.float64).contiguous()
    sm = b["single_mask"].to(torch.float64)
    pm = b["pair_mask"].to(torch.float64)

    cap = torch.load(a.cap_last, map_location="cpu", weights_only=False)
    cot_s, cot_z = cap["cot"][0], cap["cot"][1]
    if cot_s is None or cot_z is None:
        raise SystemExit("captured cotangent missing")
    c = a.crop
    cot_s = cot_s.to(torch.float64)[:, :c].contiguous() if c else cot_s.to(torch.float64)
    cot_z = (cot_z.to(torch.float64)[:, :c, :c].contiguous() if c else cot_z.to(torch.float64))

    with torch.no_grad():
        for m in mods[:-1]:
            s, z = m(s, z, sm, pm)
    s = s.detach().clone()
    z = z.detach().clone()
    t1 = time.perf_counter()
    s_out, z_out = mods[-1](s, z, sm, pm)
    pairs = R.ancestor_pairs([("s_out", s_out), ("z_out", z_out)])
    if pairs != [("z_out", "s_out")]:
        raise SystemExit(f"the last block did not reproduce D242s ancestry: {pairs}")
    corr = torch.autograd.grad(outputs=s_out.to(torch.float64), grad_outputs=cot_s,
                               inputs=z_out)[0].detach().to(torch.float64)
    if tuple(corr.shape) != tuple(cot_z.shape):
        raise SystemExit(f"correction {tuple(corr.shape)} vs cot_z {tuple(cot_z.shape)}")

    torch.save({"cot_z_correction": corr,
                "cot": (cot_s, cot_z - corr),
                "injection_convention": "graph-cut-external",
                "from": {"boundary": a.boundary, "capture": a.cap_last,
                         "producer": a.producer}}, a.out)

    rep = {"what": __doc__.strip().splitlines()[0],
           "host": socket.gethostname(), "row": "of3t-angle", "defect": "D242",
           "device_involved": False,
           "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
           "frame": "of3t-frame384: boundary_n384.pt / block47_boundary.pt, padded 384",
           "NOT_THE_MODEL_FRAME": {
               "model_frame_cot_external_sha256":
                   "1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4",
               "why_it_cannot_be_used_here":
                   "it is d<cot_s,s_out>/dz_out on boundary_model_n384.pt with cot_model_n384.pt. "
                   "This row scores the softmax ladder on the frame384 frame, a different "
                   "boundary and a different capture, so the correction has to be recomputed "
                   "here. A correction is a property of its frame."},
           "tree": a.tree, "openfold3_file": openfold3.__file__,
           "producer": {"path": a.producer, "sha256": sha256_file(a.producer)},
           "boundary": a.boundary, "boundary_sha256": sha256_file(a.boundary),
           "cotangent_from": a.cap_last, "cotangent_sha256": sha256_file(a.cap_last),
           "graph_cut": {"is_cut": not pairs,
                         "ancestor_descendant_pairs": [list(x) for x in pairs]},
           "dims": dims, "blocks": a.blocks, "crop": a.crop, "threads": a.threads,
           "norms": {"cot_s": float(cot_s.norm()),
                     "cot_z_hooked": float(cot_z.norm()),
                     "correction": float(corr.norm()),
                     "cot_z_external": float((cot_z - corr).norm()),
                     "duplicate_share_of_the_hooked_cot_z":
                         float(corr.norm()) / float(cot_z.norm()),
                     "hooked_over_external":
                         float(cot_z.norm()) / float((cot_z - corr).norm()),
                     "s_out": float(s_out.norm()), "z_out": float(z_out.norm())},
           "seconds": {"total": time.perf_counter() - t0,
                       "last_block_and_grad": time.perf_counter() - t1},
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "out": {"path": a.out, "sha256": sha256_file(a.out),
                   "bytes": os.path.getsize(a.out)}}
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
