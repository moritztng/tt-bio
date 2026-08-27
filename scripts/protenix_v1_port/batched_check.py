#!/usr/bin/env python3
"""Does the batched Protenix path give protenix-v1 the same answer as folding serially?

worker.predict_many folds several targets in ONE diffusion trajectory. Its gate was
`model == "protenix-v2"`; protenix-v1 now shares it, because the two ids share the class, the
features and the sampler. Sharing it is only correct if the batched answer matches the serial
one, which is what this measures: B copies of one target through predict_many against the same
target through predict_one, same seed.

The CLI cannot reach predict_many (`predict` takes one DATA argument) and its only other caller
is a throughput benchmark, so without this the extended gate would ship unexercised.

    TT_VISIBLE_DEVICES=0 ... PYTHONPATH=$WT env/bin/python3 \\
        scripts/protenix_v1_port/batched_check.py --target examples/multimer.yaml
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="examples/multimer.yaml")
    ap.add_argument("--model", default="protenix-v1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    work = Path(tempfile.mkdtemp(prefix="ptxv1-batch-"))
    struct = work / "out"
    struct.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=args.model, fast=False, output_format="cif",
               recycling_steps=None, sampling_steps=args.steps,
               diffusion_samples=1, seed=args.seed, trace=False,
               msa_dir=str(work / "msa"), struct_dir=str(struct),
               use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
               single_sequence=True, write_pae=False, write_pde=False,
               write_embeddings=False, method=None, max_msa_seqs=8192)
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)

    t = Path(args.target)
    paths = [t] * args.batch
    torch.set_grad_enabled(False)

    many = state.predict_many(paths, cfg)
    ones = [state.predict_one(p, cfg) for p in paths]

    print("\n%-8s %-12s %-12s %-12s" % ("member", "plddt(many)", "plddt(one)", "|delta|"))
    ok = True
    for i, (m, o) in enumerate(zip(many, ones)):
        pm, po = m[0]["plddt"], o[0]["plddt"]
        d = abs(pm - po)
        ok = ok and d < 5e-3
        print("%-8d %-12.6f %-12.6f %-12.3g" % (i, pm, po, d))
    for k in ("ptm", "iptm", "n_tokens", "n_atoms"):
        vs_m = [m[0].get(k) for m in many]
        vs_o = [o[0].get(k) for o in ones]
        same = vs_m == vs_o
        ok = ok and (same if k in ("n_tokens", "n_atoms") else True)
        print("%-10s many=%s  one=%s  %s" % (k, vs_m, vs_o, "SAME" if same else "DIFF"))
    print("\nBATCHED %s" % ("OK" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
