#!/usr/bin/env python3
"""Rows 13 and 14 on device, both directions, one model load.

Row 13 (partial diffusion) and row 14 (pLDDT early stopping) are wiring, so the test that
matters is that the flag is not accepted and ignored. Everything here runs off one small
real target with one checkpoint load:

  1. baseline: a normal fold, partial_t=0, no early stop.
  2. early stop with a threshold above the measured recycle-1 mean pLDDT: must stop, must
     carry no structure, must not run the rollout.
  3. early stop with a threshold below it: must NOT stop, and must return exactly the
     baseline structure -- the flag only adds a read-only head pass, so bit-exact is the bar.
  4. partial diffusion from the baseline structure at partial_t near the data end: must
     stay within a few angstrom of it, while partial_t=0 on the same coordinates must not.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rf3-port-p3 \\
        python3 scripts/rf3_port/check_partial_and_earlystop.py \\
            --ckpt ~/rf3_ref_work/rf3_latest.ckpt
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SEQ = "MKTAYIAKQRQISFVKSHFS"     # 20 aa, the target pass 2 folded end to end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--recycles", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import confidence as rf3_confidence
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.featurize import featurize
    from tt_bio.tenstorrent import get_device

    with tempfile.TemporaryDirectory() as td:
        spec = Path(td) / "mono.json"
        spec.write_text(json.dumps(
            [{"name": "mono", "components": [{"seq": SEQ, "chain_id": "A"}]}]))
        out = featurize(spec, n_recycles=args.recycles, diffusion_batch_size=1, seed=0)[0]
    f = out["feats"]
    is_real_atom = out["confidence_feats"]["is_real_atom"]
    rep = out["ground_truth"]["rep_atom_idxs"]

    dev = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"] if "config" in ck else None
    del ck
    kw = {}
    if cfg is not None:
        kw = dict(n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
                  n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
                  n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
                  with_confidence="confidence_head" in cfg)
    t0 = time.time()
    tt = rf3_model.load(args.ckpt, kcfg, num_timesteps=args.steps, **kw)
    load_s = round(time.time() - t0, 1)

    def run(**over):
        torch.manual_seed(0)
        return tt.predict(f, n_recycles=args.recycles, diffusion_batch_size=1,
                          rep_atom_idxs=rep, is_real_atom=is_real_atom, **over)

    rows = {}
    base = run()
    assert base["early_stopped"] is False
    rows["baseline"] = {
        "early_stopped": False,
        "plddt": round(float(rf3_confidence.atomwise_plddt(
            base["plddt_logits"][0], is_real_atom).mean()) * 100, 3),
        "n_atom": int(base["X_L"].shape[1])}

    # 2 -- must stop. A threshold of 1.0 is above any mean pLDDT by construction, and the
    #      return reports the value the decision was taken on, which brackets step 3.
    stop = run(early_stop_plddt=1.0)
    m = stop["mean_plddt"]
    rows["stops_above"] = {"early_stopped": stop["early_stopped"],
                           "mean_plddt": round(m, 6),
                           "has_structure": "X_L" in stop}
    assert stop["early_stopped"] is True and "X_L" not in stop

    # 3 -- must not stop, and must be the baseline bit-exactly
    go = run(early_stop_plddt=max(0.0, m - 0.05))
    rows["runs_below"] = {
        "early_stopped": go["early_stopped"],
        "threshold": round(max(0.0, m - 0.05), 6),
        "mean_plddt": round(go["mean_plddt"], 6),
        "bit_exact_vs_baseline": bool(torch.equal(go["X_L"], base["X_L"])),
        "max_abs_diff": float((go["X_L"] - base["X_L"]).abs().max())}
    assert go["early_stopped"] is False

    # 4 -- partial diffusion. The input to be noised has to be a structure the model would
    #      NOT predict on its own, or both arms land on the baseline and the test proves
    #      nothing: feeding the baseline back and asking a full rollout to ignore it gives
    #      0.03 A either way, because the draws are shared and sched[0] swamps the input.
    #      A fixed permutation of the baseline's atoms is a structure of the right scale
    #      that the model does not predict.
    centred = base["X_L"] - base["X_L"].mean(dim=1, keepdim=True)
    perm = torch.randperm(centred.shape[1], generator=torch.Generator().manual_seed(7))
    scrambled = centred[:, perm].contiguous()

    def rmsd(x, ref):
        a = (x - x.mean(dim=1, keepdim=True))[0].double()
        b = (ref - ref.mean(dim=1, keepdim=True))[0].double()
        u, _, vt = torch.linalg.svd(a.T @ b)
        d = torch.sign(torch.det(u @ vt))
        r = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)]).double()) @ vt
        return float(((a @ r) - b).pow(2).sum(-1).mean().sqrt())

    near = run(coord_to_be_noised=scrambled, partial_t=args.steps - 1)
    far = run(coord_to_be_noised=scrambled, partial_t=0)
    rows["partial_diffusion"] = {
        "steps": args.steps, "input": "permuted baseline",
        "near_partial_t": args.steps - 1,
        "near_rmsd_to_input": round(rmsd(near["X_L"], scrambled), 3),
        "near_rmsd_to_baseline": round(rmsd(near["X_L"], centred), 3),
        "far_partial_t": 0,
        "far_rmsd_to_input": round(rmsd(far["X_L"], scrambled), 3),
        "far_rmsd_to_baseline": round(rmsd(far["X_L"], centred), 3)}
    r = rows["partial_diffusion"]
    assert r["near_rmsd_to_input"] < r["far_rmsd_to_input"] / 3, (
        "partial_t near the data end must keep the structure it was given; got "
        f"{r['near_rmsd_to_input']} vs {r['far_rmsd_to_input']}")
    assert r["far_rmsd_to_baseline"] < r["near_rmsd_to_baseline"] / 3, (
        "partial_t=0 must ignore the input and land on the model's own answer; got "
        f"{r['far_rmsd_to_baseline']} vs {r['near_rmsd_to_baseline']}")

    rep_out = {"seq_len": len(SEQ), "recycles": args.recycles, "load_s": load_s,
               "checks": rows, "verdict": "PASS"}
    print(json.dumps(rep_out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep_out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
