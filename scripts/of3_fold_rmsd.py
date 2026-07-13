"""OpenFold3 end-to-end ``fold()`` -> vs-ground-truth Kabsch Cα-RMSD (the OF3 port
merge gate, docs/openfold3-port.md "Concretely what remains" item 4).

Runs the real device ``OpenFold3.fold()`` (tt_bio.openfold3_fold): device OF3Trunk ->
fresh-rollout device OF3SampleDiffusion -> atom coordinates, on the real ubiquitin
target (the fixture wired into every OF3 component golden), then Kabsch-aligns the
predicted Cα positions against the experimental structure (1UBQ, the human-ubiquitin
reference, matching the query sequence) and prints the real RMSD.

Features: the real ubiquitin batch features from ~/of3_ref_out.pkl (the P1 data
pipeline + reference embedder outputs -- the existing OF3 fixture, reused to avoid
redundant data-pipeline work). The trunk + diffusion sampler run on device for real;
the sampler rollout is freshly seeded (NOT a golden-replayed trajectory).

Ground truth: examples/ground_truth_structures/ubiquitin.pdb (1UBQ). Cα atom indices:
tests/fixtures/of3_ubiquitin_ca_mask.npy (601-atom axis, 76 Cα), derived once from the
vendored data pipeline's atom_array.

Usage (one device context per process, physical card 1):

    TT_VISIBLE_DEVICES=1 /home/ttuser/tt-bio/env/bin/python scripts/of3_fold_rmsd.py \\
        --no-rollout-steps 4 --no-samples 1 --seed 1234
    # full production rollout (200 steps x 5 samples -- slow):
    TT_VISIBLE_DEVICES=1 /home/ttuser/tt-bio/env/bin/python scripts/of3_fold_rmsd.py \\
        --no-rollout-steps 200 --no-samples 5 --seed 1234
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio.openfold3_fold import OpenFold3, kabsch_rmsd, load_pdb_ca

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
_GT_PDB = os.path.join(REPO, "examples/ground_truth_structures/ubiquitin.pdb")
_CA_MASK = os.path.join(REPO, "tests/fixtures/of3_ubiquitin_ca_mask.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rollout-steps", type=int, default=4)
    ap.add_argument("--no-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_fold import OpenFold3

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(_GOLD, "rb"))["intermediates"]
    tr = I["trunk_real"]
    te = I["template_embedder_real"]["feat"]
    me = I["msa_module_embedder_real"]["msa_feat"]
    cond = I["diffusion_conditioning_real"]
    xl = I["diffusion_module_xlout_real"]
    dec = I["diffusion_decoder_real"]
    at = I["input_embedder_atom_transformer_real"]

    n_atom = xl["n_atom"]; n_token = xl["n_token"]; nb = xl["nb"]; NP = xl["NP"]
    dm_aux_host = dict(
        cl0=xl["cl0"], plm0=xl["plm0"], atom_mask=dec["atom_mask"],
        atom_to_token_index=dec["atom_to_token_index"],
        npe_q_indices=xl["npe_q_indices"], npe_k_indices=xl["npe_k_indices"],
        zij_mask=xl["zij_mask"], key_block_idxs=dec["key_block_idxs"],
        invalid_mask=dec["invalid_mask"], mask_trunked=dec["mask_trunked"],
        atom_to_token_mean=at["atom_to_token_mean"], nb=nb, NP=NP)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    model = OpenFold3(sd, ckc, num_cycles=tr["num_cycles"])
    print(f"OpenFold3.fold(): no_rollout_steps={args.no_rollout_steps} "
          f"no_samples={args.no_samples} seed={args.seed} "
          f"n_atom={n_atom} n_token={n_token} sigma_data={model.sigma_data}")

    samples = model.fold(
        s_init=tr["s_init"], z_init=tr["z_init"], template_feat=te, msa_feat=me,
        s_input=tr["s_input"], si_input=cond["si_input"], relpos=cond["relpos"],
        token_mask=cond["token_mask"], dm_aux_host=dm_aux_host,
        n_atom=n_atom, n_token=n_token,
        no_rollout_steps=args.no_rollout_steps, seed=args.seed,
        no_samples=args.no_samples)

    ca_mask = np.load(_CA_MASK).astype(bool)
    assert len(ca_mask) == n_atom, f"ca_mask len {len(ca_mask)} != n_atom {n_atom}"
    gt_ca = load_pdb_ca(_GT_PDB)
    print(f"GT Cα (1UBQ): {gt_ca.shape[0]} atoms; predicted Cα: {int(ca_mask.sum())}")

    rmsds = []
    for i, xl_final in enumerate(samples):
        if not torch.isfinite(xl_final).all():
            print(f"  sample {i}: NON-FINITE coordinates -- wiring bug, aborting")
            sys.exit(2)
        pred_ca = xl_final[ca_mask].double()
        if pred_ca.shape[0] != gt_ca.shape[0]:
            print(f"  sample {i}: Cα count mismatch pred={pred_ca.shape[0]} gt={gt_ca.shape[0]}")
            sys.exit(2)
        r = kabsch_rmsd(pred_ca, gt_ca)
        rmsds.append(r)
        print(f"  sample {i}: Kabsch Cα-RMSD = {r:.4f} Å")
    rmsds = sorted(rmsds)
    print(f"\nRESULT: Kabsch Cα-RMSD over {len(rmsds)} sample(s): "
          f"min={rmsds[0]:.4f} Å median={rmsds[len(rmsds)//2]:.4f} Å "
          f"max={rmsds[-1]:.4f} Å")
    print(f"CONFIG: no_rollout_steps={args.no_rollout_steps} no_samples={args.no_samples} "
          f"seed={args.seed} target=ubiquitin(1UBQ) "
          f"rollout={'FULL' if args.no_rollout_steps>=200 else 'REDUCED'}")


if __name__ == "__main__":
    main()
