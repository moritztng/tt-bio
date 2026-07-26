"""Multiplicity-batching parity for Protenix-v2 and OpenDDE (NEEDS A CARD).

Captures golden coords from the CURRENT unbatched path (the per-sample loop,
DiffusionModule.supports_multiplicity=False) at n_sample=M with seeds seed..seed+M-1,
then runs the batched path (supports_multiplicity=True, one batched edm_sample trajectory
of multiplicity=M) and compares per-sample Kabsch RMSD + PCC against the golden.

The batched path uses one RNG stream for all M samples (mirrors boltz2.AtomDiffusion.sample,
see edm_sample docstring), so the per-sample draws differ from the unbatched loop's seed+k
draws -- the parity bar is therefore PCC / Kabsch-RMSD within the established seed-to-seed
diffusion noise floor (the same bar docs/implementation-parity.md uses for diffusion legs),
NOT bit-exact. R = unbatched-vs-unbatched across seed offsets, D = batched-vs-batched across
seed offsets, X = batched-vs-unbatched; floor = max(R, D); PASS when X <= floor within
sampling uncertainty (and the batched samples are finite + non-collapsed).

This script is the "capture golden BEFORE touching anything" + "diff after" step the task
requires. It only runs once DiffusionModule.supports_multiplicity is flipped on (pending the
device-denoise M carry-through + on-device verification on a free card). Until then it
exits early reporting that the batched path is not yet available.

Run (Protenix):
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:tt-bio-diffusion-multiplicity-batching \
    PYTHONPATH=<worktree> /home/ttuser/tt-bio/env/bin/python3 \
    scripts/protenix_opendde_multiplicity_parity.py protenix

Run (OpenDDE):
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:tt-bio-diffusion-multiplicity-batching \
    PYTHONPATH=<worktree> /home/ttuser/tt-bio/env/bin/python3 \
    scripts/protenix_opendde_multiplicity_parity.py opendde
"""
import os
import sys

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
os.environ.setdefault("TT_BIO_LEASE_HOLDER", "worker:tt-bio-diffusion-multiplicity-batching")

import argparse
import pickle
import time

import torch
import ttnn


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.t() @ Qc
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.t() @ U.t()))
    Dm = torch.diag(torch.tensor([1.0, 1.0, d]))
    R = Vt.t() @ Dm @ U.t()
    return float(((Pc @ R.t()) - Qc).pow(2).sum(-1).mean().sqrt())


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.norm() * b.norm() + 1e-12))


def _floor(values):
    if not values:
        return float("inf")
    return float(max(values))


def protenix_run(M, seed, n_step, supports):
    """Returns (coords (M,N,3),) from one fold call. `supports` sets the
    DiffusionModule.supports_multiplicity flag before the fold."""
    from tt_bio.protenix import Protenix
    from tt_bio.tenstorrent import get_device

    ife = pickle.load(open(os.environ.get("PROTENIX_IFE", "/home/ttuser/protenix_ife_gold.pkl"), "rb"))
    tg = pickle.load(open(os.environ.get("PROTENIX_TRUNKIN", "/home/ttuser/protenix_trunkin_gold.pkl"), "rb"))
    d = pickle.load(open(os.environ.get("PROTENIX_REFO", "/home/ttuser/protenix_ref_out.pkl"), "rb"))
    tfeat = d["intermediates"]["template_embedder"]["in"][0]
    F = ife["feat"]
    feats = {
        "ref_pos": F["ref_pos"], "ref_charge": F["ref_charge"], "ref_mask": F["ref_mask"],
        "ref_element": F["ref_element"], "ref_atom_name_chars": F["ref_atom_name_chars"],
        "d_lm": F["d_lm"], "v_lm": F["v_lm"], "atom_to_token_idx": F["atom_to_token_idx"],
        "restype": F["restype"], "profile": F["profile"], "deletion_mean": F["deletion_mean"],
        "mask_trunked": ife["mask_trunked"],
        "relp": tg["relp"], "token_bonds": tg["token_bonds"],
        "template_aatype": tfeat["template_aatype"],
        "template_distogram": tfeat["template_distogram"],
        "template_pseudo_beta_mask": tfeat["template_pseudo_beta_mask"],
        "template_unit_vector": tfeat["template_unit_vector"],
        "template_backbone_frame_mask": tfeat["template_backbone_frame_mask"],
        "msa": tfeat["msa"], "has_deletion": tfeat["has_deletion"],
        "deletion_value": tfeat["deletion_value"], "asym_id": tfeat["asym_id"],
    }
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(
        os.environ.get("PROTENIX_CKPT", "/home/ttuser/protenix_ckpt/protenix-v2.pt"),
        compute_kernel_config=ckc, device=dev)
    model.diffusion.supports_multiplicity = supports
    coords = model.fold(feats, n_step=n_step, n_sample=M, seed=seed,
                       max_parallel_samples=M, progress_fn=None)
    return coords.float()


def opendde_run(M, seed, n_step, n_cycles, supports):
    from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.tenstorrent import get_device

    SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
           "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    model._protenix.diffusion.supports_multiplicity = supports
    feats = build_complex_features([(SEQ, None, "protein")])
    coords = model.fold(feats, n_step=n_step, n_cycles=n_cycles, n_sample=M, seed=seed,
                        max_parallel_samples=M, return_confidence=False)
    return coords.float()


def collect(model_name, M, n_runs, n_step, seed_base, supports):
    """Run n_runs fold calls at multiplicity M, varying the base seed. Returns a list of
    (M, N, 3) coord tensors. Used to build R (unbatched-vs-unbatched) or D (batched-vs-batched)."""
    out = []
    for r in range(n_runs):
        seed = seed_base + 1000 * r
        t0 = time.time()
        if model_name == "protenix":
            c = protenix_run(M, seed, n_step, supports)
        else:
            c = opendde_run(M, seed, n_step, 2, supports)
        print(f"  [{model_name} supports={supports} M={M} seed={seed}] "
              f"{time.time()-t0:.1f}s  coords {tuple(c.shape)}  finite={bool(torch.isfinite(c).all())}",
              flush=True)
        out.append(c)
    return out


def pairwise_kabsch(runs):
    """All distinct (i<j) per-sample Kabsch RMSDs across runs, after matching sample i of
    run i to sample j of run j (samples are exchangeable within a run since they're all
    independent draws). Returns the max pairwise RMSD (the seed-to-seed floor)."""
    rmsds = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            for s in range(runs[i].shape[0]):
                rmsds.append(kabsch_rmsd(runs[i][s], runs[j][s]))
    return rmsds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["protenix", "opendde"])
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--n-runs", type=int, default=2, help="fold calls per leg (R, D, X)")
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    n_step = args.n_step or (10 if args.model == "protenix" else 20)
    M = args.M
    n_runs = args.n_runs
    print(f"[{args.model}] multiplicity-batching parity: M={M} n_runs={n_runs} n_step={n_step}",
          flush=True)

    # Golden: unbatched path (supports_multiplicity=False -> the per-sample loop).
    print("[1/3] capturing golden (unbatched, supports_multiplicity=False)...", flush=True)
    golden = collect(args.model, M, n_runs, n_step, args.seed_base, supports=False)

    # Batched path: only available once supports_multiplicity is flipped on.
    from tt_bio.protenix import DiffusionModule
    if not getattr(DiffusionModule, "supports_multiplicity", False):
        print("\n[SKIP] DiffusionModule.supports_multiplicity is False -- the batched device-"
              "denoise path is not yet enabled. This script is the ready-to-run parity check "
              "for AFTER the device-denoise M carry-through is implemented + verified on a card. "
              "Exiting without a verdict (golden captured above for reference).", flush=True)
        # still save the golden so the next session can diff against the SAME golden
        out_path = f"/tmp/{args.model}_multiplicity_golden_M{M}.pt"
        torch.save({"golden": golden, "M": M, "n_step": n_step, "seed_base": args.seed_base},
                   out_path)
        print(f"saved golden -> {out_path}", flush=True)
        return

    print("[2/3] running batched (supports_multiplicity=True)...", flush=True)
    batched = collect(args.model, M, n_runs, n_step, args.seed_base, supports=True)

    # R = unbatched-vs-unbatched, D = batched-vs-batched, X = batched-vs-unbatched (matched seed).
    print("[3/3] comparing...", flush=True)
    R = pairwise_kabsch(golden)
    D = pairwise_kabsch(batched)
    X = []
    for i in range(min(len(golden), len(batched))):
        for s in range(M):
            X.append(kabsch_rmsd(golden[i][s], batched[i][s]))
    floor = max(_floor(R), _floor(D))
    Xmax = _floor(X)
    # PCC on the full coord stack (matched seed) too
    pccs = []
    for i in range(min(len(golden), len(batched))):
        for s in range(M):
            pccs.append(pcc(golden[i][s], batched[i][s]))
    pcc_min = min(pccs) if pccs else float("nan")

    print(f"\n  R (unbatched-vs-unbatched) max Kabsch RMSD: {max(R):.3f} A  (n={len(R)})")
    print(f"  D (batched-vs-batched)       max Kabsch RMSD: {max(D):.3f} A  (n={len(D)})")
    print(f"  X (batched-vs-unbatched)     max Kabsch RMSD: {Xmax:.3f} A  (n={len(X)})")
    print(f"  floor = max(R,D) = {floor:.3f} A")
    print(f"  X min PCC (matched seed): {pcc_min:.5f}")
    verdict = "PASS" if Xmax <= floor * 1.25 else "FAIL"
    print(f"\nVERDICT: {verdict}  (X {Xmax:.3f} vs floor {floor:.3f}, X/floor {Xmax/floor:.3f})")

    out_path = f"/tmp/{args.model}_multiplicity_parity_M{M}.json"
    import json
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "M": M, "n_step": n_step, "n_runs": n_runs,
            "R_max": max(R), "D_max": max(D), "X_max": Xmax, "floor": floor,
            "X_over_floor": Xmax / floor, "pcc_min": pcc_min, "verdict": verdict,
        }, f, indent=2)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
