"""Multiplicity-batching wall-clock benchmark for Protenix-v2 and OpenDDE.

Compares producing M samples two ways:
  before (per-sample loop): M independent n_sample=1 folds, seeds seed..seed+M-1.
  after  (batched):         one n_sample=M, max_parallel_samples=M fold, one seed stream.

Reports per-leg wall-clock + speedup. The diffusion leg is what carries the M dim
(the trunk/confidence are shared or per-sample in both paths); the full-fold speedup
is what a user sees. Run on a free card:
  TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150> PYTHONPATH=<worktree> \
    /home/ttuser/tt-bio/env/bin/python3 scripts/protenix_opendde_multiplicity_bench.py <model> --M 4
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
os.environ.setdefault("TT_BIO_LEASE_HOLDER", "worker:tt-bio-diffusion-multiplicity-batching")
import argparse, pickle, time, json
import torch, ttnn


def protenix_model():
    from tt_bio.protenix import Protenix
    from tt_bio.tenstorrent import get_device
    ife = pickle.load(open("/home/ttuser/protenix_ife_gold.pkl", "rb"))
    tg = pickle.load(open("/home/ttuser/protenix_trunkin_gold.pkl", "rb"))
    d = pickle.load(open("/home/ttuser/protenix_ref_out.pkl", "rb"))
    tfeat = d["intermediates"]["template_embedder"]["in"][0]
    F = ife["feat"]
    feats = {
        "ref_pos": F["ref_pos"], "ref_charge": F["ref_charge"], "ref_mask": F["ref_mask"],
        "ref_element": F["ref_element"], "ref_atom_name_chars": F["ref_atom_name_chars"],
        "d_lm": F["d_lm"], "v_lm": F["v_lm"], "atom_to_token_idx": F["atom_to_token_idx"],
        "restype": F["restype"], "profile": F["profile"], "deletion_mean": F["deletion_mean"],
        "mask_trunked": ife["mask_trunked"], "relp": tg["relp"], "token_bonds": tg["token_bonds"],
        "template_aatype": tfeat["template_aatype"], "template_distogram": tfeat["template_distogram"],
        "template_pseudo_beta_mask": tfeat["template_pseudo_beta_mask"],
        "template_unit_vector": tfeat["template_unit_vector"],
        "template_backbone_frame_mask": tfeat["template_backbone_frame_mask"],
        "msa": tfeat["msa"], "has_deletion": tfeat["has_deletion"],
        "deletion_value": tfeat["deletion_value"], "asym_id": tfeat["asym_id"],
    }
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    m = Protenix.load_from_checkpoint("/home/ttuser/protenix_ckpt/protenix-v2.pt",
                                      compute_kernel_config=ckc, device=dev)
    m.diffusion.supports_multiplicity = True
    return m, feats


def opendde_model():
    from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.tenstorrent import get_device
    SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
           "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    m = OpenDDE(load_opendde_checkpoint(), ckc, dev)
    m._protenix.diffusion.supports_multiplicity = True
    feats = build_complex_features([(SEQ, None, "protein")])
    return m, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["protenix", "opendde"])
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_step = args.n_step or (10 if args.model == "protenix" else 20)
    M = args.M
    if args.model == "protenix":
        model, feats = protenix_model()
        fold = lambda **kw: model.fold(feats, n_step=n_step, progress_fn=None, **kw)
    else:
        model, feats = opendde_model()
        fold = lambda **kw: model.fold(feats, n_step=n_step, n_cycles=2,
                                       return_confidence=False, **kw)

    # warm the device / weight caches with one tiny fold (excluded from timing)
    print(f"[warmup] one n_sample=1 fold...", flush=True)
    _ = fold(n_sample=1, seed=999)

    # before: M independent per-sample folds (the true unbatched loop)
    t0 = time.time()
    before = []
    for k in range(M):
        c = fold(n_sample=1, seed=args.seed + k)
        before.append(c)
    t_before = time.time() - t0
    print(f"[before] M={M} per-sample folds: {t_before:.2f}s  "
          f"({t_before/M:.2f}s/sample)  shapes {[tuple(c.shape) for c in before]}", flush=True)

    # after: one batched fold at multiplicity=M
    t0 = time.time()
    after = fold(n_sample=M, max_parallel_samples=M, seed=args.seed)
    t_after = time.time() - t0
    print(f"[after]  1 batched fold M={M}: {t_after:.2f}s  "
          f"({t_after/M:.2f}s/sample)  shape {tuple(after.shape)}", flush=True)

    speedup = t_before / t_after
    print(f"\nSPEEDUP: {speedup:.2f}x  ({t_before:.2f}s -> {t_after:.2f}s for {M} samples)")
    print(f"finite: before={all(torch.isfinite(c).all() for c in before)} "
          f"after={bool(torch.isfinite(after).all())}")

    out = {"model": args.model, "M": M, "n_step": n_step,
           "before_s": t_before, "after_s": t_after, "speedup": speedup}
    with open(f"/tmp/{args.model}_multiplicity_bench_M{M}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> /tmp/{args.model}_multiplicity_bench_M{M}.json")


if __name__ == "__main__":
    main()
