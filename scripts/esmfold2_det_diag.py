"""ESMFold2 lysozyme sampler-determinism diagnostic.

Goal: separate three candidate causes of the lysozyme L129 PASS-caveated
residual (committed X = 0.130 A, floor R = 0.095 A, D = 0.077 A, X/floor 1.37):

  (a) the diffusion sampler's RNG stream differs torch-vs-ttnn even at the same
      seed (independent randn implementations) -> "sampler stochasticity";
  (b) ttnn device run-to-run nondeterminism (the documented parallel-reduction
      confound) -> NOT sampler stochasticity, a different problem;
  (c) a genuine bf16 precision / algorithm mismatch.

Decisive test: run the DEVICE N times at a FIXED seed. `run_forward` calls
`torch.manual_seed(seed)` before every forward, so the diffusion sampler's
`torch.randn` draws (initial coords + per-step noise) are byte-identical across
those N runs. Any CA-RMSD spread between them is therefore pure ttnn run-to-run
nondeterminism (b), with the RNG stream (a) held fixed. We also run the REF N
times at the fixed seed as a determinism sanity check (torch CPU -> ~0).

We then read D_det (device fixed-seed run-to-run floor) against the committed
cross X and the cross-seed device floor D:
  - D_det ~ X  -> residual dominated by ttnn run-to-run nondeterminism (b).
  - D_det ~ 0  -> device is deterministic at fixed seed; residual is (a) or (c),
                  and a shared-draws test is the next lever.

Reuses the esmfold2_e2e_parity harness setup (shared LM hidden states, same
featurization, same kabsch_rmsd / pair_metrics) so the numbers are directly
comparable to the committed leg.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import torch

from pharma_parity import noise_floor_verdict, summarize  # noqa: F401 (parity-sacred: same core)


PROTEINS = {
    "lysozyme": ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDG"
                 "RTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",
}

_FORWARD_KEYS = {
    "token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type",
    "res_type", "token_bonds", "token_attention_mask", "ref_pos", "ref_element",
    "ref_charge", "ref_atom_name_chars", "ref_space_uid", "atom_attention_mask",
    "atom_to_token", "distogram_atom_idx", "deletion_mean", "msa", "has_deletion",
    "deletion_value", "msa_attention_mask", "input_ids",
}


def pcc(a, b) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def dist_matrix(x):
    return torch.cdist(x.float(), x.float())


def build_features(seq, seed, device):
    from tt_bio._vendor.esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    feats, _chain = ESMFold2InputBuilder().prepare_input(spi, seed=seed, device=device)
    return feats


def run_forward(model, feats, lm_hs, *, loops, steps, samples, seed=0):
    fwd = {k: v for k, v in feats.items() if k in _FORWARD_KEYS}
    torch.manual_seed(seed)  # seeds the (torch) diffusion sampler's global RNG
    with torch.no_grad():
        return model(**fwd, lm_hidden_states=lm_hs, num_loops=loops,
                     num_sampling_steps=steps, num_diffusion_samples=samples)


def kabsch_rmsd(a_coords, b_coords, atom_mask):
    import tt_bio.esmfold2 as E
    a = a_coords.float(); b = b_coords.float()
    aligned = E._weighted_rigid_align(a.unsqueeze(0), b.unsqueeze(0), atom_mask, atom_mask)[0]
    m = atom_mask[0] > 0.5
    return (aligned[m] - b[m]).pow(2).sum(-1).mean().sqrt().item()


def pair_rmsd(a, b, atom_mask):
    ac = a["sample_atom_coords"][0].float()
    bc = b["sample_atom_coords"][0].float()
    return kabsch_rmsd(ac, bc, atom_mask)


def floor(vals):
    import statistics
    if not vals:
        return None
    return statistics.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default="lysozyme")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--n", type=int, default=4, help="repeats at the fixed seed per backend")
    ap.add_argument("--fixed_seed", type=int, default=0)
    ap.add_argument("--feature_seed", type=int, default=7)
    ap.add_argument("--esmfold2_repo", default="biohub/ESMFold2")
    ap.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="/tmp/ef2_det/det.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2

    seq = PROTEINS[args.protein]
    print(f"=== {args.protein} (L={len(seq)}) fixed_seed={args.fixed_seed} n={args.n} ===", flush=True)

    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()

    print("loading torch reference model ...", flush=True)
    ref_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    print(f"loading ttnn model (fast={args.fast}) ...", flush=True)
    tenstorrent.set_fast_mode(args.fast)
    tt_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    patch_esmfold2(tt_model, esmc_repo=args.esmc_repo)
    tt_model._esmc = esmc

    feats = build_features(seq, args.feature_seed, ref_model.device)
    lm_hs = compute_lm_hidden_states(
        esmc, feats["input_ids"], feats["asym_id"], feats["residue_index"],
        feats["mol_type"], feats["token_attention_mask"])
    atom_mask = feats["atom_attention_mask"].float()
    if atom_mask.dim() == 1:
        atom_mask = atom_mask.unsqueeze(0)

    ref_runs, tt_runs = [], []
    for i in range(args.n):
        print(f"  ref  run#{i} seed={args.fixed_seed} ...", flush=True)
        ref_runs.append(run_forward(ref_model, feats, lm_hs, loops=args.loops,
                                    steps=args.steps, samples=1, seed=args.fixed_seed))
        print(f"  dev  run#{i} seed={args.fixed_seed} ...", flush=True)
        tt_runs.append(run_forward(tt_model, feats, lm_hs, loops=args.loops,
                                   steps=args.steps, samples=1, seed=args.fixed_seed))

    # Fixed-seed run-to-run floors (the determinism test).
    r_det = [pair_rmsd(ref_runs[i], ref_runs[j], atom_mask)
             for i, j in itertools.combinations(range(args.n), 2)]
    d_det = [pair_rmsd(tt_runs[i], tt_runs[j], atom_mask)
             for i, j in itertools.combinations(range(args.n), 2)]
    # Matched-seed cross (device fixed_seed vs ref fixed_seed), all n*n pairs.
    x_diag = [pair_rmsd(tt_runs[i], ref_runs[j], atom_mask)
              for i, j in itertools.product(range(args.n), range(args.n))]

    def stats(v):
        import statistics
        if not v:
            return None
        return {"mean": statistics.mean(v), "min": min(v), "max": max(v),
                "stdev": statistics.pstdev(v), "n": len(v)}

    out = {
        "protein": args.protein, "L": len(seq), "fixed_seed": args.fixed_seed,
        "n_repeats": args.n, "steps": args.steps, "loops": args.loops,
        "R_det_ref_fixed_seed_self": stats(r_det),   # expect ~0 (torch CPU deterministic)
        "D_det_dev_fixed_seed_self": stats(d_det),   # the ttnn run-to-run nondeterminism floor
        "X_diag_dev_fixed_seed_vs_ref_fixed_seed": stats(x_diag),  # matched-seed cross
        "verdict_logic": (
            "D_det ~ X_diag => ttnn run-to-run nondeterminism dominates (residual is NOT "
            "sampler RNG). D_det ~ 0 => device deterministic at fixed seed; residual is "
            "torch-vs-ttnn RNG stream or bf16 precision (needs shared-draws test)."
        ),
    }
    print(json.dumps(out, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
