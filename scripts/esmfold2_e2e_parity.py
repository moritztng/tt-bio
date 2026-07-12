"""End-to-end ESMFold2 parity: ttnn on-device pipeline vs the vendored torch reference.

Both paths share the *same* featurization (``ESMFold2InputBuilder.prepare_input``)
and the *same* language-model hidden states (computed once with the ttnn ESMC-6B
and passed to both ``forward``s via ``lm_hidden_states=``). That isolates the
ESMFold2 neural port under test -- inputs embedder, relpos, folding trunk (48 or
24 blocks), parcae recurrence, diffusion structure head, distogram + confidence
heads -- from the separately-validated ESMC-6B port (tests/test_esmc.py) and from
featurization. The torch reference is ``ESMFold2Model`` left unpatched; the test
path is the same model after ``patch_esmfold2`` (every learnable submodule -> ttnn).

Diffusion noise is not bit-identical across the torch and ttnn samplers (independent
RNG streams per backend), so a single seed pair cannot separate genuine port drift
from sampling variance. Instead each protein is folded at several sampler seeds on
*both* backends, giving three coordinate-metric distributions -- reference-vs-
reference (R), device-vs-device (D) and device-vs-reference (X) -- summarized with
the same statistical core (`pharma_parity.summarize` / `noise_floor_verdict`) the
rest of the parity benchmark uses. Parity holds when X sits within max(R, D).

Reported per protein:
  * plddt_pcc / plddt_mae   -- per-residue confidence, the metric ESMFold ranks on
  * distogram_pcc, ptm      -- sampler-independent (computed once, first seed)
  * kabsch_rmsd, coord_dm_pcc R/D/X distributions across the sampler seeds

Usage:
  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=1 \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/esmfold2_e2e_parity.py \
      [--fast] [--proteins trpcage,gb1] [--steps 20] [--loops 3] \
      [--seeds 0,1,2] [--out /tmp/x.json]
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from pharma_parity import noise_floor_verdict, summarize

# Representative single-domain proteins (no MSA): short, medium, medium-long.
PROTEINS = {
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",                                              # 20
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",             # 56
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 76
}

# forward() kwargs that prepare_input supplies (extras are dropped by name).
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


def dist_matrix(x):  # x: [n,3] -> [n,n]
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
    """RMSD (Angstrom) of a_coords onto b_coords after weighted rigid alignment."""
    import tt_bio.esmfold2 as E
    a = a_coords.float(); b = b_coords.float()
    aligned = E._weighted_rigid_align(a.unsqueeze(0), b.unsqueeze(0), atom_mask, atom_mask)[0]
    return (aligned - b).pow(2).sum(-1).mean().sqrt().item()


def pair_metrics(a, b, atom_mask):
    ac = a["sample_atom_coords"][0].float()
    bc = b["sample_atom_coords"][0].float()
    return kabsch_rmsd(ac, bc, atom_mask), pcc(dist_matrix(ac), dist_matrix(bc))


def compare_multiseed(ref_runs: dict, tt_runs: dict, atom_mask, seeds):
    """R (ref-vs-ref), D (tt-vs-tt), X (tt-vs-ref) distributions over sampler seeds."""
    r_rmsd, r_pcc = [], []
    for s1, s2 in itertools.combinations(seeds, 2):
        rmsd, p = pair_metrics(ref_runs[s1], ref_runs[s2], atom_mask)
        r_rmsd.append(rmsd); r_pcc.append(1 - p)
    d_rmsd, d_pcc = [], []
    for s1, s2 in itertools.combinations(seeds, 2):
        rmsd, p = pair_metrics(tt_runs[s1], tt_runs[s2], atom_mask)
        d_rmsd.append(rmsd); d_pcc.append(1 - p)
    x_rmsd, x_pcc = [], []
    for s1, s2 in itertools.product(seeds, seeds):
        rmsd, p = pair_metrics(tt_runs[s1], ref_runs[s2], atom_mask)
        x_rmsd.append(rmsd); x_pcc.append(1 - p)
    return {
        "kabsch_rmsd": noise_floor_verdict(x_rmsd, r_rmsd, d_rmsd, "kabsch_rmsd"),
        "coord_dm_1mpcc": noise_floor_verdict(x_pcc, r_pcc, d_pcc, "1-coord_dm_pcc"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--proteins", default="trpcage,gb1,ubiquitin")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated sampler seeds, run on both backends")
    ap.add_argument("--feature_seed", type=int, default=7, help="seed for featurization (not the sampler)")
    ap.add_argument("--esmfold2_repo", default="biohub/ESMFold2")
    ap.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    ap.add_argument("--out", default="/tmp/ef2_parity/summary.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2

    names = [n.strip() for n in args.proteins.split(",") if n.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Shared ttnn ESMC-6B (loaded once): produces the LM hidden states fed to BOTH paths.
    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()

    # Torch reference model (unpatched). Real ESMFold2 weights, no CPU ESMC (we
    # inject shared LM states instead).
    print("loading torch reference model ...", flush=True)
    ref_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()

    # ttnn model: same weights, every submodule swapped to ttnn.
    print(f"loading ttnn model (fast={args.fast}) ...", flush=True)
    tenstorrent.set_fast_mode(args.fast)
    tt_model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    patch_esmfold2(tt_model, esmc_repo=args.esmc_repo)
    tt_model._esmc = esmc  # reuse the already-loaded ESMC (LM states are passed in anyway)

    results = []
    for name in names:
        seq = PROTEINS[name]
        print(f"\n=== {name} (L={len(seq)}), seeds={seeds} ===", flush=True)
        feats = build_features(seq, args.feature_seed, ref_model.device)
        lm_hs = compute_lm_hidden_states(
            esmc, feats["input_ids"], feats["asym_id"], feats["residue_index"],
            feats["mol_type"], feats["token_attention_mask"])
        atom_mask = feats["atom_attention_mask"].float()
        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0)

        ref_runs, tt_runs = {}, {}
        for s in seeds:
            print(f"  ref seed={s} ...", flush=True)
            ref_runs[s] = run_forward(ref_model, feats, lm_hs, loops=args.loops, steps=args.steps, samples=1, seed=s)
            print(f"  device seed={s} ...", flush=True)
            tt_runs[s] = run_forward(tt_model, feats, lm_hs, loops=args.loops, steps=args.steps, samples=1, seed=s)

        base_ref, base_tt = ref_runs[seeds[0]], tt_runs[seeds[0]]
        verdicts = compare_multiseed(ref_runs, tt_runs, atom_mask, seeds)
        m = dict(
            protein=name, L=len(seq), n_seeds=len(seeds),
            plddt_pcc=pcc(base_tt["plddt"], base_ref["plddt"]),
            plddt_mae=(base_tt["plddt"].float() - base_ref["plddt"].float()).abs().mean().item(),
            distogram_pcc=pcc(base_tt["distogram_logits"], base_ref["distogram_logits"]),
            ptm_tt=float(base_tt["ptm"].mean()), ptm_ref=float(base_ref["ptm"].mean()),
            **verdicts,
        )
        results.append(m)
        print(json.dumps(m, indent=2), flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
