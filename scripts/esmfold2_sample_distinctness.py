"""ESMFold2 multi-sample distinctness gate (Phase-4, empirical).

Verifies the campaign's "1 trunk pass, N diffusion samples" protocol on device:
  1. pairwise CA-RMSD across the N samples (not N copies of one structure);
  2. per-sample plddt/ptm/iptm spread;
  3. determinism: repeating the fold seed gives bit-identical coords (torch.equal); a
     different seed gives different coords (the fold seed reaches the sampler's private
     RNG via torch.initial_seed() in _StructureHeadAdapter.sample);
  4. chunk-seed advance: TT_ESMFOLD2_DIFFUSION_BUDGET set so N samples take 2 chunks — the
     chunked set stays bit-reproducible and chunk-2 samples differ from chunk-1 (chunk
     bases are seed+done, tt_bio/esmfold2_runtime.py).

Usage:
  TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/esmfold2_sample_distinctness.py \
      [--protein ubiquitin] [--loops 10] [--steps 100] [--samples 4] [--seed 42] [--out /tmp/d.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import torch

PROTEINS = {
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",                                              # 20
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


def build_features(seq, seed, device):
    from tt_bio._vendor.esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    feats, _chain_infos = ESMFold2InputBuilder().prepare_input(spi, seed=seed, device=device)
    return feats


def run_fold(model, feats, lm_hs, *, loops, steps, samples, seed):
    fwd = {k: v for k, v in feats.items() if k in _FORWARD_KEYS}
    torch.manual_seed(seed)  # the fold seed: _StructureHeadAdapter reads torch.initial_seed()
    with torch.no_grad():
        return model(**fwd, lm_hidden_states=lm_hs, num_loops=loops,
                     num_sampling_steps=steps, num_diffusion_samples=samples)


def ca_mask(feats):
    """ref_atom_name_chars holds (ord-32) char indices [B, N, 4], 0-padded
    (encode_atom_name, vendored prepare_input.py); CA = 'CA'."""
    chars = feats["ref_atom_name_chars"]
    if chars.dim() == 4:  # already one-hot
        chars = chars.argmax(-1)
    mask = []
    for codes in chars[0].long().tolist():
        name = "".join(chr(c + 32) for c in codes if c > 0)
        mask.append(name == "CA")
    return torch.tensor(mask)


def kabsch_rmsd(a, b):
    import tt_bio.esmfold2 as E
    w = torch.ones(1, a.shape[0])
    aligned = E._weighted_rigid_align(a.unsqueeze(0), b.unsqueeze(0), w, w)[0]
    return (aligned - b).pow(2).sum(-1).mean().sqrt().item()


def summarize_leg(out, cam):
    coords = out["sample_atom_coords"].float()          # [N, n_atoms, 3]
    ca = coords[:, cam]
    n = coords.shape[0]
    pair = {f"{i}-{j}": kabsch_rmsd(ca[i], ca[j]) for i, j in itertools.combinations(range(n), 2)}
    plddt = out["plddt"].float().reshape(n, -1).mean(-1)
    iptm = out.get("iptm")
    return {
        "pairwise_ca_rmsd": pair,
        "plddt_per_sample": [round(float(x), 4) for x in plddt],
        "ptm_per_sample": [round(float(x), 4) for x in out["ptm"].float().reshape(n)],
        "iptm_per_sample": ([round(float(x), 4) for x in iptm.float().reshape(n)]
                            if iptm is not None else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default="ubiquitin")
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--esmfold2_repo", default="biohub/ESMFold2")
    ap.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    ap.add_argument("--out", default="/tmp/ef2_distinctness/summary.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2
    import tt_bio.esmfold2 as E

    seq = PROTEINS[args.protein]
    print(f"=== {args.protein} (L={len(seq)}): {args.samples} samples, loops={args.loops} "
          f"steps={args.steps}, seed={args.seed} ===", flush=True)

    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()
    print("loading ttnn model ...", flush=True)
    model = ESMFold2Model.from_pretrained(args.esmfold2_repo, load_esmc=False).eval()
    patch_esmfold2(model, esmc_repo=args.esmc_repo)
    model._esmc = esmc

    feats = build_features(seq, 7, model.device)
    lm_hs = compute_lm_hidden_states(
        esmc, feats["input_ids"], feats["asym_id"], feats["residue_index"],
        feats["mol_type"], feats["token_attention_mask"])
    cam = ca_mask(feats)
    n_tokens = int(feats["token_index"].shape[1])
    print(f"CA atoms: {int(cam.sum())} / {cam.shape[0]}", flush=True)

    # Count sampler chunk invocations via the diffusion progress hook (one i=0 per chunk).
    _orig_report = E.report_progress
    _chunk_starts = []

    def _report(stage, step=0, total=0):
        if stage == "diffusion" and step == 0:
            _chunk_starts.append(time.time())
        return _orig_report(stage, step, total)
    E.report_progress = _report

    def leg(seed, budget=None):
        if budget is None:
            os.environ.pop("TT_ESMFOLD2_DIFFUSION_BUDGET", None)
        else:
            os.environ["TT_ESMFOLD2_DIFFUSION_BUDGET"] = str(budget)
        t0 = time.time()
        n_chunks0 = len(_chunk_starts)
        out = run_fold(model, feats, lm_hs, loops=args.loops, steps=args.steps,
                       samples=args.samples, seed=seed)
        return out, time.time() - t0, len(_chunk_starts) - n_chunks0

    os.environ.pop("TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG", None)  # production RNG path
    print("leg A: seed=42 (one chunk) ...", flush=True)
    out42a, t_a, chunks_a = leg(args.seed)
    print("leg B: seed=42 repeat ...", flush=True)
    out42b, t_b, _ = leg(args.seed)
    print("leg C: seed=43 ...", flush=True)
    out43, t_c, _ = leg(args.seed + 1)
    budget2 = 2 * n_tokens * n_tokens
    print(f"leg D: seed=42 chunked (budget={budget2} -> 2 chunks) ...", flush=True)
    out42c1, t_d1, chunks_d1 = leg(args.seed, budget=budget2)
    out42c2, t_d2, chunks_d2 = leg(args.seed, budget=budget2)
    os.environ.pop("TT_ESMFOLD2_DIFFUSION_BUDGET", None)

    c42a = out42a["sample_atom_coords"].float()
    c42b = out42b["sample_atom_coords"].float()
    c43 = out43["sample_atom_coords"].float()
    cc1 = out42c1["sample_atom_coords"].float()
    cc2 = out42c2["sample_atom_coords"].float()

    report = {
        "protein": args.protein, "L": len(seq), "n_tokens": n_tokens,
        "loops": args.loops, "requested_steps": args.steps,
        "samples": args.samples, "seed": args.seed,
        "chunks_legA": chunks_a, "chunks_legD": [chunks_d1, chunks_d2],
        "runtime_s": {"A": round(t_a, 1), "B": round(t_b, 1), "C": round(t_c, 1),
                      "D1": round(t_d1, 1), "D2": round(t_d2, 1)},
        "legA_seed42": summarize_leg(out42a, cam),
        "legC_seed43_pairwise": summarize_leg(out43, cam)["pairwise_ca_rmsd"],
        "legD_chunked": summarize_leg(out42c1, cam),
        "determinism": {
            "seed42_repeat_bit_identical": bool(torch.equal(c42a, c42b)),
            "seed42_repeat_max_abs_diff": float((c42a - c42b).abs().max()),
            "seed43_differs": not bool(torch.equal(c42a, c43)),
            "seed43_max_abs_diff": float((c42a - c43).abs().max()),
        },
        "chunk_seed_advance": {
            "chunked_repeat_bit_identical": bool(torch.equal(cc1, cc2)),
            "chunked_repeat_max_abs_diff": float((cc1 - cc2).abs().max()),
            "chunked_vs_onechunk_max_abs_diff": float((cc1 - c42a).abs().max()),
            "note": "chunk-2 rows re-base the RNG (seed+done), so the chunked set differs "
                    "from the one-chunk set by design; it must be reproducible run-to-run "
                    "and chunk-2 samples must differ from chunk-1 (see legD pairwise).",
        },
    }
    print(json.dumps(report, indent=2), flush=True)

    pair_vals = list(report["legA_seed42"]["pairwise_ca_rmsd"].values())
    assert min(pair_vals) > 0.5, f"samples collapsed: min pairwise CA-RMSD {min(pair_vals):.3f} A"
    assert report["determinism"]["seed42_repeat_bit_identical"], "seed-42 repeat not bit-identical"
    assert report["determinism"]["seed43_differs"], "seed 43 did not change the samples"
    assert report["chunk_seed_advance"]["chunked_repeat_bit_identical"], "chunked run not reproducible"
    assert chunks_d1 == 2 and chunks_d2 == 2, f"budget did not force 2 chunks: {chunks_d1},{chunks_d2}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
