#!/usr/bin/env python3
"""Seed-to-seed plDDT spread of a shipped ESMFold2 checkpoint, at the perf page's own fold.

`site/data/perf-512aa.json`'s esmfold2-fast p150a cell reports plDDT 0.8987 against three NVIDIA
folds at 0.9148-0.9155 and leaves the 0.017 gap open. The two sides do not evaluate one function
twice: ESMFold2's structure head is a diffusion sampler, and the device path draws its noise from
CPU MT19937 while the GPU reference draws from CUDA Philox, so they are two independent draws.
This measures what one draw is worth. Same checkpoint, same target, same protocol, N seeds, one
card. The spread it reports is the scale any cross-backend plDDT difference has to be read
against.

Every seed folds the perf fixture the cell folded (`perf/size512/fixtures/cdk2x2_<size>.yaml`),
single-sequence: esmfold2-fast has no MSA encoder, so the fixture's a3m is not read either way.

Usage:
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD \
    /home/ttuser/tt-bio-dev/env/bin/python perf/esmfold2-fast-parity/plddt_seed_spread.py \
      --model esmfold2-fast --size 512 --seeds 0,1,2,3,4 --out <out.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# forward() kwargs prepare_input supplies (extras are dropped by name), as in
# scripts/esmfold2_e2e_parity.py.
_FORWARD_KEYS = {
    "token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type",
    "res_type", "token_bonds", "token_attention_mask", "ref_pos", "ref_element",
    "ref_charge", "ref_atom_name_chars", "ref_space_uid", "atom_attention_mask",
    "atom_to_token", "distogram_atom_idx", "deletion_mean", "msa", "has_deletion",
    "deletion_value", "msa_attention_mask", "input_ids",
}


def fixture_sequence(size: int) -> str:
    doc = yaml.safe_load((ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{size}.yaml").read_text())
    seqs = [e["protein"]["sequence"] for e in doc["sequences"] if "protein" in e]
    assert len(seqs) == 1, f"expected one protein chain, got {len(seqs)}"
    return seqs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2-fast", choices=("esmfold2", "esmfold2-fast"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    torch.set_grad_enabled(False)
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio._vendor.esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.weights import ARTIFACTS

    # The protocol the perf cell publishes: the model's own defaults, resolved the way
    # `tt-bio predict` resolves them rather than restated here.
    loops = _resolve_recycling_steps(None, args.model)
    steps = _resolve_sampling_steps(None, args.model)
    repo = ARTIFACTS[args.model].repo
    seq = fixture_sequence(args.size)
    print(f"{args.model} ({repo}) on cdk2x2_{args.size} L={len(seq)}, "
          f"loops={loops} requested_steps={steps}, seeds={seeds}", flush=True)

    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()
    model = ESMFold2Model.from_pretrained(repo, load_esmc=False).eval()
    blocks = model.config.folding_trunk.n_layers
    patch_esmfold2(model, esmc_repo=args.esmc_repo)
    model._esmc = esmc

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    feats, _chain_infos = ESMFold2InputBuilder().prepare_input(spi, seed=7, device=model.device)
    lm_hs = compute_lm_hidden_states(esmc, feats["input_ids"], feats["asym_id"],
                                     feats["residue_index"], feats["mol_type"],
                                     feats["token_attention_mask"])
    fwd = {k: v for k, v in feats.items() if k in _FORWARD_KEYS}

    rows = []
    for s in seeds:
        # The fold seed reaches the device sampler's private generator through
        # torch.initial_seed() (_StructureHeadAdapter.sample), so manual_seed IS the draw.
        torch.manual_seed(s)
        t0 = time.time()
        with torch.no_grad():
            out = model(**fwd, lm_hidden_states=lm_hs, num_loops=loops,
                        num_sampling_steps=steps, num_diffusion_samples=1)
        coords = out["sample_atom_coords"].float().cpu().numpy()
        row = {"seed": s, "plddt": float(out["plddt"].float().mean()),
               "ptm": float(out["ptm"].float().mean()),
               "coords_sha256": hashlib.sha256(coords.tobytes()).hexdigest()[:16],
               "seconds": round(time.time() - t0, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    plddt = [r["plddt"] for r in rows]
    distinct = len({r["coords_sha256"] for r in rows})
    report = {
        "model": args.model, "repo": repo, "trunk_blocks": blocks,
        "target": f"cdk2x2_{args.size}", "L": len(seq),
        "recycling_steps": loops, "requested_sampling_steps": steps,
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "seeds": seeds, "per_seed": rows,
        "plddt_min": min(plddt), "plddt_max": max(plddt),
        "plddt_spread": max(plddt) - min(plddt),
        "plddt_mean": sum(plddt) / len(plddt),
        "distinct_structures": distinct,
    }
    assert distinct == len(seeds), (
        f"only {distinct} distinct structures across {len(seeds)} seeds: the seed is not reaching "
        "the sampler, so this spread would be an artifact and not the model's draw-to-draw range")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nplDDT {report['plddt_min']:.4f}-{report['plddt_max']:.4f}, "
          f"spread {report['plddt_spread']:.4f} over {len(seeds)} seeds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
