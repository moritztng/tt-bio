"""Where the 15.894 s of host template embedding goes at 848 residues.

Pass 6 of the port put the template embedding on host and cleared it against a 1.0 s bar at 208
residues. Pass 5 of this program measured it at 15.894 s at 848, 9.0 % of the fold. This splits it
at the four seams `TemplateEmbedding` already has, so the fix can be aimed rather than guessed:

  prologue   the 88-channel pair build (distogram, two one-hots, the two masks) and embedding2d
  pair_stack the 2 PairBlocks at c_t=64 -- the only O(L^3) part, and the part `AF2PairBlock`
             already knows how to run on card (`evoformer_order=False`)
  norm       output_norm
  attn       the pointwise attention over templates, degenerate at one template

Host torch only. No device, no accuracy claim: the inputs are the production featurisation of a
synthetic two-chain complex, same as fold_timing.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from fold_timing import feats_from_pdb, to_torch
    from tap_gate import DEFAULT_PARAMS
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict

    feats_np, binder_seq = feats_from_pdb(args.pdb)
    feats = {k: to_torch(v) for k, v in feats_np.items()}

    model = load_af2_model(load_af2_state_dict(args.params or DEFAULT_PARAMS), template=True)
    model.eval()
    template = model.template

    tokens = int(feats["seq_mask"].shape[0])
    dtype = torch.float32
    pair = torch.zeros(tokens, tokens, 128, dtype=dtype)
    mask_2d = feats["seq_mask"][:, None] * feats["seq_mask"][None, :]
    mask_2d = mask_2d.to(dtype)
    asym = feats["asym_id"]
    multichain_mask = (asym[:, None] == asym[None, :]).to(dtype)

    # The same seams `pair_representation` runs, timed one at a time. Single template, which is
    # what AF2-IG feeds it, so the template loop runs once.
    def legs() -> dict:
        from torch.nn import functional as F

        from tt_bio.af2_reference import ATOM_ORDER, TEMPLATE_DGRAM, dgram_from_positions

        out = {}
        t0 = time.perf_counter()
        pb_mask = feats["template_pseudo_beta_mask"][0].float()
        mask_pb = pb_mask[:, None] * pb_mask[None, :] * multichain_mask
        dgram = dgram_from_positions(feats["template_pseudo_beta"][0], *TEMPLATE_DGRAM)
        aatype = F.one_hot(feats["template_aatype"][0].long(), 22).to(dtype)
        atom_mask = feats["template_all_atom_mask"][0].float()
        bb = atom_mask[:, ATOM_ORDER["N"]] * atom_mask[:, ATOM_ORDER["CA"]] \
            * atom_mask[:, ATOM_ORDER["C"]]
        mask_bb = (bb[:, None] * bb[None, :] * multichain_mask).to(dtype)
        act = torch.cat([
            (dgram * mask_pb[..., None]).to(dtype),
            mask_pb.to(dtype)[..., None],
            aatype[None].expand(tokens, -1, -1),
            aatype[:, None].expand(-1, tokens, -1),
            torch.zeros(tokens, tokens, 3, dtype=dtype),
            mask_bb[..., None],
        ], dim=-1) * mask_bb[..., None]
        act = template.embedding2d(act)
        out["prologue_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for block in template.pair_stack:
            act = block(act, mask_2d)
        out["pair_stack_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        act = template.output_norm(act)
        out["norm_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        repr_ = torch.stack([act], dim=0)
        template_mask = feats["template_mask"].to(dtype)
        flat_query = pair.reshape(tokens * tokens, 1, 128)
        flat_templates = repr_.permute(1, 2, 0, 3).reshape(tokens * tokens, 1, 64)
        bias = 1e9 * (template_mask[None, None, None, :] - 1.0)
        emb = template.attn(flat_query, flat_templates, bias).reshape(tokens, tokens, 128)
        emb = emb * (template_mask.sum() > 0).to(emb.dtype)
        out["attn_s"] = time.perf_counter() - t0
        out["total_s"] = sum(out[k] for k in ("prologue_s", "pair_stack_s", "norm_s", "attn_s"))
        out["out_abs_mean"] = float(emb.abs().mean())
        return out

    with torch.no_grad():
        reps = [legs() for _ in range(args.reps)]
        whole = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            template(pair, feats, mask_2d, multichain_mask)
            whole.append(time.perf_counter() - t0)

    warm = reps[1:] or reps
    report = {
        "mode": "af2ig_template_split",
        "tokens": tokens,
        "binder_residues": len(binder_seq),
        "threads": torch.get_num_threads(),
        "reps": args.reps,
        "warm_median_s": {k: statistics.median([r[k] for r in warm])
                          for k in ("prologue_s", "pair_stack_s", "norm_s", "attn_s", "total_s")},
        "whole_forward_s": whole,
        "whole_forward_warm_median_s": statistics.median(whole[1:] or whole),
        "rows": reps,
    }
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
