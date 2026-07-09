"""On-hardware parity gate for the standalone ESMC embedding API.

Runs the *shipped* embedding path (``tt_bio.esmc.load_esmc`` + ``embed_sequences``)
against the reference esm ESMC on real trained weights, and reports per-residue
embedding PCC — the accuracy metric that gates this capability. Also reports
pooled-embedding PCC and (300m/600m) sequence-head argmax agreement.

Usage (single device):
    ESM_ROOT=/path/to/esm TT_VISIBLE_DEVICES=0 \
        python3 scripts/esmc_embed_parity.py --model esmc-300m

The reference builder lives in tests/esmc_reference.py and needs the esm clone
on ESM_ROOT. Real weights download from the biohub HF repo on first use.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from tt_bio import esmc as tt_esmc  # noqa: E402

# Human ubiquitin (76 aa) — a real, well-folded protein the LM is confident on.
DEFAULT_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def load_reference(name: str, sd: dict):
    """Build the reference esm ESMC for `name` and load `sd` into it."""
    from esmc_reference import ESMCReference  # noqa: E402  (tests/ on path)

    cfg = tt_esmc.CONFIGS[name][0]
    ref = ESMCReference(**cfg).eval()
    ref.load_state_dict(sd, strict=False)
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.CONFIGS))
    ap.add_argument("--seq", default=DEFAULT_SEQ)
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    # --- real trained weights (downloads on first use) ---
    from huggingface_hub import hf_hub_download

    _cfg, repo_id, wpath = tt_esmc.CONFIGS[args.model]
    print(f"Fetching {args.model} weights from {repo_id} …", flush=True)
    path = hf_hub_download(repo_id, wpath)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    # --- reference (CPU torch) ---
    print("Building reference esm ESMC …", flush=True)
    ref = load_reference(args.model, sd)
    tokens = tt_esmc.tokenize(args.seq)
    ref_logits, ref_emb = ref(tokens)             # [1,L+2,64], [1,L+2,d]
    ref_per_res = ref_emb[0][1:-1].numpy()        # strip <cls>/<eos>
    ref_pooled = ref_per_res.mean(axis=0)
    ref_logits_res = ref_logits[0][1:-1].numpy()

    # --- shipped tt embedding API (load_esmc downloads + loads the same .pth) ---
    print(f"Loading tt ESMC on device{' (fast)' if args.fast else ''} …", flush=True)
    model = tt_esmc.load_esmc(args.model, fast=args.fast)
    emb = tt_esmc.embed_sequences(model, {"ubq": args.seq},
                                  return_logits=True, pool="mean")[0]

    # --- metrics ---
    per_res_pcc = pcc(emb.per_residue, ref_per_res)
    pooled_pcc = pcc(emb.pooled, ref_pooled)
    logits_pcc = pcc(emb.logits, ref_logits_res)
    argmax_agree = float((emb.logits.argmax(-1) == ref_logits_res.argmax(-1)).mean())

    print("\n=== ESMC embedding parity (tt vs reference esm) ===")
    print(f"model            : {args.model}  (fast={args.fast})")
    print(f"sequence length  : {len(args.seq)} residues")
    print(f"per-residue PCC  : {per_res_pcc:.5f}")
    print(f"pooled(mean) PCC : {pooled_pcc:.5f}")
    print(f"logits PCC       : {logits_pcc:.5f}")
    print(f"argmax agreement : {argmax_agree:.4f}")

    ok = per_res_pcc >= args.pcc_threshold
    print(f"\n{'PASS' if ok else 'FAIL'}: per-residue PCC "
          f"{per_res_pcc:.5f} vs threshold {args.pcc_threshold}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
