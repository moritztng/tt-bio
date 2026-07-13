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


def run_esmc_parity(
    name: str,
    seq: str = DEFAULT_SEQ,
    *,
    fast: bool = False,
    pcc_threshold: float = 0.99,
    verbose: bool = True,
) -> dict:
    """Run the shipped ESMC embedding path vs the reference esm ESMC on `seq`.

    This is the on-hardware accuracy check that gates the ESMC embedding
    capability — including the fused-RoPE numerics change in ``esmc._rope``,
    which the bucketed embed path always takes (``BUCKET=64`` pads L to a
    tile-aligned length, so ``_rope`` selects ``ttnn.experimental.rotary_embedding``).

    Returns a metrics dict; ``ok`` is True iff per-residue PCC >= ``pcc_threshold``.
    Reused by ``scripts/release_gate.py``'s ESMC leg — do not re-derive here.
    """
    torch.set_grad_enabled(False)

    # --- real trained weights (downloads on first use) ---
    from huggingface_hub import hf_hub_download

    _cfg, repo_id, wpath = tt_esmc.CONFIGS[name]
    if verbose:
        print(f"Fetching {name} weights from {repo_id} …", flush=True)
    path = hf_hub_download(repo_id, wpath)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    # --- reference (CPU torch) ---
    if verbose:
        print("Building reference esm ESMC …", flush=True)
    ref = load_reference(name, sd)
    tokens = tt_esmc.tokenize(seq)
    ref_logits, ref_emb = ref(tokens)             # [1,L+2,64], [1,L+2,d]
    ref_per_res = ref_emb[0][1:-1].numpy()        # strip <cls>/<eos>
    ref_pooled = ref_per_res.mean(axis=0)
    ref_logits_res = ref_logits[0][1:-1].numpy()

    # --- shipped tt embedding API (load_esmc downloads + loads the same .pth) ---
    if verbose:
        print(f"Loading tt ESMC on device{' (fast)' if fast else ''} …", flush=True)
    model = tt_esmc.load_esmc(name, fast=fast)
    emb = tt_esmc.embed_sequences(model, {"ubq": seq},
                                  return_logits=True, pool="mean")[0]

    per_res_pcc = pcc(emb.per_residue, ref_per_res)
    pooled_pcc = pcc(emb.pooled, ref_pooled)
    logits_pcc = pcc(emb.logits, ref_logits_res)
    argmax_agree = float((emb.logits.argmax(-1) == ref_logits_res.argmax(-1)).mean())
    ok = per_res_pcc >= pcc_threshold

    res = {"model": name, "seq_len": len(seq), "fast": fast,
           "per_res_pcc": per_res_pcc, "pooled_pcc": pooled_pcc,
           "logits_pcc": logits_pcc, "argmax_agree": argmax_agree,
           "threshold": pcc_threshold, "ok": ok}

    if verbose:
        print("\n=== ESMC embedding parity (tt vs reference esm) ===")
        print(f"model            : {name}  (fast={fast})")
        print(f"sequence length  : {len(seq)} residues")
        print(f"per-residue PCC  : {per_res_pcc:.5f}")
        print(f"pooled(mean) PCC : {pooled_pcc:.5f}")
        print(f"logits PCC       : {logits_pcc:.5f}")
        print(f"argmax agreement : {argmax_agree:.4f}")
        print(f"\n{'PASS' if ok else 'FAIL'}: per-residue PCC "
              f"{per_res_pcc:.5f} vs threshold {pcc_threshold}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.CONFIGS))
    ap.add_argument("--seq", default=DEFAULT_SEQ)
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    res = run_esmc_parity(args.model, args.seq,
                          fast=args.fast, pcc_threshold=args.pcc_threshold)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
