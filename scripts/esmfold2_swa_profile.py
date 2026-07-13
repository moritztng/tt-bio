"""Sub-op timing breakdown inside ESMFold2's SWA atom transformer.

Wraps the SWA atom block's elementwise piles (apply_rotary, rms_norm-based
_modulate, SDPA, projections, SwiGLU FFN) with device-sync timers, aggregated
over a real fold. Quantifies whether the ESMFold2-specific SWA surface has a
dispatch-bound pile (rotary / adaLN modulation) large enough to be a real
fusion win. Read-only w.r.t. the model.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import ttnn

SEQS = {
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "prot": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", choices=SEQS, default="prot")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    import tt_bio.esmfold2 as E
    import tt_bio.esmc as Esmc
    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex

    model = load_ttnn_esmfold2()
    seq = SEQS[args.protein]
    device = model.structure_head.m.tt_device

    buckets = {k: [] for k in [
        "swa_block", "swa_attn", "modulate", "rotary", "rms_norm", "sdpa",
        "attn_projs", "ffn", "scatter_gather",
        "atom_encoder_full", "atom_decoder_full",
    ]}

    def sync():
        ttnn.synchronize_device(device)

    def wrap_cls(cls, name, key):
        orig = getattr(cls, name)

        def w(self, *a, **kw):
            sync(); t0 = time.perf_counter()
            r = orig(self, *a, **kw)
            sync(); buckets[key].append(time.perf_counter() - t0)
            return r
        setattr(cls, name, w)

    def wrap_fn(owner, name, key):
        orig = getattr(owner, name)

        def w(*a, **kw):
            sync(); t0 = time.perf_counter()
            r = orig(*a, **kw)
            sync(); buckets[key].append(time.perf_counter() - t0)
            return r
        setattr(owner, name, w)

    wrap_cls(E.SWAAtomBlock, "__call__", "swa_block")
    wrap_cls(E.SWAAtomBlock, "_modulate", "modulate")
    wrap_cls(E.SWAAttention, "__call__", "swa_attn")
    wrap_fn(E, "apply_rotary", "rotary")  # esmfold2 imports apply_rotary into its ns
    wrap_fn(E, "_rms_norm", "rms_norm")
    wrap_fn(E, "_sdpa_bf16", "sdpa")

    # Scatter/gather matmuls (atom<->token aggregation) in encoder/decoder.
    wrap_cls(E.AtomEncoder, "__call__", "atom_encoder_full")
    wrap_cls(E.AtomDecoder, "__call__", "atom_decoder_full")

    def one():
        for v in buckets.values():
            v.clear()
        sync(); t0 = time.perf_counter()
        from tt_bio.esmfold2_runtime import fold_complex as fc
        fc(model, [("A", seq)], num_loops=args.loops,
           num_sampling_steps=args.steps, num_diffusion_samples=1, seed=0)
        sync(); return time.perf_counter() - t0

    print(f"[warmup {args.protein} L={len(seq)}] ...", flush=True)
    one()
    print(f"[timed fold] ...", flush=True)
    total = one()

    def s(k):
        return sum(buckets[k]) if buckets.get(k) else 0.0

    swa_block = s("swa_block")
    out = {
        "model": "esmfold2", "tokens": len(seq), "loops": args.loops,
        "sampling_steps": args.steps, "timed_total_s": total,
        "swa_block_total_s": swa_block, "swa_block_share": swa_block / total,
        "within_swa": {
            "rotary_s": s("rotary"), "rotary_share_of_fold": s("rotary") / total,
            "modulate_s": s("modulate"), "modulate_share_of_fold": s("modulate") / total,
            "rms_norm_s": s("rms_norm"), "rms_norm_share_of_fold": s("rms_norm") / total,
            "sdpa_s": s("sdpa"), "sdpa_share_of_fold": s("sdpa") / total,
            "swa_attn_s": s("swa_attn"), "swa_attn_share_of_fold": s("swa_attn") / total,
            "atom_encoder_full_s": s("atom_encoder_full"), "atom_decoder_full_s": s("atom_decoder_full"),
        },
        "n_swa_block_calls": len(buckets["swa_block"]),
        "n_rotary_calls": len(buckets["rotary"]),
        "n_modulate_calls": len(buckets["modulate"]),
        "n_sdpa_calls": len(buckets["sdpa"]),
    }
    print(json.dumps(out, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
