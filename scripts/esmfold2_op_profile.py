"""Op-level timing breakdown of the ESMFold2 fold on qb2.

Wraps the ESMFold2-specific diffusion sub-components (atom encoder/decoder SWA
transformers, token DiT, conditioning) and the folding trunk with
device-synchronized timers, aggregates over a real fold (warmup + timed), and
prints wall-clock seconds + share-of-fold per surface. Read-only w.r.t. the
model. Used by the ESMFold2 fusion scout to locate the real bottleneck and
isolate ESMFold2-specific novel compute from closed shared primitives.
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
    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex

    model = load_ttnn_esmfold2()
    seq = SEQS[args.protein]
    device = model.structure_head.m.tt_device

    buckets = {k: [] for k in [
        "trunk", "diff_cond_pair", "diff_cond_single",
        "atom_encoder", "token_dit", "atom_decoder", "structure", "confidence",
    ]}

    def sync():
        ttnn.synchronize_device(device)

    def wrap_inst(owner, name, key):
        orig = getattr(owner, name)

        def w(*a, **kw):
            sync()
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            sync()
            buckets[key].append(time.perf_counter() - t0)
            return r

        setattr(owner, name, w)

    def wrap_cls(cls, name, key):
        orig = getattr(cls, name)

        def w(self, *a, **kw):
            sync()
            t0 = time.perf_counter()
            r = orig(self, *a, **kw)
            sync()
            buckets[key].append(time.perf_counter() - t0)
            return r

        setattr(cls, name, w)

    # Whole-stage timers (instance methods called by name from the runtime).
    wrap_inst(model.structure_head, "sample", "structure")
    wrap_inst(model.confidence_head, "forward", "confidence")
    wrap_inst(model.folding_trunk, "forward", "trunk")

    # Diffusion inner components — wrap on the CLASS so __call__ is honored.
    wrap_cls(E.DiffusionConditioningModel, "cond_pair", "diff_cond_pair")
    wrap_cls(E.DiffusionConditioningModel, "cond_single", "diff_cond_single")
    wrap_cls(E.AtomEncoder, "__call__", "atom_encoder")
    wrap_cls(E.AtomDecoder, "__call__", "atom_decoder")
    wrap_cls(E.DiffusionTransformerModel, "__call__", "token_dit")

    def one():
        for v in buckets.values():
            v.clear()
        sync()
        t0 = time.perf_counter()
        fold_complex(model, [("A", seq)], num_loops=args.loops,
                     num_sampling_steps=args.steps, num_diffusion_samples=1, seed=0)
        sync()
        return time.perf_counter() - t0

    print(f"[warmup fold {args.protein} L={len(seq)}] ...", flush=True)
    one()
    print(f"[timed fold] ...", flush=True)
    total = one()

    def s(k):
        return sum(buckets[k])

    structure = s("structure")
    confidence = s("confidence")
    trunk = s("trunk")
    enc = s("atom_encoder")
    dec = s("atom_decoder")
    dit = s("token_dit")
    cpair = s("diff_cond_pair")
    csingle = s("diff_cond_single")
    denoise_inner = enc + dec + dit + csingle
    out = {
        "model": "esmfold2", "tokens": len(seq), "loops": args.loops,
        "sampling_steps": args.steps, "timed_total_s": total,
        "trunk_s": trunk, "trunk_share": trunk / total,
        "structure_s": structure, "structure_share": structure / total,
        "confidence_s": confidence, "confidence_share": confidence / total,
        "diff_cond_pair_s": cpair, "diff_cond_single_s": csingle,
        "atom_encoder_s": enc, "atom_encoder_share": enc / total,
        "atom_decoder_s": dec, "atom_decoder_share": dec / total,
        "token_dit_s": dit, "token_dit_share": dit / total,
        "swa_atom_total_s": enc + dec, "swa_atom_total_share": (enc + dec) / total,
        "denoise_inner_s": denoise_inner, "denoise_inner_share": denoise_inner / total,
        "n_atom_encoder_calls": len(buckets["atom_encoder"]),
        "n_atom_decoder_calls": len(buckets["atom_decoder"]),
        "n_token_dit_calls": len(buckets["token_dit"]),
    }
    print(json.dumps(out, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
