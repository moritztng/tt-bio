"""A/B for the fused-rotary attention fusion (docs/esmc-attention-kernel-scout.md).

Toggles the shipped ``esmc._rope`` between the fused ttnn rotary kernel (used for
tile-aligned L) and the manual rotate-half fallback, and reports:
  * end-to-end warm-forward speedup (fused vs manual), per model size and length;
  * per-residue embedding PCC vs the reference esm ESMC for both paths (the
    release-gate accuracy metric) when --ref is passed (needs ESM_ROOT).

Usage:
    TT_VISIBLE_DEVICES=1 python3 scripts/esmc_rotary_fusion_ab.py \
        --model esmc-300m --tokens 96,416 [--ref]
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch, ttnn

from tt_bio import esmc as tt_esmc
from tt_bio import tenstorrent

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

def seq_for_tokens(n_tokens: int) -> str:
    n = n_tokens - 2                       # tokenize adds <cls>/<eos>
    return (UBQ * (n // len(UBQ) + 1))[:n]

def pcc(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])

def manual_rope(q, k, cos, sin):
    return tt_esmc.apply_rotary(q, cos, sin), tt_esmc.apply_rotary(k, cos, sin)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.MODELS))
    ap.add_argument("--tokens", default="96,416")
    ap.add_argument("--ref", action="store_true", help="also report PCC vs esm reference")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = tenstorrent.get_device()
    model = tt_esmc.load_esmc(args.model)
    shipped_rope = tt_esmc._rope     # fused when L%32==0

    def emb(tokens):
        out = model(tokens)
        return (out[1] if isinstance(out, tuple) else out).float().numpy()

    def median_ms(tokens, n=7):
        v = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter(); model(tokens); ttnn.synchronize_device(dev)
            v.append(time.perf_counter() - t0)
        return sorted(v)[len(v) // 2] * 1e3

    ref_res = {}
    if args.ref:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        from esmc_reference import ESMCReference
        cfg, repo_id, wpath = tt_esmc.CONFIGS[args.model]
        from huggingface_hub import hf_hub_download
        sd = torch.load(hf_hub_download(repo_id, wpath), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        rmod = ESMCReference(**cfg).eval(); rmod.load_state_dict(sd, strict=False)

    for tt in [int(x) for x in args.tokens.split(",")]:
        tokens = tt_esmc.tokenize(seq_for_tokens(tt))
        assert tokens.shape[-1] == tt, (tokens.shape[-1], tt)

        tt_esmc._rope = manual_rope
        for _ in range(2): model(tokens)
        e_manual = emb(tokens); t_manual = median_ms(tokens)

        tt_esmc._rope = shipped_rope
        for _ in range(2): model(tokens)
        e_fused = emb(tokens); t_fused = median_ms(tokens)

        line = (f"{args.model} tokens={tt}: manual={t_manual:.2f} ms  fused={t_fused:.2f} ms  "
                f"speedup={t_manual/t_fused:.3f}x  PCC(manual,fused)={pcc(e_manual, e_fused):.6f}")
        if args.ref:
            _, remb = rmod(tokens); rr = remb[0][1:-1].numpy()
            line += (f"  | ref-PCC manual={pcc(rr, e_manual[0][1:-1]):.6f} "
                     f"fused={pcc(rr, e_fused[0][1:-1]):.6f}")
        print(line, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
