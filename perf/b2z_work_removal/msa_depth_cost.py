"""Is the Boltz-2 MSA track's cost linear in alignment depth, and what does the 1024 bucket cost?

Times the four shipped `MSALayer` blocks on real weights at a ladder of padded depths, with the
true row count held at the fixture's 35 so the mask path is the shipped one. One process, one
device open, depths interleaved round-robin so host drift and a contended box cancel out of the
ratios. Reports device+dispatch milliseconds per MSALayer call.

    python msa_depth_cost.py --out out.json --tokens 512 --depths 32,64,128,256,512,1024
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))   # patch_boltz2_cfg: one copy, not a second one

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--n-msa", type=int, default=35)
    ap.add_argument("--depths", default="32,64,128,256,512,1024")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]

    import tt_bio.tenstorrent as T
    import ttnn
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    patch_boltz2_cfg()
    target = ROOT / "perf/size512/fixtures" / f"cdk2x2_{a.tokens}.yaml"
    a3m = target.with_suffix(".a3m")
    _one_fold, meta, state = B.build_fold("boltz2", ROOT / f".msa_b2zwr_{a.tokens}", target, a3m)
    model = state.model

    msa_wrapper = model.msa_module
    msa = msa_wrapper.module
    n_blocks = len(msa.blocks)
    tok = a.tokens
    seq_pad = T.pad_amount(tok, T.PAIRFORMER_PAD_MULTIPLE) if T.bucket_enabled() else 0
    padded_seq = tok + seq_pad
    c_z = int(model.z_norm.weight.shape[-1])
    c_s = int(model.s_init.in_features)   # s_inputs width, what MSA.s_proj consumes

    # Per-depth device inputs. `z` is depth-independent AND mutated in place by the blocks
    # (`ttnn.add_(z, ...)`), so it is re-uploaded outside the timed region before every call
    # rather than shared: a shared `z` would accumulate 45 blocks of updates across the run and
    # a per-depth `z` would put five 64 MB pair tensors on the part at once, which is an L1 OOM.
    z_host = torch.randn(1, padded_seq, padded_seq, c_z, dtype=torch.float32) * 0.02
    emb_host = torch.randn(1, padded_seq, c_s, dtype=torch.float32) * 0.02
    mask_1d = z_host.new_ones(1, padded_seq)
    if seq_pad:
        mask_1d[:, tok:] = 0.0
    mask_tt = msa_wrapper._from_torch(mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1))
    attn_tt = msa_wrapper._from_torch((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9)
    emb_tt = msa_wrapper._from_torch(emb_host)

    legs = {}
    for d in depths:
        assert d >= a.n_msa, f"depth {d} < true rows {a.n_msa}"
        assert d % 32 == 0, f"depth {d} is not a multiple of the 32 tile"
        m = torch.zeros(1, d, padded_seq, 36, dtype=torch.float32)
        m[:, : a.n_msa, :tok, :] = torch.randn(1, a.n_msa, tok, 36) * 0.1
        rowmask = torch.zeros(d, 1, 1, dtype=torch.float32)
        rowmask[: a.n_msa] = 1.0
        legs[d] = dict(m=msa_wrapper._from_torch(m),
                       rowmask=msa_wrapper._from_torch(rowmask))

    dev = T.get_device()

    def one(d):
        L = legs[d]
        z_tt = msa_wrapper._from_torch(z_host)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        msa(z_tt, L["m"], emb_tt, mask_tt, attn_tt, L["rowmask"], a.n_msa)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        ttnn.deallocate(z_tt)
        return dt

    for d in depths:                                    # warm every program cache first
        for _ in range(a.warm):
            one(d)

    samples = {d: [] for d in depths}
    for r in range(a.reps):                             # round-robin: contention cancels
        for d in depths:
            samples[d].append(one(d))

    rows = []
    for d in depths:
        s = sorted(samples[d])
        rows.append({
            "padded_depth": d, "true_rows": a.n_msa, "tokens": tok, "padded_tokens": padded_seq,
            "n_blocks": n_blocks,
            "module_ms": round(st.median(s) * 1e3, 4),
            "per_layer_ms": round(st.median(s) * 1e3 / n_blocks, 4),
            "min_ms": round(s[0] * 1e3, 4), "max_ms": round(s[-1] * 1e3, 4),
            "reps": a.reps,
        })
        print(json.dumps(rows[-1]), flush=True)

    # a + b*depth by least squares on the per-layer medians
    xs = [r["padded_depth"] for r in rows]
    ys = [r["per_layer_ms"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    aa = my - b * mx
    import importlib.metadata as im
    import socket
    out = {"host": socket.gethostname(), "arch": T.arch_name(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "ttnn": im.version("ttnn"),
           "fit": {"intercept_ms": round(aa, 4), "slope_ms_per_row": round(b, 6),
                   "depth_free_share_at_1024": round(aa / (aa + b * 1024), 4)},
           "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["fit"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
