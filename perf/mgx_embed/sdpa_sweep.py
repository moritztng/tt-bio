"""Which ttnn SDPA program configs return the wrong attention?

    python3 perf/mgx_embed/sdpa_sweep.py --out sweep.jsonl [--grid 8,9]
    python3 perf/mgx_embed/sdpa_sweep.py --out sweep.jsonl --lengths 1024,1536,2048 \
        --chunks 128,256,512 --arms none,zero,bias
    python3 perf/mgx_embed/sdpa_sweep.py --out sweep.jsonl --configs 1536,4,1536,32,256,256

esmc-6b on Wormhole returned PCC 0.33 against fp32 for every 126-residue sequence (128 tokens,
so no padding and no mask), and a different wrong vector each time. The op alone reproduces it:
unmasked SDPA at q_chunk = k_chunk = 128 on an 8x9 grid. This sweeps q/k/v [B, H, L, d] with
random inputs over chunk pairs that divide L, runs every config twice and scores both runs
against torch's fp32 SDPA. One row per (config, arm). The arms:

  none  attn_mask=None
  zero  an all-zero [1, 1, L, L] additive mask, which changes nothing mathematically (what
        tenstorrent.fused_sdpa passes in place of None)
  bias  a real [1, H, L, L] bias, N(0, 2) in bf16, shared over the batch the way a pair model's
        triangle attention shares it; the reference takes the same bf16 bias, times the scale,
        because ttnn scales the mask with the scores (rows before `bias_ref` did not)

--configs takes explicit B,H,L,d,q_chunk,k_chunk tuples separated by `;`, for the shapes a census
names; otherwise the product of --batch/--heads/--dims/--lengths/--chunks is swept. --grid
defaults to the device's own compute grid.
"""
import argparse
import itertools
import json

import torch


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def _reference(q, k, v, bias, scale):
    # ttnn's SDPA computes softmax((q k^T + mask) * scale): it scales the mask with the scores,
    # where torch adds it after. So the reference takes bias * scale.
    # One batch row at a time: a triangle-attention shape has B = L, and [B, H, L, L] fp32 scores
    # at 1536 tokens would be 58 GB.
    bias = None if bias is None else bias * scale
    return torch.cat([torch.nn.functional.scaled_dot_product_attention(
        q[b:b + 1], k[b:b + 1], v[b:b + 1], attn_mask=bias, scale=scale)
        for b in range(q.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="", help="x,y; default: the device's compute grid")
    ap.add_argument("--lengths", default="64,96,128,160,192,256,384,512")
    ap.add_argument("--heads", default="1,16,40")
    ap.add_argument("--dims", default="32,64,128")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--chunks", default="32,64,128,256")
    ap.add_argument("--arms", default="none,zero")
    ap.add_argument("--configs", default="", help="B,H,L,d,q_chunk,k_chunk;...")
    a = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    gx, gy = map(int, a.grid.split(",")) if a.grid else (g.x, g.y)
    t = lambda x: ttnn.from_torch(x.bfloat16(), device=dev, layout=ttnn.TILE_LAYOUT)
    ints = lambda s: [int(v) for v in s.split(",")]
    arms = a.arms.split(",")
    if a.configs:
        todo = [tuple(ints(c)) for c in a.configs.split(";")]
    else:
        todo = [(B, H, L, d, qc, kc) for d, H, L, B in itertools.product(
                    ints(a.dims), ints(a.heads), ints(a.lengths), ints(a.batch))
                for qc, kc in itertools.product(ints(a.chunks), ints(a.chunks))
                if not (L % qc or L % kc)]
    torch.manual_seed(0)
    shape = None
    with open(a.out, "a") as out:
        for B, H, L, d, qc, kc in todo:
            if shape != (B, H, L, d):  # inputs and references shared by every chunk pair
                shape = (B, H, L, d)
                q, k, v = (torch.randn(B, H, L, d) for _ in range(3))
                bias = (2 * torch.randn(1, H, L, L)).bfloat16().float()
                tq, tk, tv = t(q), t(k), t(v)
                masks = {"none": None, "zero": t(torch.zeros(1, 1, L, L)), "bias": t(bias)}
                refs = {}
            for arm in arms:
                if arm not in refs:
                    refs[arm] = _reference(q, k, v, bias if arm == "bias" else None, d ** -0.5)
                row = dict(grid=[gx, gy], arch=str(dev.arch()), B=B, d=d, H=H, L=L, q_chunk=qc,
                           k_chunk=kc, mask=arm != "none", arm=arm, bias_ref="scaled")
                cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                             exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)
                try:
                    runs = [ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
                        tq, tk, tv, attn_mask=masks[arm], is_causal=False,
                        scale=d ** -0.5, program_config=cfg)).float() for _ in range(2)]
                except Exception as e:  # a refusal is a row too
                    row["error"] = str(e).splitlines()[0][:200]
                else:
                    row["pcc"] = [_pcc(o, refs[arm]) for o in runs]
                    row["repeat_equal"] = bool(torch.equal(runs[0], runs[1]))
                out.write(json.dumps(row) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
