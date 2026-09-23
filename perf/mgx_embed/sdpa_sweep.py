"""Which ttnn SDPA program configs return the wrong attention?

    python3 perf/mgx_embed/sdpa_sweep.py --out sweep.jsonl [--grid 8,9]

esmc-6b on Wormhole returned PCC 0.33 against fp32 for every 126-residue sequence (128 tokens,
so no padding and no mask), and a different wrong vector each time. The op alone reproduces it:
unmasked SDPA at q_chunk = k_chunk = 128 on an 8x9 grid. This sweeps q/k/v [1, H, L, d] with
random inputs over chunk pairs that divide L, with and without an all-zero additive mask (which
changes nothing mathematically), runs every config twice and scores both runs against torch's
fp32 SDPA. One row per config. --grid defaults to the device's own compute grid.
"""
import argparse
import itertools
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="", help="x,y; default: the device's compute grid")
    ap.add_argument("--lengths", default="64,96,128,160,192,256,384,512")
    ap.add_argument("--heads", default="1,16,40")
    ap.add_argument("--dims", default="32,64,128")
    a = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    gx, gy = map(int, a.grid.split(",")) if a.grid else (g.x, g.y)
    t = lambda x: ttnn.from_torch(x.bfloat16(), device=dev, layout=ttnn.TILE_LAYOUT)
    ints = lambda s: [int(v) for v in s.split(",")]
    torch.manual_seed(0)
    with open(a.out, "a") as out:
        for d, H, L in itertools.product(ints(a.dims), ints(a.heads), ints(a.lengths)):
            q, k, v = (torch.randn(1, H, L, d) for _ in range(3))
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            tq, tk, tv = t(q), t(k), t(v)
            zero = t(torch.zeros(1, 1, L, L))
            for qc, kc, mask in itertools.product((32, 64, 128, 256), (32, 64, 128, 256), (False, True)):
                if L % qc or L % kc:
                    continue
                row = dict(grid=[gx, gy], arch=str(dev.arch()), d=d, H=H, L=L, q_chunk=qc,
                           k_chunk=kc, mask=mask)
                cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                             exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)
                try:
                    runs = [ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
                        tq, tk, tv, attn_mask=zero if mask else None, is_causal=False,
                        scale=d ** -0.5, program_config=cfg)).float() for _ in range(2)]
                except Exception as e:  # a refusal is a row too
                    row["error"] = str(e).splitlines()[0][:200]
                else:
                    row["pcc"] = [torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
                                  for o in runs]
                    row["repeat_equal"] = bool(torch.equal(runs[0], runs[1]))
                out.write(json.dumps(row) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
