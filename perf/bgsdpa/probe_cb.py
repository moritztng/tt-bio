"""What the stock SDPA's static circular buffers cost in L1, measured rather than modelled.

The refusal a fold hits is `Statically allocated circular buffers on core range ... grow to N B
which is beyond max L1 size of 1572864 B`, and N is a function of (q_chunk, k_chunk, head_dim,
dtype) alone -- not of the sequence length, because every CB is sized in CHUNK tiles. So the
whole (q_chunk, k_chunk) surface can be probed on a 32x32 tensor in milliseconds instead of
allocating the 5 GB the real shape needs: `q_chunk_size` larger than Sq is legal (the factory
pads Sq up to a multiple of it).

Prints one row per config: the bytes tt-metal reports, the bytes the model in `cb_model.py`
predicts, and their difference.
"""
import os, re, sys, json
import torch, ttnn

sys.path.insert(0, os.environ["WT"])
from tt_bio import tenstorrent as T          # noqa: E402
from tt_bio.sdpa_generic import plan         # noqa: E402

GROW = re.compile(r"grow to (\d+) B which is beyond max L1 size of (\d+) B")


def cb_bytes(p, q, k, v, mask, out):
    """Sum of the CB table in `sdpa_generic.build`, in bytes."""
    tb = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}
    im = 2048
    return (p["q_tiles"] * tb[q.dtype] + p["k_tiles"] * tb[k.dtype]
            + p["v_tiles"] * tb[v.dtype] + p["mask_tiles"] * tb[mask.dtype]
            + 3 * im                                   # two scalars + recip scratch
            + p["qk_tiles"] * im + 2 * p["out_im_tiles"] * im
            + 5 * p["statistics_tiles"] * im
            + p["out0_t"] * tb[out.dtype])


def main():
    dev = T.get_device()
    grid = T.COMPUTE_GRID_MAIN
    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
    rows = []
    S, H, D = 32, 4, 32
    def mk(shape):
        return ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    q, k, v = mk([S, H, S, D]), mk([S, H, S, D]), mk([S, H, S, D])
    bias = mk([1, H, S, S])
    for qc, kc in [(256, 256), (2208, 256), (2592, 256), (256, 736), (736, 256),
                   (608, 256), (640, 256), (96, 736), (224, 736), (256, 2208),
                   (128, 128), (512, 512), (1024, 256), (256, 1024)]:
        try:
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=1.0,
                program_config=T._sdpa_program_config(qc, kc))
            ttnn.synchronize_device(dev)
            got, maxl1 = None, None
            ttnn.deallocate(o)
        except Exception as exc:                                        # noqa: BLE001
            m = GROW.search(str(exc))
            if not m:
                raise
            got, maxl1 = int(m.group(1)), int(m.group(2))
        p = plan(q, k, v, bias, q, qc, kc, grid, ckc, 1.0)
        pred = cb_bytes(p, q, k, v, bias, q)
        rows.append({"q_chunk": qc, "k_chunk": kc, "reported": got, "max_l1": maxl1,
                     "cb_model": pred, "delta": None if got is None else got - pred,
                     "q_buffer_factor": p["q_buffer_factor"]})
        r = rows[-1]
        print(f"q={qc:5d} k={kc:5d}  reported={str(got):>9}  model={pred:9d}  "
              f"delta={str(r['delta']):>8}  qbf={r['q_buffer_factor']}", flush=True)
    print(json.dumps(rows))
    ttnn.close_device(dev)


main()
