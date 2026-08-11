"""C1 at the sizes the sweep skipped. Section 10.6 item 4: C1 ships ON by default, so it has to be
verified it never loses at a size nobody measured, not only at 320/352/384/448/512/576/640/768/1024.

What is timed is the POLICY (`T._tri_att_sdpa`, including its runtime L1-overflow fallback) against
the shipped program config, not a hand-picked q_chunk -- the fallback is part of the lever and a
sweep that bypasses it would be measuring something production does not run.

PREDICTIONS, written before the run:
  I1  the candidate lists are determined by "divisors of the padded length that are multiples of 32
      and wider than the shipped pick", so most intermediate sizes offer exactly ONE wider option:
      480->(480,256), 544->(544,256), 608->(608,256), 672->(672,256), 736->(736,256), 800->(800,256),
      while 288->(288,96,64), 704->(704,352,256), 896->(896,448,256).
  I2  the widest candidate overflows L1 above ~640 (q640/k256 fits at 640 and overflows by 512 B at
      768), so 672/736/800 fall all the way back to the shipped 256 and must measure ~1.000x --
      no win, and critically no loss. 704 falls back to 352 and 896 to 448, which should win.
  I3  every row is >= 1.0x. The 0.797x loss mode (q512 at seq 768, which pads 768->1024 and pays the
      mask padding twice) cannot occur here: every candidate divides the padded length by
      construction. This is the prediction that decides whether C1 is safe on by default.
  I4  every row is torch.equal to the shipped config. q_chunk splits output rows only; k_chunk is
      untouched, so the online-softmax reduction order does not change.
"""
import json, sys, time
from importlib.metadata import version as _v
import torch, ttnn
TTNN_V = _v("ttnn")
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
SIZES = (288, 416, 480, 544, 608, 672, 704, 736, 800, 896)

print("--- I1: candidate lists (no device) ---", flush=True)
chunks = {}
for s in SIZES:
    c = T._tri_att_q_chunks(s, s)
    chunks[s] = list(c)
    print("  seq %4d -> %-24s (shipped %s)" % (s, c, T._sdpa_chunks_shipped(s, s)), flush=True)

dev = ttnn.open_device(device_id=0)
T.COMPUTE_GRID_MAIN = dev.compute_with_storage_grid_size()
grid = T.COMPUTE_GRID_MAIN
print("grid %dx%d = %d cores, ttnn %s" % (grid.x, grid.y, grid.x * grid.y, TTNN_V),
      flush=True)
res = {"grid": [grid.x, grid.y], "ttnn": TTNN_V, "chunks": chunks, "rows": []}


def timed(fn, iters):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


for s in SIZES:
    iters = 5 if s <= 640 else 3
    scale = 32 ** -0.5
    gf = 4 * s * 8 * s * s * 32 / 1e9
    row = {"seq": s, "shipped": list(T._sdpa_chunks_shipped(s, s)), "cands": chunks[s]}
    try:
        q, k, v = (ttnn.from_torch(torch.randn(s, 8, s, 32) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16) for _ in range(3))
        bias = ttnn.from_torch(torch.randn(1, 8, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
    except Exception as e:
        row["alloc_error"] = str(e)[:200]
        res["rows"].append(row)
        print("s=%4d ALLOC FAIL %s" % (s, str(e)[:120]), flush=True)
        continue

    ship = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
        program_config=T._tri_att_sdpa_program_config(s, s))
    # warm the policy once so the L1-overflow throw is paid before the timed region
    ttnn.deallocate(T._tri_att_sdpa(q, k, v, bias, scale))
    pms = timed(lambda: ttnn.deallocate(ship()), iters)
    cms = timed(lambda: ttnn.deallocate(T._tri_att_sdpa(q, k, v, bias, scale)), iters)
    ref = ttnn.to_torch(ship())
    got = ttnn.to_torch(T._tri_att_sdpa(q, k, v, bias, scale))
    over = sorted(x[2] for x in T._SDPA_Q_CHUNK_OVER_L1 if x[0] == s)
    taken = [qc for qc in chunks[s] if qc not in over]
    row.update(shipped_ms=round(pms, 4), cand_ms=round(cms, 4), ratio=round(pms / cms, 4),
               shipped_tflops=round(gf / (pms / 1e3) / 1e3, 2),
               cand_tflops=round(gf / (cms / 1e3) / 1e3, 2),
               q_chunk_taken=taken[0], over_l1=over,
               exact="torch.equal" if torch.equal(got, ref) else "DIFFERS")
    if row["exact"] == "DIFFERS":
        d = got.float() - ref.float()
        row["rmsd_over_std"] = round(float(d.pow(2).mean().sqrt() / ref.float().std()), 8)
    print("s=%4d ship q%-4d %9.4f ms | policy q%-4d %9.4f ms | %.4fx | over-L1 %s | %s" %
          (s, row["shipped"][0], pms, taken[0], cms, row["ratio"], over, row["exact"]), flush=True)
    res["rows"].append(row)
    for t in (q, k, v, bias, ref, got):
        if isinstance(t, torch.Tensor):
            del t
        else:
            ttnn.deallocate(t)
    json.dump(res, open(OUT, "w"), indent=1)

json.dump(res, open(OUT, "w"), indent=1)
ttnn.close_device(dev)
print("wrote", OUT, flush=True)
