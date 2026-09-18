"""Is the pair-row reuse axis expressible in the shipped SDPA path, and what is the bias re-read?

Anthropic amortise one `[BLOCK_M, BLOCK_N]` fp32 bias tile over R consecutive pair rows: R = 2 in
Triton, 3 in `cuda_b`, 4 in `cuda_80`, capped by the register file because the accumulator is
`ROWS x BLOCK_M x D` fp32, 32 KB at R=4 of a 64 KB per-CTA allocation
(`kernels/flash_triattn.py:362`, `kernels/triattn/triattn_native/DESIGN.md`).

Two separate questions, and the shipped path answers them differently:

  EXPRESSED?   Yes. In triangle attention the pair-row axis IS the SDPA batch axis: q/k/v are
               `[512, 4, 512, 32]` and the bias is `[1, 4, 512, 512]`, so `bcast_batch` is true
               (`tt_bio/sdpa_generic.py:120`), and the work factory saturates that axis first,
               `batch_pf = min(B, num_cores)` (`:124`). At 512 aa on an 11x10 grid that is R =
               `batch_per_core` = 5 pair rows per core against one bias, already larger than any R
               their hardware can reach, and bounded by the core count rather than by a register
               file.

  RESIDENT?    No. The stock reader's mask read sits INSIDE the pair-row loop:
               `.../transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp:292` opens
               `for (nb = local_batch_start; nb < local_batch_end; ++nb)` and the
               `cb_reserve_back(cb_mask_in, mask_chunk_tiles)` + `noc_async_read_tile` block is at
               :483-518, three loops in. With `broadcast_provided_mask_batch` the offset
               computation is skipped (`:304-313`) but the READ is not: every pair row re-reads the
               same DRAM tiles. Broadcast saves address arithmetic, not traffic.

So the axis needs a reader-kernel change (hoist the mask read above the `nb` loop, or hold the mask
CB resident across it), not a config. This prices that change from the shipped plan.

CAVEAT, stated because it is the whole reason this is a price and not a measurement: a `ttnn.graph`
capture charges one operand read per PROGRAM, so neither this row's ledger nor the campaign's
profiler byte census can see an intra-kernel re-read. The figure below is read out of the shipped
kernel's loop structure and the shipped plan. It is an upper bound on a quantity nobody has
measured.

Usage:  python3 perf/anthro_zpass/pairrow.py [--seq 512] [--heads 4] [--head-dim 32]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TILE_BYTES = 2048          # 32x32 bf16


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    p.add_argument("--q-chunk", type=int, default=512)
    p.add_argument("--k-chunk", type=int, default=512)
    p.add_argument("--c-z", type=int, default=128)
    p.add_argument("--calls-per-fold", type=int, default=528,
                   help="triangle-attention calls in a 512 aa fold: 2 per PairformerLayer x 264")
    a = p.parse_args()

    import tt_bio.sdpa_generic as S
    pl = S.plan_for_shape(a.seq, a.heads, a.head_dim, a.q_chunk, a.k_chunk, grid=(11, 10))
    z = a.seq * a.seq * a.c_z * 2
    chunk_b = pl["Sq_chunk_t"] * pl["Sk_chunk_t"] * TILE_BYTES
    per_q = pl["q_num_chunks"] * pl["k_num_chunks"]          # mask chunk reads per (nb, nq)

    once = a.heads * a.seq * a.seq * 2
    shipped = pl["B"] * pl["NQH"] * per_q * chunk_b
    hoisted = pl["num_cores"] * pl["NQH"] * per_q * chunk_b

    print(f"shipped plan at {a.seq} aa: B={pl['B']} pair rows, NQH={pl['NQH']}, "
          f"bcast_batch={pl['bcast_batch']}, batch_pf={pl['batch_pf']}, "
          f"R=batch_per_core={pl['batch_per_core']}, q_num_chunks={pl['q_num_chunks']}, "
          f"k_num_chunks={pl['k_num_chunks']}, mask CB {pl['mask_tiles']} tiles")
    print(f"one z unit                      {z:>14,} B")
    print(f"bias read ONCE                  {once:>14,} B   {once / z:7.4f} z units")
    print(f"bias read as shipped            {shipped:>14,} B   {shipped / z:7.2f} z units")
    print(f"bias read hoisted above nb      {hoisted:>14,} B   {hoisted / z:7.2f} z units")
    d = shipped - hoisted
    print(f"deletable by the hoist          {d:>14,} B   {d / z:7.2f} z units per call")
    print(f"over {a.calls_per_fold} calls in a fold      {d * a.calls_per_fold / 1e9:>14.1f} GB")
    print("\nUNMEASURED: a graph capture cannot see an intra-program re-read. Falsifier: profile "
          "the sdpa generic_op\nalone at this shape and check its DRAM read bytes against "
          f"{(shipped + 3 * z) / 1e9:.3f} GB (q+k+v+bias) rather than "
          f"{(once + 3 * z) / 1e9:.3f} GB.")


if __name__ == "__main__":
    main()
