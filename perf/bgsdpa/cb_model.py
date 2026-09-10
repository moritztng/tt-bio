"""The measured refusals the SDPA L1 model in `tt_bio.sdpa_generic` has to reproduce, and a
printer for the (q_chunk, k_chunk) surface at a given padded length.

A fold above ~640 padded tokens prints

    TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
    grow to 4844032 B which is beyond max L1 size of 1572864 B

and keeps running: `tenstorrent._tri_att_sdpa_at` catches it, memoises the q_chunk in
`_SDPA_Q_CHUNK_OVER_L1` and takes the next entry down its ladder. The message is the ladder
finding its own ceiling, not a fold failing, which is worth saying loudly because it was once
read as BoltzGen's Blackhole capacity ceiling.

The arithmetic lives next to the CB table it prices, in `sdpa_generic.cb_bytes`. What is here is
the evidence: every refusal measured on qb1 card 2 (p150a, 11x10 grid, ttnn 0.68.0), eight from
`probe_cb.py` and two from a live BoltzGen design at 2100 target residues.
`tests/test_sdpa_cb_model.py` is the assertion that the model still reproduces them.

    python3 perf/bgsdpa/cb_model.py 2208 2592 1536
"""
import ttnn

from tt_bio.sdpa_generic import (L1_PER_CORE, PROGRAM_RESERVE, cb_bytes,   # noqa: F401
                                 cb_fits_l1, plan)

TILE = 32
MAX_L1 = L1_PER_CORE
CB_BUDGET = L1_PER_CORE - PROGRAM_RESERVE


def reported_bytes(p, **kw) -> int:
    """What tt-metal prints in the refusal, or would print had it refused."""
    return cb_bytes(p, **kw) + PROGRAM_RESERVE


def fits(p, **kw) -> bool:
    return cb_fits_l1(p, **kw)


def plan_for(seq, heads, head_dim, q_chunk, k_chunk, grid=(11, 10), split=None):
    """`sdpa_generic.plan` without a device: the only tensor properties it reads are shapes and
    dtypes, so the whole CB surface can be enumerated on the host."""
    class _T:
        def __init__(self, shape):
            self.padded_shape = self.shape = shape
            self.dtype = ttnn.bfloat16

    q = _T([seq, heads, seq, head_dim])
    b = _T([1, heads, seq, seq])
    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
    return plan(q, q, q, b, q, q_chunk, k_chunk, grid, ckc, 1.0, split)


# (seq, heads, head_dim, q_chunk, k_chunk, reported_B, source)
MEASURED = [
    (32, 4, 32, 2208, 256, 4844032, "probe_cb"),
    (32, 4, 32, 2592, 256, 5655040, "probe_cb"),
    (32, 4, 32, 256, 736, 1581568, "probe_cb"),
    (32, 4, 32, 736, 256, 1735168, "probe_cb"),
    (32, 4, 32, 256, 2208, 4219392, "probe_cb"),
    (32, 4, 32, 512, 512, 2114048, "probe_cb"),
    (32, 4, 32, 1024, 256, 2343424, "probe_cb"),
    (32, 4, 32, 256, 1024, 2097664, "probe_cb"),
    # A live BoltzGen design, 2100 target residues -> 2180 tokens -> padded 2208. The second
    # entry is the one the tiny-tensor probe cannot reach: q_chunk 736 leaves a core three q
    # chunks at seq 2208, so `q_buffer_factor` is 2 and the q CB doubles. At seq 32 the same
    # q_chunk is one chunk and the factor is 1, which is 47104 B (23 tiles) lighter.
    (2208, 4, 32, 2208, 256, 4844032, "log_r2100"),
    (2208, 4, 32, 736, 256, 1782272, "log_r2100"),
]
# The 1536/q1536/k256 config prices at 3424768 B, which is the clash
# `tests/test_capacity_gate.py:1125` already quotes from a 13x10 grid: same arithmetic, and the
# core count does not enter it.


def surface(seq, heads=4, head_dim=32, grid=(11, 10)):
    """Every (q_chunk, k_chunk) pair of 32-aligned divisors of the padded length, with the L1 cost
    on both routes. The divisor restriction is not cosmetic: a chunk that does not divide sets
    `use_padded_mask`, which the fused kernel declines outright."""
    padded = -(-seq // TILE) * TILE
    divs = sorted({padded // n for n in range(1, padded // TILE + 1)
                   if padded % n == 0 and (padded // n) % TILE == 0})
    rows = []
    for kc in divs:
        for qc in divs:
            p = plan_for(padded, heads, head_dim, qc, kc, grid)
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            rows.append({"q_chunk": qc, "k_chunk": kc,
                         "stock_b": reported_bytes(p),
                         "fused_b": reported_bytes(p, mask_cb_tiles=pers),
                         "q_per_core": p["q_per_core"]})
    return padded, divs, rows


if __name__ == "__main__":
    import sys
    for seq in [int(a) for a in sys.argv[1:]] or [2208]:
        padded, divs, rows = surface(seq)
        print(f"seq {seq} -> padded {padded}, 32-aligned divisors {divs}, "
              f"CB budget {CB_BUDGET} B")
        for r in rows:
            tag = "".join(("S" if r["stock_b"] <= MAX_L1 else "-",
                           "F" if r["fused_b"] <= MAX_L1 else "-"))
            print(f"  q={r['q_chunk']:6d} k={r['k_chunk']:6d}  stock={r['stock_b']:9d}  "
                  f"fused={r['fused_b']:10d}  q/core={r['q_per_core']:3d}  {tag}")
