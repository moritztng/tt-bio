"""What the triangle-attention SDPA's static circular buffers cost in L1, exactly.

A fold above ~640 padded tokens prints

    TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
    grow to 4844032 B which is beyond max L1 size of 1572864 B

and keeps running: `tenstorrent._tri_att_sdpa_at` catches it, memoises the q_chunk in
`_SDPA_Q_CHUNK_OVER_L1` and takes the next entry down its ladder. So the message is the ladder
finding its own ceiling, not a fold failing -- which is worth stating loudly, because the
throw was read as BoltzGen's Blackhole capacity ceiling once already.

The reported figure is the CB table in `sdpa_generic.build` plus a FIXED per-program reserve:

    reported = sum(CB sizes) + 109056

109056 B is the same on every config measured on a p150a, so the CB budget on this part is
1572864 - 109056 = 1463808 B. Neither term is ours to move: 1572864 B is the physical L1 per
Tensix core (Blackhole and Wormhole B0 are both 1.5 MiB) and the reserve is the program's own
kernel/runtime-arg region. What IS ours is the demand, which the chunk sizes set:

    tiles = Sqc*qbf*DHt + 4*Skc*DHt + 3*Sqc*Skc + 8*Sqc + 3          (bf16, DHt = head_dim/32)

with `qbf = 2` when a core owns more than one q chunk. The `3*Sqc*Skc` term is the mask CB
(double-buffered, 2 tiles) plus the score CB (1), and it is what blows up: a q_chunk spanning
2208 padded tokens against k_chunk 256 asks 1656 of the 714 tiles the budget holds.

`MEASURED` below is every refusal recorded on qb1 card 2 (p150a, 11x10 grid, ttnn 0.68.0),
nine from `probe_cb.py` and two from a live BoltzGen design at 2100 target residues. The model
reproduces all eleven to the byte; `tests/test_sdpa_cb_model.py` is that assertion.
"""
import ttnn

TILE = 32
MAX_L1 = 1572864          # physical L1 per Tensix core, p150a and wormhole_b0 alike
PROGRAM_RESERVE = 109056  # measured, constant across all 11 refusals below
CB_BUDGET = MAX_L1 - PROGRAM_RESERVE

_TILE_BYTES = {"bfloat16": 2048, "bfloat8_b": 1088, "float32": 4096}


def _tb(dtype) -> int:
    return _TILE_BYTES[str(dtype).rsplit(".", 1)[-1]]


def cb_bytes(p, q_dtype="bfloat16", k_dtype="bfloat16", v_dtype="bfloat16",
             mask_dtype="bfloat16", out_dtype="bfloat16", mask_cb_tiles=None) -> int:
    """The CB table of `sdpa_generic.build`, in bytes, from a `sdpa_generic.plan` dict.

    `mask_cb_tiles` is the fused K1/K2 kernel's persistent mask override
    (`k_num_chunks * Sq_chunk_t * Sk_chunk_t`); left None this is the stock op.
    """
    im = 2048   # the intermediate and statistics CBs are bf16 always, factory :651-653
    nmask = p["mask_tiles"] if mask_cb_tiles is None else mask_cb_tiles
    return (p["q_tiles"] * _tb(q_dtype) + p["k_tiles"] * _tb(k_dtype)
            + p["v_tiles"] * _tb(v_dtype) + nmask * _tb(mask_dtype)
            + 3 * im                                    # two scalars + the recip scratch
            + p["qk_tiles"] * im + 2 * p["out_im_tiles"] * im
            + 5 * p["statistics_tiles"] * im
            + p["out0_t"] * _tb(out_dtype))


def reported_bytes(p, **kw) -> int:
    """What tt-metal will print in the refusal, or would have printed had it refused."""
    return cb_bytes(p, **kw) + PROGRAM_RESERVE


def fits(p, **kw) -> bool:
    return reported_bytes(p, **kw) <= MAX_L1


def plan_for(seq, heads, head_dim, q_chunk, k_chunk, grid=(11, 10), split=None):
    """`sdpa_generic.plan` without a device: the only tensor properties it reads are shapes and
    dtypes, so the CB surface can be enumerated on the host."""
    from tt_bio.sdpa_generic import plan

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
# The 512x512 row is also the clash `tests/test_capacity_gate.py:1125` quotes at 3424768 B on a
# 13x10 grid, which is the same arithmetic at a different core count.


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
