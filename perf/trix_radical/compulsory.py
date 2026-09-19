#!/usr/bin/env python3
"""trix-radical: the compulsory cost of triangle multiplication, derived from the definition.

CPU only. No device is opened and nothing here is timed. Every measured input is a named
constant at the top with the card, the clock and the doc it came from; everything else is
arithmetic over those. Run with no arguments to print the whole derivation.

The operation, from tt_bio/reference.py:450-465 (TriangleMultiplicationOutgoing.forward):

    zn        = LayerNorm(z)                       # [N,N,D]
    u         = p_in(zn) * sigmoid(g_in(zn))       # D -> 2D, so [N,N,2D]
    u         = u * mask
    a, b      = chunk(u, 2, dim=-1)                # each [N,N,D]
    x[i,j,d]  = sum_k a[i,k,d] * b[j,k,d]          # D independent N x N @ N x N
    out       = p_out(LayerNorm(x)) * sigmoid(g_out(zn))

p_in is D -> 2D and the chunk halves it, so hidden == D and there is one shape parameter.

The question this file answers is not "what do our kernels cost" but "what must ANY correct
implementation move and compute, given this part's L1". Those are different questions and the
campaign has been answering the first while calling it the second.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# MEASURED INPUTS. Each one names its source. Nothing else in this file is measured.
# ---------------------------------------------------------------------------------------------

# trix-floor, qb2 p300c card 0, AICLK FORCED and verified at 1350 MHz, shipped kernel config
# (HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True).
ROOF_TFLOPS = 123.65e12          # square 4096^3, pipelined
ROOF_DRAM_COMBINED = 410.3e9     # starved 2R+1W add, 128 MiB
ROOF_DRAM_READ = 343.6e9
ROOF_DRAM_WRITE = 245.2e9
AICLK_HZ = 1350e6

# trix-floor, same card and clock: the production call sites' own rates, serial median.
RATE_INPROJ = 54.97e12           # _in_proj_matmul at G=8
RATE_CONTRACT = 42.20e12         # ttnn.matmul under _triangle_mul_program_config(16)
RATE_OUTPROJ = 43.34e12          # _pair_proj_linear

# trix-floor, same card and clock: the module as it runs today, standalone, per call.
MODULE_MS_UNMASKED = 11.519
MODULE_MS_MASKED = 12.330

# L1. Raw usable total is the campaign's A3 figure; the per-bank size is from the
# tt-bio-l1-residency-row-blocking-pincer measurement. What a PROGRAM can hold is smaller and is
# measured in trimul-fused-kernel-final S4 (perf/trimul_f2/l1_cap.py, qb2 p300c card 1).
CORES = 110
L1_BANK_B = 1_461_760
L1_TOTAL_B = CORES * L1_BANK_B                      # 160.79 MB raw usable
# Both caps are quoted in MiB by their source (its chunks are 32 MiB), so they are converted
# once here and every print below is decimal MB. 127.9 MiB = 134.11 MB, and Z is 128.0 MiB
# exactly -- the two are within 0.1 %, which is the whole story of section 3.
L1_SINGLE_ALLOC_CAP_B = 48.0 * 2**20                # largest single tensor, measured
L1_CORESIDENT_CAP_B = 127.9 * 2**20                 # co-residency in 32 MiB chunks, measured

# The transpose, measured. perfwar-trimul-kernel: the channel move exchanges a batch axis with an
# intra-tile axis, forcing 64 B NOC transactions, predicted 66.0 GB/s and measured 66.1 on the
# DRAM path. trimul-fused-kernel-final S1 measured the same move L1->L1 at C=32: 0.154 ms for
# 16.777 MB. trix-layout measured the shipped gated forward move at 512 aa at 1.534 ms (p150a).
MOVE_GBPS_DRAM_PATH = 66.1e9
MOVE_L1_MS_PER_CALL = 0.154e-3   # C=32, one role, N=512
MOVE_L1_BYTES_PER_CALL = 512 * 512 * 32 * 2
SHIPPED_FWD_MOVE_MS = 1.534      # reblock_permute_gated, 512 aa, qb1 p150a -- DIFFERENT PART
SHIPPED_BACK_MOVE_MS = 1.017     # reblock_permute_back, E2's measured op

# Tile arithmetic on Tensix. A matmul_tiles instruction issues a 32x32x32 tile product; an
# eltwise (or broadcast-eltwise) instruction issues 32x32 lanes. That ratio is the reason the
# contraction cannot be expressed with the channel axis inside a tile.
TILE = 32

# ---------------------------------------------------------------------------------------------
# THE SHAPE
# ---------------------------------------------------------------------------------------------

N = 512          # tokens, the cdk2x2_512 cell the campaign's ground truth is measured on
D = 256          # protenix-v2 token_z; hidden == D because p_in is D -> 2D and chunk halves it
BYTES_BF16 = 2
BFP8_RATIO = 1088 / 2048         # tt-metal bfp8_b: 1024 B mantissa + 64 B exponents per tile

Z = N * N * D * BYTES_BF16       # the pair tensor, and every role tensor, in bf16
W_BYTES = (2 * D * 2 * D + 2 * D * D) * BYTES_BF16   # p_in, g_in (D->2D), p_out, g_out (D->D)


def gflop() -> dict[str, float]:
    """Compulsory arithmetic, multiply-accumulate counted as 2 FLOP."""
    return {
        "p_in  (N^2 x D @ D x 2D)": 2 * N * N * D * (2 * D),
        "g_in  (N^2 x D @ D x 2D)": 2 * N * N * D * (2 * D),
        "contraction (D indep. N x N @ N x N)": 2 * N**3 * D,
        "p_out (N^2 x D @ D x D)": 2 * N * N * D * D,
        "g_out (N^2 x D @ D x D)": 2 * N * N * D * D,
    }


# ---------------------------------------------------------------------------------------------
# THE SCHEDULES. A schedule is fixed by ONE choice: which axis of the intermediate is blocked.
# ---------------------------------------------------------------------------------------------

def schedule_traffic() -> list[tuple[str, float, float, str]]:
    """(name, DRAM bytes, resident bytes required, note).

    F0  no blocking at all. z read once, out written once. Requires a, b and the gate all live
        from the single z read, because every out[i,j] needs all of a and all of b.
    C64 channel-blocked at C=64, which is what ships. Each channel pass must re-read all of z,
        because the projection mixes all D channels, and x must be materialised because the tail
        reduces over all D of it.
    R   row-blocked. b is built once at FULL D and stays resident; then row blocks of i stream
        through projection -> contraction -> tail with zn[I] still live, so the tail folds into
        the block and never sees DRAM.
    """
    out = []
    out.append(("F0  infinite L1", 2 * Z, 3 * Z,
                "a + b + gate all live from one z read"))
    for C in (32, 64, 128):
        passes = D // C
        # z re-read per channel pass, x written and read back, z re-read for the tail gate, out written
        traffic = passes * Z + Z + Z + Z + Z
        resident = 3 * Z * C / D
        out.append((f"C{C:<3} channel-blocked", traffic, resident,
                    f"{passes} z re-reads + x round trip + z for the gate + out"))
    out.append(("R   row-blocked, b resident", 3 * Z, Z,
                "z read twice (b pass, row pass), out written once"))
    return out


def schedule_r_working_set(bi: int, b_bytes: float) -> float:
    """Per-core L1 at row-block height bi: resident b plus zn[I], a[I] and x[I]."""
    per_block = 3 * (bi / N) * Z
    return (b_bytes + per_block) / CORES


def stage_time(flop: float, byts: float, rate: float, bw: float) -> float:
    """A stage streams, so its cost is the larger of its arithmetic and its traffic."""
    return max(flop / rate, byts / bw)


def main() -> None:
    P = print
    P("=" * 94)
    P(f"TRIANGLE MULTIPLICATION, N={N} tokens, D={D} channels, bf16.")
    P(f"Z = one pair tensor = N^2 * D * 2 B = {Z/1e6:.3f} MB.  Weights = {W_BYTES/1e3:.1f} kB "
      f"= {100*W_BYTES/Z:.2f} % of Z.")
    P("=" * 94)

    P("\n--- 1. COMPULSORY ARITHMETIC (from the definition; no implementation assumed) ---")
    g = gflop()
    for k, v in g.items():
        P(f"  {k:<40s} {v/1e9:9.3f} GFLOP")
    total = sum(g.values())
    P(f"  {'TOTAL':<40s} {total/1e9:9.3f} GFLOP   = 12*N^2*D^2 + 2*N^3*D")
    P(f"  at the measured {ROOF_TFLOPS/1e12:.2f} TFLOP/s roof: {1e3*total/ROOF_TFLOPS:.3f} ms")
    P(f"  the contraction is {100*g['contraction (D indep. N x N @ N x N)']/total:.1f} % of it; "
      f"projections overtake it only below N = 6D = {6*D}")

    P("\n--- 2. COMPULSORY DATA MOVEMENT is a function of what L1 can hold ---")
    P(f"  usable L1 raw            {L1_TOTAL_B/1e6:8.2f} MB  ({CORES} banks x {L1_BANK_B} B)")
    P(f"  largest single alloc     {L1_SINGLE_ALLOC_CAP_B/1e6:8.2f} MB  MEASURED (bank fragments "
      f"long before it fills)")
    P(f"  co-resident ceiling      {L1_CORESIDENT_CAP_B/1e6:8.2f} MB  MEASURED, in 32 MB chunks")
    P(f"  one pair tensor Z        {Z/1e6:8.2f} MB  bf16")
    P(f"  one pair tensor Z        {Z*BFP8_RATIO/1e6:8.2f} MB  bfp8_b")
    P("")
    P(f"  {'schedule':<28s} {'DRAM':>10s} {'xZ':>6s} {'resident':>10s} {'fits?':>7s}  note")
    for name, traffic, resident, note in schedule_traffic():
        fits = "yes" if resident <= L1_CORESIDENT_CAP_B else "NO"
        P(f"  {name:<28s} {traffic/1e6:9.1f}M {traffic/Z:5.0f}x {resident/1e6:9.1f}M {fits:>7s}  {note}")
    P("")
    P(f"  So the minimum is 3Z = {3*Z/1e6:.1f} MB, and it needs ONE role tensor resident at full D.")
    P(f"  2Z is unreachable: it needs 3Z = {3*Z/1e6:.1f} MB resident against a "
      f"{L1_CORESIDENT_CAP_B/1e6:.1f} MB measured ceiling.")
    P(f"  C64 is what ships and it moves {8*Z/(3*Z):.2f}x the minimum.")

    P("\n--- 3. DOES SCHEDULE R FIT? this is the design's single load-bearing question ---")
    for label, bb in (("b in bf16 ", Z), ("b in bfp8_b", Z * BFP8_RATIO)):
        head = "fits" if bb <= L1_CORESIDENT_CAP_B else "REFUSED by the measured ceiling"
        P(f"  {label}: resident {bb/1e6:7.2f} MB = {100*bb/L1_CORESIDENT_CAP_B:5.1f} % of the "
          f"co-resident ceiling -> {head}")
        for bi in (16, 32, 64, 128):
            per_core = schedule_r_working_set(bi, bb)
            P(f"      BI={bi:<4d} per-core {per_core/1e3:8.1f} kB of {L1_BANK_B/1e3:.1f} kB "
              f"({100*per_core/L1_BANK_B:5.1f} %), spare {(L1_BANK_B-per_core)/1e3:7.1f} kB for CBs")

    P("\n--- 4. THE FLOOR UNDER EACH SCHEDULE ---")
    P("  C64 adds its two stages (the tail reduces over all D of x, so it cannot join a channel")
    P("  pass); R has one pipelined stage because the tail folds into the row block.")
    # C64, as trix-floor priced it: stage 1 = projections + contraction, stage 2 = tail.
    s1_flop = sum(v for k, v in g.items() if "out" not in k)
    s2_flop = total - s1_flop
    c64_roof = (stage_time(s1_flop, 5 * Z, ROOF_TFLOPS, ROOF_DRAM_COMBINED)
                + stage_time(s2_flop, 3 * Z, ROOF_TFLOPS, ROOF_DRAM_COMBINED))
    r_roof = max(total / ROOF_TFLOPS, 3 * Z / ROOF_DRAM_COMBINED)
    P(f"  C64 at the machine roof : {1e3*c64_roof:.3f} ms   (byte-bound: 8Z is "
      f"{1e3*8*Z/ROOF_DRAM_COMBINED:.3f} ms of traffic against {1e3*total/ROOF_TFLOPS:.3f} ms of math)")
    P(f"  R   at the machine roof : {1e3*r_roof:.3f} ms   (compute-bound: 3Z is "
      f"{1e3*3*Z/ROOF_DRAM_COMBINED:.3f} ms of traffic, "
      f"{total/ROOF_TFLOPS/(3*Z/ROOF_DRAM_COMBINED):.2f}x under the math)")
    P(f"  -> the schedule decides WHICH SIDE BINDS. R is the true compulsory floor: "
      f"{1e3*r_roof:.3f} ms.")

    ach = (sum(v for k, v in g.items() if "_in" in k) / RATE_INPROJ
           + g["contraction (D indep. N x N @ N x N)"] / RATE_CONTRACT
           + sum(v for k, v in g.items() if "_out" in k) / RATE_OUTPROJ)
    P(f"\n  the same arithmetic at the production call sites' own measured rates "
      f"({RATE_INPROJ/1e12:.2f} / {RATE_CONTRACT/1e12:.2f} / {RATE_OUTPROJ/1e12:.2f} TFLOP/s): "
      f"{1e3*ach:.3f} ms")
    P("  at those rates BOTH schedules are compute-bound, so 8Z -> 3Z is worth 0.000 ms today.")
    P(f"  the byte saving starts to pay only once the arithmetic drops under 8Z's "
      f"{1e3*8*Z/ROOF_DRAM_COMBINED:.3f} ms, i.e. above a class rate of "
      f"{1e-12*total/(8*Z/ROOF_DRAM_COMBINED):.1f} TFLOP/s "
      f"({total/(8*Z/ROOF_DRAM_COMBINED)/RATE_CONTRACT:.2f}x today's contraction rate).")

    P("\n--- 5. THE COMPULSORY TRANSPOSE, and why it is sub-tile ---")
    P("  The projections are pair-local and contract over d; the contraction is channel-local and")
    P("  contracts over k. Each output x[i,j,:] needs a[i,:,:] and b[j,:,:], so a core holding")
    P("  channels must hear from every core holding pairs: a full all-to-all, 3Z of it (a, b, x).")
    P(f"  payload = 3Z = {3*Z/1e6:.1f} MB.")
    P("")
    P("  Why it cannot be done tile-granular ACROSS cores: a matmul tile op contracts the inner")
    P(f"  tile axis, so for x[i,j,d] = sum_k a[i,k,d] b[j,k,d] the Hadamard axis d must sit OUTSIDE")
    P("  the tile. Keeping d inside the tile turns the contraction into a broadcast-multiply-")
    P(f"  accumulate at {TILE*TILE} lanes per instruction instead of {TILE**3} MACs, i.e. {TILE}x less")
    P(f"  arithmetic per issue: the contraction would go {1e3*g['contraction (D indep. N x N @ N x N)']/ROOF_TFLOPS:.3f}"
      f" -> {1e3*TILE*g['contraction (D indep. N x N @ N x N)']/ROOF_TFLOPS:.1f} ms. Dead.")
    P("  So d is inside the tile for the projections and outside it for the contraction, and the")
    P(f"  granule of the exchange is one {TILE}-element column = {TILE*BYTES_BF16} B. That is exactly the")
    P(f"  {MOVE_GBPS_DRAM_PATH/1e9:.1f} GB/s the channel move measures, and it is COMPULSORY at the axis level.")
    P("")
    l1_rate = MOVE_L1_BYTES_PER_CALL / MOVE_L1_MS_PER_CALL
    P(f"  measured move rates: DRAM path {MOVE_GBPS_DRAM_PATH/1e9:6.1f} GB/s    "
      f"L1->L1 at C=32 {l1_rate/1e9:6.1f} GB/s")
    P(f"  3Z at the L1->L1 rate                : {1e3*3*Z/l1_rate:6.3f} ms")
    P(f"  shipped today (fwd {SHIPPED_FWD_MOVE_MS} + back {SHIPPED_BACK_MOVE_MS} ms, DIFFERENT PARTS): "
      f"{SHIPPED_FWD_MOVE_MS+SHIPPED_BACK_MOVE_MS:6.3f} ms = "
      f"{3*Z/((SHIPPED_FWD_MOVE_MS+SHIPPED_BACK_MOVE_MS)*1e-3)/1e9:.0f} GB/s")
    P("")
    P("  What is NOT compulsory is paying 64 B on the NOC. Block the exchange into 32x32x32 cubes")
    P(f"  ({TILE**3*BYTES_BF16/1e3:.1f} kB = {TILE} whole tiles): move the {TILE} tiles core-to-core at "
      f"{TILE*TILE*BYTES_BF16} B each and")
    P("  rotate the cube inside the receiving core's own L1, where a 64 B granule is a local SRAM")
    P("  read and not a NOC packet. The inter-core traffic is then tile-granular and the sub-tile")
    P("  work never leaves a core.")

    P("\n--- 6. WHERE THE 11.519 ms GOES (module measured; the split is derived) ---")
    ach_ms = 1e3 * ach
    kern = ach_ms - 1e3 * total / ROOF_TFLOPS
    move = SHIPPED_FWD_MOVE_MS + SHIPPED_BACK_MOVE_MS
    rest = MODULE_MS_UNMASKED - ach_ms - move
    rows = [
        ("compulsory arithmetic at the measured roof", 1e3 * total / ROOF_TFLOPS,
         "derived from the definition / measured roof"),
        ("kernel-rate deficit (class rates vs the roof)", kern,
         "class rates measured, subtraction derived"),
        ("the compulsory transpose, at 64 B granularity", move,
         "measured, but on mixed parts -- treat as a band"),
        ("everything else (DRAM round trips, dispatch, barriers)", rest,
         "BY SUBTRACTION -- trix-scaffold-attribute owns the direct split"),
    ]
    for name, ms, note in rows:
        P(f"  {name:<52s} {ms:6.3f} ms  {100*ms/MODULE_MS_UNMASKED:5.1f} %  {note}")
    P(f"  {'MODULE, standalone, unmasked':<52s} {MODULE_MS_UNMASKED:6.3f} ms  100.0 %")
    P("")
    P(f"  Schedule R attacks rows 3 and 4 ({move+rest:.3f} ms, "
      f"{100*(move+rest)/MODULE_MS_UNMASKED:.1f} %) and does nothing about row 2.")
    P(f"  Its ceiling on the module is therefore {MODULE_MS_UNMASKED:.3f} -> {ach_ms:.3f} ms = "
      f"{MODULE_MS_UNMASKED/ach_ms:.2f}x, not the {MODULE_MS_UNMASKED/(1e3*r_roof):.2f}x the "
      f"compulsory floor would suggest.")
    P(f"  Against the compulsory floor the module is {MODULE_MS_UNMASKED/(1e3*r_roof):.2f}x "
      f"(unmasked) / {MODULE_MS_MASKED/(1e3*r_roof):.2f}x (masked, the production call).")

    P("\n--- 7. TWO ROBUSTNESS CHECKS ON THE CONCLUSION ---")
    zb = Z * BFP8_RATIO
    P(f"  (a) 3Z does not depend on precision. 2Z needs a, b and the gate all resident; even at")
    P(f"      bfp8_b that is 3 x {zb/1e6:.2f} = {3*zb/1e6:.1f} MB against the {L1_CORESIDENT_CAP_B/1e6:.1f} MB "
      f"measured ceiling.")
    P(f"      Holding two role tensors at bfp8_b is {2*zb/1e6:.1f} MB, still over. 3Z stands at every dtype.")
    P("")
    P("  (b) the kernel-rate deficit is NOT an arithmetic-throughput problem, so raising the math")
    P("      fidelity does not collect it. HiFi2 halves the FPU passes per bf16 tile product, and")
    P("      `TT_BIO_TRUNK_MATH_FIDELITY` already exposes it; the knowledgebase records the trunk")
    P("      measurement as 6 % and a broken fp32_dest_acc, and bf4 as 0 % plus accuracy loss.")
    P(f"      A matmul at {100*RATE_CONTRACT/ROOF_TFLOPS:.0f} % of roof is not waiting on FPU passes, which is")
    P("      consistent with row 2 above and with fixterm's per-output-tile decomposition.")
    P("=" * 94)


if __name__ == "__main__":
    main()
