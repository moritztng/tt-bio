# trix-radical — the compulsory cost of triangle multiplication

CPU only. No device was opened. `compulsory.py` derives everything below from the definition of the
operation plus a table of measured constants, each of which names the card, the clock and the doc it
came from. `compulsory.txt` is its output. Full reasoning in `~/.coworker/state/trix-radical.md`.

At N = 512 tokens and D = 256 channels, bf16, `Z = N²·D·2 B = 134.218 MB`:

- **Arithmetic is 274.877 GFLOP** (`12·N²·D² + 2·N³·D`) and irreducible.
- **DRAM traffic is 3Z = 402.7 MB**, not the 8Z the shipped schedule moves. z is read twice and the
  answer written once. 2Z is unreachable at any dtype: reading z once forces a, b and the output
  gate to be simultaneously live, 402.7 MB against a 134.1 MB measured co-residency ceiling.
- So **the compulsory floor is 2.223 ms/call and it is compute-bound**, with 2.27x of byte headroom.
  The campaign's 2.648 ms is byte-bound because it prices the shipped 8Z schedule, not because the
  operation is byte-bound.
- **The 64 B transpose granule is compulsory at the axis level.** A `matmul_tiles` contracts the
  inner tile axis, so the Hadamard axis d must sit outside the tile for the contraction and inside
  it for the projections. Keeping d inside the tile turns the contraction into a broadcast-multiply
  at 1024 lanes per issue instead of 32768 MACs and costs 0.556 -> 17.8 ms. That derives
  `perfwar-trimul-kernel`'s measured 66.1 GB/s rather than blaming it on a kernel.
- **Paying 64 B on the NOC is not compulsory.** Block the exchange into 32x32x32 cubes, move 32
  whole tiles core-to-core, rotate the cube in the receiving core's L1. Worth 1.3-2.5 ms and it is a
  standalone op swap.

Where the measured 11.519 ms goes: 2.223 ms compulsory arithmetic, 3.491 ms kernel-rate deficit,
2.551 ms transpose at 64 B, 3.254 ms everything else (by subtraction). A row-blocked fused program
attacks the last two, 50.4 % of the module, and its ceiling is 11.519 -> 5.714 = 2.02x.

It is not recommended for building. It needs one role tensor L1-resident at full D, which is
134.22 MB at bf16 against a 134.11 MB measured ceiling — already refused, twice, by measurements
this campaign has taken. bfp8_b brings it to 71.30 MB and is the first configuration not obviously
refused, which is not the same as measured.
