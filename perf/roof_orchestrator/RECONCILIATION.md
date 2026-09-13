# The roofline budget's three numbers, reconciled by provenance

The ROOF campaign opened with a stated roofline of ~10.15 s and a 1.71x gap, built from a
bandwidth floor of 3.405 TB / 445 GB/s = 7.65 s and a compute floor of 873 TFLOP / 86.0 TFLOP/s =
10.15 s, with the brief's own warning that the two inputs came from different instruments and could
not be reconciled against a separately reported 3-18 % of the compute roof.

This pass did not re-measure anything. It traced each number to the artifact that produced it. One
of the three is a category error, and once it is corrected all three agree.

## The finding

**873 TFLOP is not the fold. It is the Pairformer alone, at a different recycle count, on a
different counting basis.** It appears as `872.95 TFLOP` in two places, both Pairformer-scoped:

- `state/moonshot-4x-512aa-ledger.md:39` — *"4x gives the Pairformer 27.869 s to execute 872.95
  TFLOP"*. That ledger's fold is **79.172 s at 10 recycles** and 200 sampling steps.
- `state/b2x-hifi3-endtoend.md:4` — *"a TFLOP census (831.2 of the Pairformer's 872.95 TFLOP at
  512 aa)"*.

**The fold's own counted FLOPs are 206.706 TFLOP**, at the configuration the published cell runs:
512 tokens, 7168 atoms, 35-sequence MSA, **3 recycles (4 trunk passes)**, 200 sampling steps.
`perf/bioir_roofline/flops_bytes_512.json`, `fold_flops = 206705568776192`, counted per phase under
torch's `FlopCounterMode` — counted, not a hand formula.

873 / 206.706 = **4.22x**. Two independent reasons for the gap, and they compound:

1. **Scope.** 872.95 is the Pairformer stack; 206.706 is every phase of the fold.
2. **Configuration.** The moonshot ledger folds at 10 recycles, the cell at 3. At 10 recycles the
   Pairformer runs 11 x 64 = 704 calls; at 3 it runs 4 x 64 = 256.

There is residual basis drift on top of those. At 704 calls of the census's own 502.796 GFLOP
per Pairformer call, 10 recycles gives **353.97 TFLOP**, still 2.47x below 872.95 — so the moonshot
figure also counts more FLOP per call than `FlopCounterMode` does, most likely tile-padded rather
than logical. That third term is not needed for the verdict and is not claimed here.

## Once 873 is replaced by 206.706, all three numbers agree

| quantity | value | source |
|---|---|---|
| fold FLOPs, 512 aa, 3 recycles, 200 steps | **206.706 TFLOP** | `flops_bytes_512.json`, torch `FlopCounterMode` |
| dense bf16 HiFi4 roof, N=8192 | **85.96 TFLOP/s** | `roofs_p300c_qb2_card2.json`, measured |
| read+write stream roof, 8192^2 | **429.9 GB/s** | same file, measured |
| fold traffic, buffer-address dedup, at the 23.841 s fold | **3.405 TB** | `state/b2x-baseline-attrib.md:212` |
| fold compulsory (resident) bytes | **220.063 GB** | `flops_bytes_512.json`, `fold_resident_bytes` |

- **Compute floor: 206.706 TFLOP / 85.96 TFLOP/s = 2.405 s.** Not 10.15 s.
- **Bandwidth floor as the fold is currently written: 3.405 TB / 429.9 GB/s = 7.920 s.**
- **Bandwidth floor if nothing were re-materialised: 220.063 GB / 429.9 GB/s = 0.512 s.**

`perf/b2x-baseline-attrib/roof_table.py:150` already carried the corrected arithmetic
(`206.705568776192e12 / COMPUTE_ROOF`), and `state/b2x-baseline-attrib.md:187` states it in words:
*"the fold's 206.71 TFLOP is 2.405 s at the compute roof."* The campaign brief did not use it.

**And the 3-18 % phase figures survive.** 206.706 TFLOP spread over the 17.34 s cell is 11.92
TFLOP/s, **13.9 % of the 85.96 TFLOP/s roof** — inside the band, not the 59 % that a 10.15 s
compute floor against a 17.34 s fold would require. The band was never in conflict with anything.
It was in conflict with one number, and that number was the wrong one.

## What this does to the campaign

**The compute roof does not bind.** At the cell's configuration, arithmetic is 2.405 s of a 17.34 s
fold: 13.9 %. Phase B of the campaign as briefed — arithmetic reduction, described as *"the only
line that moves the roofline itself"* — is sized against a roofline that is not the binding one.
Halving the fold's FLOPs removes at most 1.2 s and only if that arithmetic is exposed, which
`b2x-hifi3-endtoend` measured it is not: removing a whole math pass bought 0.139 s of a 12.876 s
trunk, 7.5x less than its own TFLOP census projected.

**The binding constraint is traffic the implementation chooses to move.** 3.405 TB against a
compulsory 220.063 GB is **15.5x**. That gap is not hardware and it is not arithmetic. It is
intermediates materialised to DRAM because two programs are two programs.

**Three floors, not one.** Stating a single roofline for this fold is what produced the error in
the first place. The honest form is a ladder:

| floor | s | what it assumes |
|---|---|---|
| arithmetic, dense HiFi4 roof | 2.405 | every FLOP at the 8192-cube rate. Unreachable: the fold's dominant matmuls are K=128 and measure 28-38 TFLOP/s |
| arithmetic, shape-honest | higher, unmeasured | each matmul at the rate its own shape achieves. This is the number worth having |
| traffic, as currently decomposed | 7.920 | today's program boundaries, at 100 % of the stream roof |
| traffic, compulsory only | 0.512 | perfect fusion. Priced and refuted: the Pairformer megakernel lost 2.9 % |

## What is still open, and why it needs a device

Every number above is a Blackhole p300c number, and the two largest are **stale at the tip**:

- **3.405 TB was counted at the 23.841 s fold**, commit `13604183`. The fold is now 17.34 s after
  the b2z2 and K10 landings, several of which (`TT_BIO_TRIMUL_MASK_L1`, `TT_BIO_RESIDUAL_L1`,
  the work-removal levers) exist specifically to move fewer bytes. The current traffic is unknown
  and is the single number the work queue must be ranked on.
- **The shape-honest compute floor has never been computed.** The 2.405 s uses a roof no op in this
  fold reaches.
- **429.9 GB/s vs the brief's 445 GB/s.** Both are measured, on different parts and by different
  rigs: 429.9 is `roofs_p300c_qb2_card2.json`'s 8192^2 read+write on qb2 card 2; 444.9 is
  `k10-instrument`'s, validated to 98.9 % by a starved `ttnn.add` control. They are not
  interchangeable and the table must state which it used.

`roof-budget` owns all three. Its job is now verification and a re-take at the tip, not discovery.
