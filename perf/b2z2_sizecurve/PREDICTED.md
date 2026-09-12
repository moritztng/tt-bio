# b2z2-atoml1-size-curve — PREDICTION, written before this row opened a device

ARCH: WH. Every number predicted here is Wormhole, whglx `j10glx02`, 8x9 grid, ttnn 0.68.0,
`perf/size512/fixtures/cdk2x2_{512,640,768,896,1024}.yaml` with their fixed a3m, 200 sampling
steps, 3 recycles, 1 sample, seed 0, templates off. The Blackhole leg is OWED, not projected.

## 0. The gate arithmetic, done on paper first

`_atom_branch_memory_config` computes `live = B*K*D*(W + 4*ATOM_DIM)*elem`. At B=1, W=32,
ATOM_DIM=128, D=128, bf16 that is **`live = K * 139264 B`**, K = bucketed_atoms/32, and it is
compared against `0.5 * _l1_bank_bytes() * gx * gy`.

Wormhole here is **8x9 = 72 cores** (read off `perf/whb2/wh_b2_512.json`, same box, same ttnn) at
1,461,760 B per bank, so the budget is **52.62 MB**, against Blackhole's 110 cores / **80.40 MB**.
The gate declines at `K > 377.9`, i.e. **>= 12,096 bucketed atoms, roughly 1530 aa**.

**P0 — `ATOM_L1_STATS` reads `{l1: 1200, dram: 0}` at 512, 640, 768, 896 AND 1024 aa on Wormhole.**
1200 = 200 steps x 6 atom layers. The lever's OWN gate has no knee anywhere in the range this row
measures; at 1024 aa the live set is ~36 MB against a 52.62 MB budget, 69 % of it. A `dram: n > 0`
at any of the five sizes falsifies P0 and is the single most valuable thing this row could find,
because it would mean the lever silently declines at a size a user folds.

## 1. Exponents

Base seconds anchored on the only same-box Wormhole primaries on disk: `perf/whb2/wh_b2_384.json`
31.919 s and `perf/whb2/wh_b2_512.json` 48.744 s, both 200/3, 8x9, ttnn 0.68.0. Fitting
`t = c + b*N^3` through those two gives c = 19.64 s of size-flat cost and b = 2.168e-7, hence a
predicted base ladder of **48.7 / 76.5 / 117.8 / 175.6 / 252.4 s**, whose pure power-law slope over
512 -> 1024 is **2.37**.

* **P1 — base exponent (log-log fit of median fold seconds vs N, 5 points): 2.35 +/- 0.20.**
* **P2 — `TT_BIO_ATOM_L1` exponent: 2.22 +/- 0.20**, i.e. **0.10 to 0.20 BELOW base**, no more.
* **P3 — per-size paired ratios: 512 1.03x, 640 1.05x, 768 1.07x, 896 1.09x, 1024 1.12x.**
  Monotone rising, and the 512 point at or under the session's A/A floor, exactly as the lever
  read alone on the Blackhole cell (1.01869x under a 1.04204x floor).
* **P4 — the curves separate visibly only above 768 aa.** Below it the gap is inside the floor.

## 2. The 768 aa screen, and why I predict 1.89x does not survive

`b2z2-bh-stack-atom`'s single-fold screen read base 85.504 -> L1 45.243 s at 768 aa. **I predict
that pairing kills most of it, and that the anomaly is in the BASE fold, not in the lever.**

Fit each arm's OWN Blackhole ladder as `t = c + b*N^k` through its 256 / 512 / 1024 points, the
three the 768 pair is not in:

| arm | 256 | 512 | 1024 | implied k | implied c | **predicts 768** | screen read 768 |
|---|---|---|---|---|---|---|---|
| base | 17.727 | 26.831 | 107.778 | 3.15 | 16.57 s | **54.2 s** | 85.504 s (**+58 %**) |
| L1 | 14.255 | 24.138 | 74.780 | 2.36 | 11.86 s | **44.5 s** | 45.243 s (**+1.7 %**) |

The L1 arm's 768 point lands on its own curve to 1.7 %. The base arm's is 58 % above its own. One
fold per arm, arms never reversed, loadavg 8.49 -> 9.19 across the pair: **the cheapest explanation
is that the base fold at 768 aa ate a co-tenant, not that the lever is worth 1.89x.**

* **P5 — paired at 768 aa on Wormhole, the ratio is 1.05x-1.12x, NOT 1.89x.** Reconstructed on
  Blackhole from its own fit it would be ~1.22x. Anything at or above 1.5x paired falsifies P5 and
  the lead is real and much bigger than P3.

## 3. The knee

`tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` names three gates that go dark: K2's
`fill_preconditions`, `_TRANSPOSE_L1_HEADROOM` (DRAM at **N >= 560**) and the SDPA q-chunk
overflow at 768/1024. **All three are in the trunk/Pairformer. `TT_BIO_ATOM_L1` is in the diffusion
atom branch.** They are different scopes, so:

* **P6 — the knee is real and sits between 512 and 640 aa**, because `_TRANSPOSE_L1_HEADROOM`
  flips at N >= 560 and my two lowest points bracket 560. The memory's headline "above ~640 aa" is
  where the effect becomes large; the first gate actually flips earlier, and this row's 512/640
  pair is the first measurement that can say so.
* **P7 — `TT_BIO_ATOM_L1` LEAVES THE KNEE ALONE.** It neither moves it nor removes it: both arms
  show the same knee at the same place, and the ratio between them changes smoothly across it.
  A knee that appears in base and NOT in L1 falsifies P7 and would mean the lever is paying back a
  trunk cliff it was never designed for, which is the brief's hypothesis and the more valuable
  outcome.

## 4. Bit-exactness

* **P8 — identical CIF sha256, base vs `TT_BIO_ATOM_L1`, at all five sizes.** The lever sets a
  memory config: it adds no op, removes no op, changes no operand, so it cannot change a value.
  Any mismatch at any size is a NO-GO for the lever outright, not a parity argument.

## 5. The falsifier the brief pre-registered

*If the two arms' exponents agree within their fit error, the 85.504 -> 45.243 s screen was load
noise on a contended box.*

**It is live and I am close to it.** P1 and P2 put the exponents 0.13 apart with +/- 0.20 bands, so
on five points with n>=5 per size this row may well land INSIDE its own fit error. My honest
position: **the ratio rises with N (P3) and is real, but it is a factor of ~1.1, not ~1.9, and the
size dependence may not be resolvable as an exponent separation at n=5.** If the exponents overlap
I will report the falsifier as FIRED for the 1.89x claim and still report the per-size ratio curve,
which is the quantity the product question actually needs.

## 6. What I predict for the product statement

The perf page publishes 512 aa, where this lever is under the floor. If P3 holds, a user folding an
800 aa complex gets **~1.07x-1.09x** from it, not 1.89x — real, worth having, and NOT a separate
headline. If P5 is falsified instead, it is a headline and belongs in front of Moritz immediately.

Blackhole's budget is 1.53x Wormhole's at the same live set, so **the Wormhole curve understates
the Blackhole one** and the BH leg stays OWED.
