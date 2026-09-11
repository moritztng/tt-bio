# HiFi3 on the Boltz-2 trunk: measured end to end, and priced in Angstrom

Card 1 of qb2 (p300c, one Blackhole processor, 11x10 grid, AICLK 1350 MHz), ttnn 0.68.0,
commit `8724bc1c`, under benchlock. Protocol as published: `perf/size512/fixtures/cdk2x2_512.yaml`
plus its fixed 35-row a3m, 3 recycles, 200 sampling steps, 1 sample, seed 0, templates off, timed
at `predict_one` (featurise + fold + CIF write). One cold fold discarded, then one warm-up fold per
arm, then 5 interleaved reps. Everything in one process: `perf/b2x-hifi3/hifi_ab.py`.

## 1. The fold

| arm | fold median | fold reps | trunk median | trunk range | diffusion median | trunk vs HiFi4 |
|---|---|---|---|---|---|---|
| HiFi4 (incumbent) | **23.769 s** | 23.507-24.490 | **12.8755 s** | 12.8394-12.9702 | 7.1644 s | 1.000x |
| HiFi3 | **23.558 s** | 23.420-23.767 | **12.7365 s** | 12.6905-12.7516 | 7.1728 s | **1.0109x** |
| HiFi2 (record only) | 23.519 s | n=1 | 12.6533 s | n=1 | 7.1934 s | 1.0176x |

End-to-end HiFi3 is 1.009x. The incumbent's own A/A floor in this session is **4.136 %** (0.983 s,
one 24.490 s rep), so **the fold-level delta is not separable from harness noise**.

The trunk phase is separable, and it is the honest number. The two arms' five trunk walls do not
overlap: every HiFi4 rep is above every HiFi3 rep (12.8394 > 12.7516), the incumbent's trunk-phase
A/A floor is 1.016 % against a 1.09 % effect, and the gap is **0.139 s**.

The diffusion phase is the negative control. The flip is scoped to the trunk's two
compute-kernel-config objects, and the sampler reads 7.1644 / 7.1728 / 7.1934 s across the three
arms: unmoved, slightly up with the noise. An arm that had leaked into the sampler would show there.

## 2. CENSUS-VERDICT: REFUTED, by 7.5x

The 1.071x came from a per-op cost multiplied by a TFLOP census over 831.2 of the Pairformer's
872.95 executed TFLOP. On a 23.769 s fold that projects a 1.59 s saving. The fold gives 0.211 s,
and the trunk phase, which is the only place the change can act, gives 0.139 s. The campaign's own
63-73 % realization factor would have deflated the projection to 1.045x; that is still 5x the
measurement.

**The mechanism is arithmetic, and there is just very little of it exposed.** Removing the fourth
math pass buys 0.139 s; removing the third as well buys 0.083 s more, not another 0.139 s. Two
passes for 1.6x one pass's saving is a sub-linear curve, which is what partially-hidden FPU time
looks like. Extrapolated, all four passes are worth about 0.44 s of exposed time in a 12.876 s
trunk — **3.4 %**, against the 1.61 s (12.5 % of the 85.96 TFLOP/s HiFi4 roof) the census implies.
The trunk is dispatch-bound (`b2x-baseline-attrib`: 21.846 s of main-thread CPU in a 26.037 s
fold), so most of the FPU time it does spend is behind host op issue.

## 3. The first fold in a new fidelity costs 4.45 s, and it is an A/B trap

| arm | first fold ever at that fidelity | its steady-state trunk | penalty |
|---|---|---|---|
| HiFi4 | 23.487 s / trunk 12.8904 s | 12.8755 s | none (warm from the cold fold) |
| HiFi3 | **27.884 s** / trunk 17.1876 s | 12.7365 s | **+4.45 s** |
| HiFi2 | **35.606 s** / trunk 24.8283 s | 12.6533 s | **+12.17 s** |

Kernel binaries are JIT-compiled per math fidelity. A blocked or single-shot A/B, or one that warms
the process but not each arm, reads HiFi3 as **0.84x** and HiFi2 as **0.66x**: a 19 % and a 34 %
regression that do not exist. This is also a real cost for anything that would switch fidelity at
runtime in a service.

## 4. Accuracy, in Angstrom

`perf/b2x-baseline-attrib/control_rmsd.py` (all-atom and CA Kabsch, atoms matched on chain, seq id
and atom name) against the incumbent arm from the same session. Bar: <= 0.35 A pass, 0.35-0.60 A
hold for a decision, > 0.60 A reject.

| arm | cdk2x2_298 all-atom | CA | verdict | 512 aa per domain (1-290 / 301-512) | 512 whole |
|---|---|---|---|---|---|
| HiFi3 | **0.355 A** | 0.155 A | **hold** | **1.019 A / 1.223 A** (CA 0.802 / 0.978) | 12.789 A |
| HiFi2 | 0.254 A | 0.112 A | pass | 1.078 A / 1.249 A (CA 0.834 / 0.993) | 12.852 A |

Three things this table says.

**HiFi3 lands on the bar, not inside it.** 0.355 A is in the hold band. For scale, all of main's
accepted drift on this control between 2026-08-26 and today is 0.218 A.

**The control cannot rank precision levers at this magnitude.** HiFi2 has 2.59x HiFi3's relative
RMS on a matmul and moves the structure *less* (0.254 vs 0.355 A). The ordering is inverted, so what
these numbers measure is trajectory divergence in the 200-step sampler, not precision. A 298 aa
control can tell "non-bit-exact" from "bit-exact" and nothing finer.

**A fixed-size control would have missed the size dependence**, as the campaign's own bucketing
lesson predicts. At 512 aa the per-domain deviation is 1.0-1.2 A, 3x the bar. The whole-structure
12.8 A is the chimera's unconstrained hinge and is not a fold change; per domain is the fair read.
For context, main's own accepted drift on the same fixture reads 0.865 / 0.980 A per domain, so
1.019 / 1.223 A is the same order as a month of accepted drift on a metastable fixture. The 512
number is therefore not clean evidence against HiFi3 on its own. The decision rests on the 298
control, where the fixture is stable and the answer is "hold".

Instrument checks, both passed: two HiFi4 folds of 512 aa superpose at **0.0 A** and are
byte-identical, and the incumbent arm reproduces current main bit for bit (512 aa
`4f3995a69be5d610`, 298 aa `71653ff7...`). So the incumbent arm is the incumbent, and any RMSD
above zero is the arm.

## 5. What landed

`TT_BIO_TRUNK_MATH_FIDELITY` was priced on Boltz-2's Pairformer and wired only into Protenix.
Boltz-2's trunk wrappers took `TorchWrapper`'s hardcoded HiFi4, so the env var did nothing to the
model the 1.071x was computed for. `TorchWrapper` now takes a per-instance `trunk` flag and
Boltz-2's `msa_module` and 64-block `pairformer_module` pass it. Per instance because
`PairformerModule` also builds the 8-block confidence stack and the 2-block template stack, which
keep the default.

Verified both ways, by bytes, one fold each:

* env unset: trunk reads HiFi4, all six config objects HiFi4, CIF `4f3995a69be5d610` = current
  main. Production is unchanged (`knob_default_512_qb2c1.json`).
* `TT_BIO_TRUNK_MATH_FIDELITY=hifi3`: trunk reads HiFi3, everything else HiFi4, CIF
  `bf5455fb4977a142` = the digest the in-process flip produced. Same math through a different door
  (`knob_hifi3_512_qb2c1.json`).

**The default is not flipped and NO-GO is the recommendation:** 0.6 % of the fold, not separable
end to end, for a control in the hold band and 1.0-1.2 A per domain at the production size.

## Reproduce

    # the A/B (one card, one process, ~8 min)
    benchlock.sh <owner> -- env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      PYTHONPATH=$PWD python3 perf/b2x-hifi3/hifi_ab.py \
      --out perf/b2x-hifi3/hifi_ab_512_qb2c1.json --cifdir perf/b2x-hifi3/cif --reps 5

    # the knob, both arms
    TT_VISIBLE_DEVICES=1 python3 perf/b2x-hifi3/verify_knob.py \
      --out /tmp/knob.json --expect-trunk-fidelity HiFi4 --expect-sha16 4f3995a69be5d610

    # Angstrom
    python3 perf/b2x-baseline-attrib/control_rmsd.py --ref <hifi4.cif> --arm <hifi3.cif> --out ...
    python3 perf/b2x-hifi3/domain_split.py --ref <hifi4_512.cif> --arm <hifi3_512.cif> --out ...

qb2's p300c wedges inside `ttnn.open_device` on its fourth open since the board's last reset, and
this task used three: the A/B and the two knob verifications. A fourth needs a board reset, which on
this box resets cards 0 and 1 together and would kill anything on card 0.
