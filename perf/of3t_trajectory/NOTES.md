# of3t-trajectory — what was measured and how to re-run it

## The question

Every gradient figure in this campaign is a distance from float64. Nobody had measured the
distance from what OpenFold3's own training step computes. D72 put our deviation beside
upstream's bf16 deviation, but both are distances from the same float64 reference, and two
distances from one reference do not order each other. This row removes the shared subtrahend.

## The answer

`AGREEMENT_WITH_UPSTREAMS_OWN_STEP.json`, over the 547 tensors the device arm covers, 51.1358 %
of the model's squared gradient norm:

| pair | mass-weighted rel_l2 | median | norm ratio | cos |
|---|---|---|---|---|
| our device gradient vs upstream bf16 | **7.426742e+00** | 1.753310e-01 | 7.7814 | 0.4109 |
| our device gradient vs float64 | 7.569163e+00 | 1.658849e-01 | 7.9182 | 0.4045 |
| upstream bf16 vs float64 | 5.852018e-02 | 6.915873e-02 | 1.0176 | 0.9985 |
| upstream bf16 vs upstream fp32 | 5.852284e-02 | 6.915573e-02 | 1.0176 | 0.9985 |
| upstream fp32 vs float64 | 3.117006e-05 | 2.439626e-05 | 1.0000 | 1.0000 |
| zero model vs upstream bf16 (A16) | 1.000000e+00 | 1.0 | 0.0 | n/a |
| our run with permuted cotangents vs upstream bf16 | 3.706050e+00 | 1.579021e+00 | 3.3505 | -0.2252 |
| upstream permuted draws vs upstream bf16 | 4.018294e-01 | 4.134079e-01 | 0.8955 | 0.9159 |
| the float64-softmax bound vs upstream bf16 | **1.172914e-01** | 8.752432e-02 | 0.9787 | 0.9932 |
| the float64-softmax bound vs float64 | 1.066810e-01 | 5.297022e-02 | 0.9959 | 0.9943 |

7.4267 is 126.9x upstream's own distance from the ideal. Moving the reference from float64 to
their actual training step moved our headline by 1.9 %: their bf16 sits 0.0585 from float64,
we sit 7.5 from both.

`D72_ROW_BY_ROW_WITHOUT_THE_SHARED_SUBTRAHEND.json` re-reads D72's table with the direct column
added. `layer_norm_a` survives at 3.4968e-02 against a 3.6970e-02 floor. `layer_norm_s` does
not: D72 read it AT OR BETTER at 2.3809e-02 against float64, and directly against their
gradient it is 5.2797e-02, larger than either distance and 1.70x its own floor.

## The softmax bound, at model scope

`devgrad_traj.sh sm64` installs of3t-adaln's float64 softmax rule on the tape verb, so it
reaches every softmax the tape sees: 1,440 calls intercepted over the 48 structures, all 24 DiT
blocks and both atom transformers. It is a bound, not a lever. Everything the softmax could
contribute is removed, and what is left is what the softmax cannot explain.

It takes the scope headline from 7.426742e+00 to **1.172914e-01**, a factor of **63.3**, with
the norm ratio going 7.7814 to 0.9787 and the cosine 0.4109 to 0.9932. The threshold a perfect
fix would read is 5.750945e-02, so the bound lands at **2.04x** it: the softmax is the dominant
mechanism at model scope and it does not account for all of the gap.

| section | mass | bound | shipped | own bf16 floor | bound / floor |
|---|---|---|---|---|---|
| diffusion_transformer | 43.6221 % | 1.1250e-01 | 8.0250e+00 | 5.7809e-02 | 1.95x |
| atom_attn_enc | 4.7348 % | 9.4304e-02 | 2.1925e-01 | 7.0855e-02 | 1.33x |
| atom_attn_dec | 1.2843 % | 2.8235e-01 | 2.8139e-01 | 5.0000e-02 | 5.65x |
| layer_norm_s | 0.9835 % | 4.8164e-02 | 5.2797e-02 | 3.1012e-02 | 1.55x |
| layer_norm_a | 0.2666 % | 3.7430e-02 | 3.4968e-02 | 3.6970e-02 | 1.01x |
| linear_s | 0.2445 % | 8.3418e-02 | 2.4562e-01 | 6.5442e-02 | 1.27x |

The DiT is where the whole factor comes from: 8.0250 to 0.11250, 71.3x, and 138.8x its own
floor down to 1.95x. `atom_attn_dec` does not move at all, 2.8139e-01 to 2.8235e-01, so the
softmax explains none of its 5.65x. That is the next mechanism, named by the residual.

## What the direct column does not cover

48.8642 % of the model. Five D72 rows have no direct measurement and hold 47.6582 % of it:
`diffusion_conditioning` at 36.9462 %, `pairformer_stack`, `aux_heads`, `msa_module` and
`input_embedder`. Three of those carry D72's "AT OR BETTER" readings, and conditioning is both
the largest section of the model and D72's best row. They are still shared-subtrahend readings.
Closing them needs each row's own arm to dump its tensors the way `--dump-grads` does here.

The last 1.2060 % is 1.1286 % of the diffusion arm the device run never reached and 0.0774 % in
six sections too small for D72's table.

## Re-running

The device arms, about 3 minutes each on one Blackhole card:

    perf/of3t_trajectory/devgrad_traj.sh            # the real run
    perf/of3t_trajectory/devgrad_traj.sh permcot    # the break control
    perf/of3t_trajectory/devgrad_traj.sh sm64       # the float64-softmax bound

Both write a `.pt` of the gradient tensors under `/home/ttuser/of3t_trajectory/` beside the
per-tensor JSON here. Then the comparison, a few minutes and no card:

    OMP_NUM_THREADS=2 nice -n 15 /home/ttuser/tt-bio-dev/env/bin/python \
      perf/of3t_trajectory/agreement.py \
      --device /home/ttuser/of3t_trajectory/device_grads_043all.pt \
      --device-permuted /home/ttuser/of3t_trajectory/device_grads_043all_permcot.pt \
      --device-bound /home/ttuser/of3t_trajectory/device_grads_043all_sm64.pt \
      --f64 /home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt \
      --bf16 /home/ttuser/of3t_refprec/pinned_p175/arm4_bf16_autocast/grads_f64.pt \
      --f32 /home/ttuser/of3t_refprec/pinned_p175/arm2_f32_upstream/grads_f64.pt \
      --upstream-permuted /home/ttuser/of3t_trajectory/ref/negctl_permuted_grads.pt \
      --diffcap /home/ttuser/of3t_rebase/diffcap043/sub_boundary.pt \
      --sections perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
      --out perf/of3t_trajectory/AGREEMENT_WITH_UPSTREAMS_OWN_STEP.json \
      --sidecar-dir perf/of3t_trajectory/sidecar

    python3 perf/of3t_trajectory/d72_row_by_row.py \
      --d72 <SCORED_AGAINST_THE_RECIPES_OWN_FLOOR.json from wk/of3t-orchestrator> \
      --agreement perf/of3t_trajectory/AGREEMENT_WITH_UPSTREAMS_OWN_STEP.json \
      --out perf/of3t_trajectory/D72_ROW_BY_ROW_WITHOUT_THE_SHARED_SUBTRAHEND.json

## The upstream arms this reads

See `PROVENANCE.txt` for every digest. of3t-refprec was relaunching all four arms into
`of3t_refprec/run/` while this row ran, and `torch.save` truncates in place, so the live path is
not a reference. Read `of3t_refprec/pinned_p175/` and verify the digest before loading:
`ff78d7bc...` for arm4, `09f1217c...` for arm2. The files this row's numbers were computed from
hash identically to both.

refprec's it2 relaunch landed arm4 at 09:06:48Z byte-identical to the pinned copy, so the
seeded fixed-draw arms reproduce exactly across launches.

## Clock

No timing figure is published here. A gradient comparison is arithmetic, not throughput, so a
clamped clock changes how long a run takes and not what it computes. The AICLK is recorded
anyway, so it is possible to tell later which window these numbers came from: card 0, sampled
every 2 s from `qbcard/cardtel.tsv` DURING each arm, 800 to 1350 MHz with means of 1270 MHz
over the shipped arm (n=90), 1277 over the permuted-cotangent control (n=105) and 1219 over the
float64-softmax bound (n=113). The card bursts to 1350 under the arm and falls back to 800
between structures, so the fleet's idle-window reading of min=max=800 is not what these runs
saw.

## Checks the instrument carries

`agreement.py` checks the float64 file's model squared gradient norm against the campaign's
published 10.279642678524985 before it scores anything, so no share here is in a different
denominator than D72's. That check is a **relative tolerance of 1e-12, not an equality** (D78):
the constant is a sum over 4,170 tensors, float addition is not associative, and the record
already carries two honest spellings. This run produced a third, 10.279642678524981, a relative
drift of 3.456e-16, and the measured value is printed beside the expected one so a drift is
readable rather than binary. `MODEL_NORM_TOLERANCE_CONTROL.json` controls the tolerance in both
directions: both recorded spellings and a 1e-13 drift pass, a 1e-11 drift and a 1 % denominator
error fail. It also verifies that `diffcap043`'s `grad_f64`, the reference the
device arm was scored against, is bit-identical to the bundle's own float64 gradient over all
547 tensors (max abs diff 0.0). Without that identity our gradient and arm4's are not
gradients of the same loss.

## The break control's own limit

Seeding structure k with structure k+1's cotangent moves the small sections by 8.9x to 30.1x
and flips the scope cosine from +0.4109 to -0.2252, but it LOWERS the scope headline, 7.4267 to
3.7061, because the diffusion transformer's own error already exceeds what destroying the
sample pairing does. On a healthy arm the same class of control moves the number the way it is
supposed to: upstream's permuted-draws arm reads 4.0183e-01 against 5.852e-02, 6.9x. Read the
cosine, not the rel, when the arm under test is this far out.
