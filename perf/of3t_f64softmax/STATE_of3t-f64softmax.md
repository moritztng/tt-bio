# of3t-f64softmax — the host float64 softmax, built as a code path and scored at scope

Branch `wk/of3t-f64softmax`, based on `origin/wk/of3t-softgrad` as the brief directs. Artifacts in
`perf/of3t_f64softmax/`, card 3 on qb2 (p300c, Blackhole). Nothing is merged.

VERDICT: GO — the row's four deliverables are all met: the bound is a supported per-site code path
rather than a construction on a tape verb, it reproduces the bound exactly, a shipped fold does not
move by one byte with the path present and off, and the cost is recorded at op level and at scope.
The reading itself lands in the SECOND pre-registered band: 7.777580e-02 against upstream's own
bf16 step is 0.956x A26's reachable bar and 3.889x the 2.0e-02 mass-weighted bar. The 51.1358 %
scope is therefore REACHABLE but not REPRODUCED, and that is stated as the band requires rather
than argued around.

PATH: `tenstorrent.host_f64_softmax_site(token)` selects it, with the grammar
`accurate_softmax_site` and `softmax_precise_site` already use: `TT_BIO_HOST_F64_SOFTMAX_AB`, a
bare token on, a `-` prefix off, `all` / `-all` for every site without a token of its own. Three
tokens today — `openfold3.diffusion_transformer`, `openfold3.atom_transformer`,
`protenix.atom_transformer` — which is exactly the set `softmax_ckc` already reaches, and
`tests/test_host_f64_softmax_defaults.py` fails if the two sets ever diverge, so the path cannot
silently miss a site the precise lever has. Five call sites run it through one function,
`site_softmax(x, dim, host_f64=...)`; with the site off that call IS `ttnn.softmax` with the
caller's own arguments and nothing else, which is what makes the byte-identity below structural
rather than lucky. DEFAULT: off at every site, and a `default=True` anywhere fails a test. Under
the tape it creates one node whose backward applies the softmax Jacobian in float64 to the float64
forward output, not to the rounded copy that went back to the card — taking it off the rounded copy
would put a device-precision softmax straight back into the gradient the path exists to fix. The
round trip is confined to the softmax: every op before and after it stays on the device, and the
result returns in the input's own dtype, layout and memory config. 13 tests pass; the 61 tests of
the three sibling softmax/SDPA selector suites still pass unchanged. `compose_verify.sh`'s
assertion still holds — `tt_bio/openfold3_trunk.py` reads
`scale_pair_bias=False, tri_att_scale_pair_bias=False`, and this row did not touch that file.

GRADIENT: the 51.1358 % diffusion arm re-scored with the code path on. 48 structures, 547 tensors,
the same rebuilt 0.4.3 boundary of3t-softgrad used, card 3. Both references scored in one pass
(A27): `ours_vs_their_bf16` is against upstream OpenFold3's own bf16 training step, digest
ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb verified before loading (A24),
denominator THEIR gradient; `ours_vs_float64` is against `grads_f64_043.pt` built from upstream
0.4.3, denominator the float64 gradient.

    arm                      v their bf16   v float64      median f64   >5e-2 bar   mass>bar
    shipped (CONTROL)        7.426217e+00  7.568637e+00  1.250047e-01   459 / 547   94.4376 %
    host_f64 CODE PATH       7.777580e-02  5.930664e-02  4.885781e-02   254 / 547   52.0828 %
    host_f64 tape verb       7.777580e-02  5.930664e-02  4.885781e-02   254 / 547   52.0828 %
    path + permuted cot      1.554075e+00  1.572715e+00  1.576927e+00   547 / 547  100.0000 %

The shipped control reproduces **7.426217e+00** to every published digit on a different card from
the one that published it. The code path reproduces the tape-verb bound to every published digit,
and 547 of 547 gradient tensors are BIT-IDENTICAL between the two, largest absolute difference
exactly 0.0 (`CONTROLS.json`) — two constructions, two cards, one number. The path closes 95.5x of
the shipped arm's gap. Site census: 1440 softmax calls served with the flag on and 0 declined; 0
served and 1440 declined with it off, which is the same 1440 the tape-verb rule intercepted.
Forward `forward_rel_median` 6.462246e-03 against the shipped arm's 8.474801e-03.

GRADIENTS: 547 of 547 device tensors compared, holding 51.1358 % of the model's squared gradient
norm (57.3203 % of the diffusion module's own), so the scope is stated as a share of the NORM and
not as a count (A15). The worst case is located, not averaged. Against float64 the worst tensor is
`diffusion_module.diffusion_transformer.blocks.17.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at 2.618132e-01; against their bf16 step it is
`diffusion_module.atom_attn_enc.atom_transformer.blocks.0.attention_pair_bias.layer_norm_a_q.layer_norm_s.weight`
at 4.463192e-01. The shipped arm's worst is
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at 1.850397e+01. The worst tensor stays inside the AdaLN s-norm gain family across arms, so it is a
property of that family rather than of the lever. Median against float64 is 4.885781e-02, INSIDE
the 5.0e-02 per-tensor bar, while the mass-weighted headline is outside it: A23 in one line, and
why 52.0828 % of the mass sits over a bar only 46.4 % of the tensors cross.

BARS: against the 2.0e-02 mass-weighted bar, 7.777580e-02 is 3.889x. A26's reachable bar on the
ours-vs-their-step column is of3t-softgrad's published 8.1331e-02 (upstream's own bf16 error of
5.852018e-02, sqrt(2) for two independent error sources, divided by the norm ratio r = 1.017575),
and 7.777580e-02 is 0.956x of it — inside. Against float64 the sqrt(2) does not apply (A26-SCOPE,
only one side carries error) and the floor is the threshold itself: ours reads 5.930664e-02 where
upstream's own bf16 step reads 5.852018e-02, so our residual against an exact reference is 1.013x
theirs. That is the answer the second band asks for: against upstream's own bf16 we are 1.3 %
worse, and against their fp32 — which is what the float64 reference stands in for — we are 5.93e-02
where an exact implementation would be 0.

INFERENCE: byte-identical, by digest, blast radius measured rather than argued (D63). Three models,
one 128 aa fixture, card 3, seed 0, 6 sampling steps, one diffusion sample. Each was folded on the
tree BEFORE this row's commit (`base`, path absent from the source), on this tree with the path
present and at its default (`off`), and on this tree with `TT_BIO_HOST_F64_SOFTMAX_AB=all` (`on`):

    model         base == off  sha256 of the written CIF (base and off)            on moves it
    openfold3     yes          0156ce8c0946f9d05791a5fd63e246619cb5d464ef770a…      yes  b665d1ae…
    protenix-v2   yes          39bce7750297a920c7cac53da931ae774012f9581a31f…      yes  940935ff…
    opendde       yes          0cc1cdf1d31e74c57766886d836c8aa735508cd18b1d9…      yes  ce47cf2f…

The `on` column is the control that stops the first claim being vacuous: a digest that could not
see the softmax would report byte-identity whatever the path did. `BLAST_RADIUS_*.json` carries
every digest.

COST: named, not paid down here — Moritz's sequence is fidelity then performance.
**At op level**, `[1,16,384,384]` fp32 on card 3 at **1350 MHz** AICLK sampled DURING the timed
work by two independent instruments that agree (`clocksample`: min 1350 / median 1350 / max 1350
over 3 samples; `clockwin.sh` on card 3: n=4, mean 1350, min 1350, max 1350), four arms interleaved
A/B/C/D then D/C/B/A over 12 rounds of 40 calls:

    arm        error vs float64   ms/call   x the op default
    none          2.029300e-02     0.0545            1.000
    precise       1.645770e-03     0.0790            1.449      12.3x better accuracy
    accurate      5.156148e-04     0.2576            4.724      39.4x better
    host_f64      2.081970e-08     9.0966          166.842     974,700x better

That last row is the mechanism in one number: 2.08e-08 is the fp32 storage of the result, not the
softmax, so the host arm confirms directly that a true fp32 softmax lands near 1e-7 and that no
on-device arm is within four orders of magnitude of it. What `precise_config()` recovers on one op
is 1.449x cost for 12.3x accuracy, reproducing of3t-softmax's published 1.46x on a different card.
**At scope**, four arms interleaved shipped / path / path / shipped, AICLK read DURING each arm's
own window from card 3's column in `qbcard/cardtel.tsv`:

    arm           s/structure   x shipped   wall   AICLK mean during
    shipped r1         1.2250       1.000    64 s   1316 MHz (n=32)
    host_f64 r1        1.8063       1.474    91 s   1314 MHz (n=46)
    host_f64 r2        1.7958       1.466    91 s   1326 MHz (n=46)
    shipped r2         1.2250       1.000    65 s   1315 MHz (n=32)

A/A floor is 1.000x on seconds per structure and 1.016x on wall clock, so the 1.47x is 29x the
floor. of3t-softgrad measured 1.441x for the same arithmetic on card 0; the round trip is host
bound, and cards 0 and 1 were carrying two other workers' folds throughout this row's arms, which
is the honest reason for the spread. The gap between 166.8x on the op and 1.47x on the arm is the
softmax's share of what 48 structures run, and it is why a micro-bench factor may not be carried to
a scope. Determinism: the second replicate of each arm rewrote every report file with byte-identical
content, so `git status` on the artifact directory came back clean after the re-run.

CONTROL: four, all measured, none assumed. **A16 zero model** read through the same comparator the
arms are read through: 1.0 against their bf16 step and 1.0 against float64, over the same 547
tensors (`CONTROLS.json`). **Instrument floor zero**, not estimated: the code path and
of3t-softgrad's tape-verb construction agree on 547 of 547 tensors bit for bit, largest absolute
difference exactly 0.0, across two constructions and two cards. **A28 upstream floor**, beside every
row: upstream's own bf16 step against the same float64 reference is 5.852018e-02, with 441 of its
own 547 tensors over the 5.0e-02 bar carrying 41.6989 % of the mass. **A break control that is
shown to FAIL**, run on the lever arm rather than the saturated shipped one: seeding structure k's
backward with structure k+1's cotangent takes the path from 7.777580e-02 to 1.554075e+00, 20.0x,
with all 547 tensors over the bar carrying 100.0000 % of the mass — and the forward is byte-for-byte
the arm's own (6.462246e-03 in both), so it breaks the comparison and not the model. A fifth control
is the inference `on` arm above, which moves three digests the `off` arm holds fixed.

GATED: nothing is merged and no shipped default moved. The path is off at every site, the three
fold digests prove that costs a user nothing, and the flip to a default would be a separate decision
with its own evidence. Everything lives on `wk/of3t-f64softmax`; the gate is Moritz's.

PROVES: that the host float64 softmax is a supported code path in tt-bio — selected per site by the
grammar the other softmax levers use, off by default, confined to the softmax, differentiable under
the tape — and not a construction on a tape verb; that at full scope over 547 tensors holding
51.1358 % of the model's squared gradient norm it reads 7.777580e-02 against upstream's own bf16
step and 5.930664e-02 against a float64 reference built from upstream 0.4.3, reproducing
of3t-softgrad's diagnostic bound to the bit on a different card; that this is 0.956x A26's reachable
bar and 3.889x the 2.0e-02 mass-weighted bar, the second pre-registered band; that the shipped
control reproduces 7.426217e+00 and the break control fails at 20.0x; and that OpenFold3,
Protenix-v2 and OpenDDE each write a byte-identical structure with the path present and off.

DOESNOT: this is one gradient at one boundary and it says nothing about stability over a full
training run. The campaign proves an update rule over N steps; 100k steps of it staying on
upstream's trajectory is not bounded here, and neither is long-run drift. One crop (384 tokens, 56
real), one checkpoint (`of3-p2-155k`), one card, so it does not establish the same ladder at another
crop or on another board. 48.8642 % of the model's squared gradient norm has no measurement in this
row. The lever was scored on the GRADIENT: no fold Angstrom reading and no seed floor was taken with
it on, so this says nothing about what it would do to a structure if it were ever switched on for
inference — the digests say only that it does not fire by default. And the 1.47x is the cost of a
host round trip per softmax on a quiet card with two busy neighbours; making it cheap is a
performance question this row deliberately does not open.
