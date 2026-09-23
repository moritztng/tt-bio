# of3t-softgrad — pre-registered, before any number exists

Written and pushed before the first arm runs. Nothing below moves after a result lands.

## The question

Two softmax levers could actually ship. Neither has ever been scored on the gradient at
full scope. The campaign knows only the two ENDS at scope, over the 547 tensors holding
51.1358 % of the model's squared gradient norm:

    arm                 ours_vs_their_step   ours_vs_float64   note
    shipped                   7.426217        7.568637         129.13x the threshold
    softmax_host_f64          0.0777758       0.05930664       1.3524x, not shippable

`softmax_host_f64` computes the softmax forward and backward on the host in float64. It is
a diagnostic BOUND and is not a product candidate. The middle two arms are the shippable
ones and they are what this row measures.

## The four arms

All four over the same 547-tensor diffusion device arm, same captured boundary, same
process, same card:

- `shipped` — `ttnn.softmax` through the shipped tape rule. CONTROL.
- `softmax_precise` — the shipped tape rule with `precise_config()` added to the FORWARD
  softmax call. `of3t-adaln`'s `precise_forward_rule`, verbatim.
- `softmax_accurate` — `tt_bio.tenstorrent._accurate_softmax` (max/subtract/exp/sum/divide
  out of individual ops) in the forward with `precise_config()`, and the shipped backward
  with `precise_config()` on its reduction. `of3t-adaln`'s `accurate_forward_rule`, verbatim.
- `softmax_host_f64` — the float64 host bound. CONTROL.

Both controls must reproduce. `shipped` must read 7.426217 and `softmax_host_f64` must read
0.0777758 against upstream's own bf16 step. If either does not, the harness is wrong, the
middle two arms mean nothing, and this row says so and stops rather than reporting them
against a drifting baseline.

## The bands, fixed here

On `softmax_accurate`, mass-weighted against upstream's own bf16 step at scope:

1. **inside the 2.0e-02 mass-weighted bar** — the scope is REACHABLE ON DEVICE. This is the
   campaign's biggest result in either direction, contradicts the block-8 ladder, and must
   be re-run before it is believed. Report the per-op cost beside it.
2. **between 2.0e-02 and 0.0777758** — it buys real ground and still misses. Report the gap
   as a multiple of the bar and the residual against the host bound.
3. **at or worse than 0.0777758** — the device levers do not approach the host bound at
   scope and the block-8 ladder was optimistic. Say that plainly.

Block 8 predicts band 2. `of3t-adaln`'s one-block reading, 384 tokens, 19 of 19 tensors,
bar 5.0e-02: shipped 0.785604 (15.71x), softmax_precise 0.349441 (6.99x), softmax_accurate
0.179144 (3.58x), softmax_host_f64 0.014892 (0.30x). One block is not the arm.

## Bars

Per-tensor **5.0e-02**. Mass-weighted **2.0e-02**. Reference upstream OpenFold3 **0.4.3**
float64, the revision `of3-p2-155k` binds to, pinned by digest before it is loaded (A24).

**A26-SCOPE**: against a float64 reference only one side carries error, so the reachable bar
is the threshold itself, no `sqrt(2)`. The `sqrt(2)` applies ONLY to the
ours-vs-their-bf16-step column, where both sides carry error. There the perfect-fix
threshold is 5.750945e-02 and an independent bf16 error of upstream's own size reads
8.1331e-02.

**A27**: `ours_vs_their_step` and `ours_vs_float64` are different measurements with
different denominators and are reported separately for every arm, each naming its reference.

**A28**: every reading carries its reachability floor. Upstream's own bf16 step on this
scope, scored against the same float64 reference, is 5.852018e-02 with r = 1.017575. A28
does not widen any bar.

**A23**: the headline is mass-weighted, with the norm ratio r and the error cosine beside
it (A25/D35), and a per-tensor count over the 5.0e-02 bar out of 547.

Masked always. NO CHEAT SPEEDUPS: no arm reaches a number by skipping the model's own work,
and every arm runs the same 48 structures through the same module.

## Controls, all of them

- **A16** zero model, measured, beside every reading.
- **instrument floor** — the two reproduced end arms are themselves the floor on drift.
- **upstream's own bf16 step** on this scope: 5.852018e-02 vs float64.
- **a BREAK control shown to fail** — `of3t-residual`'s permuted cotangent, which reads
  3.70587 on the shipped arm and 1.554075 on the bounded arm. It is run on a lever arm here.
- **intercept counts** — every arm publishes how many softmax calls its rule fired. A zero
  count means the arm is the shipped arm under another name and is a hard failure.

## Cost

Measured on the REAL arm, not the micro-bench. `of3t-softmax`'s per-op factors (precise
1.46x, accurate 4.71x) are a micro-bench and are not a scope cost. Seconds per structure per
arm, with the AICLK sampled DURING the timed work from `qbcard/cardtel.tsv`. A number
without a clock is not a measurement.

## Gated

Nothing ships. No default moves. Every arm stays on `wk/of3t-softgrad`, nothing merged.
