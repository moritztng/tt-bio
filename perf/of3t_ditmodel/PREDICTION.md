# of3t-ditmodel — pre-registration, written and committed BEFORE the first arm ran

Registered 2026-09-22. Nothing in this row had touched a card when this file was committed;
`git log --diff-filter=A -- perf/of3t_ditmodel/` is the proof, and every artifact below carries
a start epoch later than this commit.

## What is being re-taken

`MODEL_shipped.json` reads `diffusion_module.diffusion_transformer` at **rel_L2 8.1943** against
float64 over **456 tensors** holding **43.6221 %** of the model's squared gradient norm. That
section is, to three significant figures, the whole of the campaign's model-scope gradient error.
This row re-takes exactly those 456 tensors, in the model bundle's denominator
**10.279642678524981**, against the float64 reference `grads_f64_043.pt`
sha256 `1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`, with
`refpath.assert_resolved()` read back in-process and the capture stamp checked against it.

## What I expect, and why

The brief's AMENDMENT 1 names the variable: arm A (`MODEL_shipped`'s diffusion part,
`of3t_f64softmax/device_grads_043all_shipped.pt`) records `softmax_bw_renorm_asked` and
`_live` both **null** and predates D56; arm B (`of3t-ditref`'s `r043_ours`) ran with the
row-sum-corrected softmax backward live. Same capture, same 547 tensors, same denominator, same
reference, `ref_norm` ratio exactly 1.0000 per tensor. So the difference is ours, not the
instrument's.

**Pre-registered primary value: `diffusion_transformer` rel_L2 vs float64 = 0.1167**, the value
`of3t-wholemodel`'s `renorm` arm measured over these same 456 tensors
(`MODEL_arms.json`, `per_section.renorm_vs_FLOAT64`, 0.11670185505910508).

**One correction to that expectation, registered here rather than after the fact.** That renorm
arm ran on **2026-09-21T00:00Z**, and D56 landed at **2026-09-21T17:49Z** (`1aa7070f5`). Before
D56 the composition routed **two** of the three softmax backwards through the corrected helper;
`tt_bio.autograd.softmax`, which `__all__` exports, still inlined the uncorrected expression.
Today's default routes all three and `SOFTMAX_BW_RENORM` defaults **True**. So the arm I am
about to run is not the arm that produced 0.1167: it is the shipped default of a strictly more
repaired tree. I therefore expect **at or slightly below 0.1167**, and I register the interval
**[0.06, 0.13]** as the band inside which I will call the prediction confirmed. Outside it in
either direction is a result this row has to explain, not round off.

## The break control

One arm with **`TT_BIO_SOFTMAX_BW_RENORM=0`**, everything else identical.
**Predicted: 8.1943 comes back**, within a few percent, on the same 456 tensors. If it does not,
D56 is not the variable, AMENDMENT 1 is wrong, and deliverable 2 is live after all. That is what
the control is for and it is the only outcome that would reopen the hunt.

## The model-scope consequence I am NOT predicting as measured

Substituting arm B's sections into `MODEL_shipped`'s table projects **0.0845** over 92.1568 %
of the mass against upstream's own bf16 at 0.0753, a ratio of 1.1226x, inside the A26 reachable
bar of 0.1049. That is the orchestrator's arithmetic on artifacts already on disk, and
`of3t-wholemodel`'s measured renorm arm reads **0.08301029446109433** whole-model against
float64. Both are pre-D56. Whatever my re-take lands on is the measurement; neither number above
is quoted by this row as one.

## A16 / A18

The zero-model baseline goes in the same artifact as every arm, through the same scorer, and a
reading at or above it publishes as a ceiling rather than as a failure. Before any gradient
number is believed, the forward at this boundary is checked: `of3t-ditref` recorded it at
**8.474800850934073e-03** bit-identical across two runs, and an arm whose forward moves is a
different capture and is refused.
