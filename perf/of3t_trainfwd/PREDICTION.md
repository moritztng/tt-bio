# of3t-trainfwd pre-registration

Committed before the first scoring run of this row. Five predictions, each with the reading
that would refute it. Nothing below is edited after a number exists; a miss is the finding.

Upstream is **0.4.3** throughout (D120). Every figure quotes its padded width (D180, D193) and
every reference figure names its host (D189).

## What this row builds

One adapter registered in `tt_bio/train/catalogue.py` under the name `openfold3`, in the shape
the catalogue already defines: `adapter(path, tokens=None) -> (forward, dataset)`. The registry
ships empty today, so this is the first registration and there is nothing to duplicate. The
featuriser is upstream's own, which is what the catalogue's docstring already says a model
arrives with; the forward composes the SHIPPED device modules (`InputEmbedderGlue`, `OF3Trunk`,
`OF3SampleDiffusion`, `OF3ConfidenceHead`) under `tt_bio.autograd.tape()`. No second forward.

## P1 -- the eight outputs, and which of them a batch can actually score

`af3_loss` keys eight terms off seven device outputs (`pred_xyz`, `pred_dist`,
`distogram_logits`, `plddt_logits`, `pde_logits`, `pae_logits`, `resolved_logits`).

PREDICTION: the forward produces all seven, and **`plddt` is the one term that is SKIPPED**,
because `_TERMS["plddt"]` reads `per_atom_lddt` and `per_atom_weight` from the BATCH and both
are functions of the prediction. Upstream builds them inside the loss under `no_grad`, and
`losses.atom_bespoke_lddt` is the same function. The objective's contract assumes every label
is batch-side; pLDDT's is not. REFUTED IF the featuriser emits either key.

## P2 -- `model_forward` fires

PREDICTION: with the adapter registered, one backward from the af3 seeds moves a parameter
gradient on **more than 2000** of the 4170 parameter tensors, against a control arm whose seeds
are zero and which moves **0**. REFUTED IF the moved count is under 2000, or if the zero-seed
control moves anything at all.

The rule is `COVERAGE_UNION.json`'s own: a path is covered when a parameter gradient MOVES
against an arm with the path off.

## P3 -- `diffusion_rollout` fires

The rollout arm is R denoise steps through the shipped sampler, differentiated. The OFF arm is
the same forward with the rollout output detached from the tape, so the diffusion module
contributes nothing and every other path is byte-identical.

PREDICTION: the difference of the two parameter-gradient vectors is non-zero and carries
**more than 50 %** of `diffusion_module`'s own squared gradient norm. REFUTED IF the difference
is zero, or if it lands outside `diffusion_module`, which would mean the detach cut something
else too.

## P4 -- the unstitched model-scope gradient

`MODEL_withtrunk_n384.json` reads 0.520124 from five separately captured legs (D187). With one
end-to-end forward the legs do not have to be stitched.

PREDICTION: **this row does not produce a comparable number this pass.** The pinned reference
`grads_f64_043.pt` (sha256 `1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`,
model denominator 10.279642678524981) was taken with upstream's OWN cotangent at
`num_recycles 0` with every stochastic draw REPLAYED from `draws_recycles0.pt`. An adapter that
seeds from `af3_loss` seeds from OUR loss, and an adapter that draws its own rollout noise
evaluates a different function. Both have to be pinned to the bundle's draws before a ratio
against that reference means anything. REFUTED IF a frame-matched number falls out anyway.

D187 stays open on that reading, and saying so is the deliverable. What is NOT acceptable is
asserting the stitching is gone without the unstitched number.

## P5 -- cost, and 20 steps

The taped trunk alone is 24x the inference step at crop 384 (230.11 s vs 9.721 s, trunk only).
A model forward adds the rollout and the confidence heads.

PREDICTION: one forward+backward through the adapter at crop 384 with `rollout=20` costs
**more than 600 s** on one Blackhole card, so 20 optimizer steps is **over 3 h** and is not
affordable inside a row's pass. AICLK sampled DURING the run, median reported. REFUTED IF it
comes in under 600 s.

## Inference

`INFERENCE:` is a digest equality, not a tolerance. The adapter adds a module and one default
argument; no shipped default moves and no inference path changes. The fold digest is
byte-identical or this row has broken something.
