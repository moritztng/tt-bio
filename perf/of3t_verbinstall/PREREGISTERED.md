# of3t-verbinstall: what each arm must read, written before it ran

## D1. The package install reproduces `ceiling_hf3`

`perf/of3t_bwdaccum/dev_cot.py --lever ceiling_hf3` reads **0.4175214198121818** against
`REF_LOCAL_f64_n384` and **0.5545352626143085** against `REF_LOCAL_bf16_n384`
(`perf/of3t_trunkceiling/FRAME_CEIL_HF3.json`, qb2 p300c card 0).

`PKG_HF3` is the same frame with `--lever all` and the softmax taken from
`tt_bio.autograd.exact_softmax()` instead of from the harness patch, on qb1 card 1 (p150a).
Same boundary and cotangent: `boundary_n384.pt` sha256 `8cb3a586...` and
`block47_boundary.pt` sha256 `a55ef1c4...`, both matching `DEV_CEIL_HF3.json`'s recorded
hashes. Same `TT_BIO_SOFTMAX_PRECISE_AB=pairformer`, same `--pf-set s_fp32_residual=1`.

The board differs and the campaign has measured that this does not matter for this frame:
CTRL_B (qb1) and CEIL_B (qb2) agree to sixteen digits, and VERB_HF (qb1) and CEIL_HF (qb2)
both read 0.4179981990834974. So:

- **PKG_HF3 == 0.4175214198121818 to sixteen digits.** The install is the arm, and the
  campaign's best number now belongs to a shipped configuration.
- **Anything else is the finding**, and the first place to look is reach, not arithmetic: the
  counters banked in `EXACT_SOFTMAX_PKG_HF3.json` say how many softmaxes each half served, and
  a verb-only install leaves `triangle_attention._scores` on the card.

## D2. Where the site selector's 34.25 % lives

`ROUTE_HF` reads **0.5605347900452246** against float64 where `CEIL_HF` reads
**0.4179981990834974**: the arm that makes MORE softmaxes exact is 34.11 % farther from the
true gradient. The census difference is the lead. The route serves 1,685 calls the verb serves
none of (`served_raw`), plus 472 more taped ones: verb 5,285 served / 5,285 taped / 0 raw /
6,197 declined, route 7,442 / 5,757 / 1,685 / 0
(`CENSUS_VERB_HF.json`, `CENSUS_ROUTE_HF2.json`).

A raw serve is an exact FORWARD with no tape node (`autograd.py:920-935`), so the Jacobian
through it stays whatever the surrounding region was.

`ROUTE_NORAW` is `ROUTE_HF` with raw serves suppressed: a call arriving at the hook with
something that is not a taped `Tensor` gets `ttnn.softmax` instead of the float64 one. Nothing
else changes. Two-sided, and both outcomes are informative:

- **~0.4180 (within ~1 % of CEIL_HF)** — the raw serves carry the gap. The mechanism is that
  `_fp32_softmax_attention` serves the pair track's forward exact through the site while
  `ag.triangle_attention`'s chunked backward recomputes its own `ttnn.softmax` on the card, so
  the Jacobian is taken at activations the forward did not produce. Confirmed, and it is the
  same inconsistency `of3t-f64route` named, sitting in its own arm.
- **~0.5605 (unchanged)** — the raw serves are not the carrier, the 472 extra taped serves are,
  and **the candidate mechanism above is dead**. It does not get rescued afterwards.
- Anything between the two is a split, and it gets reported as a split with the fraction
  quoted, not rounded to whichever end is closer.

The 34 % is deterministic, so a single arm settles this: ROUTE_HF and ROUTE_HF2 on two
different cards agree to sixteen digits.

## D3. The inference A/B

Every model that executes the shared softmax site, **A/A before A/B**, with the capacity
census taken **per PID in the process that folded** -- D236 is four published A/Bs that counted
softmax calls in the launcher and read zero. `protenix-v2` leads: it buckets its token axis at
128 aa and provably never enters `_fp32_softmax_attention`, so a byte-identical fold there is
the negative control (pass 376). **A zero is evidence only once the same counter has been seen
non-zero under a condition this row controls**, so the census must show a non-zero somewhere
before any zero is quoted as a pass.
