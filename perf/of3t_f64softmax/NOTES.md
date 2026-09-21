# of3t-f64softmax — how each artifact here was made

Card 3 on qb2 (p300c, Blackhole). Branch `wk/of3t-f64softmax`, based on `wk/of3t-softgrad`.

## The arms

    perf/of3t_f64softmax/devgrad_f64.sh shipped      # CONTROL, must read 7.426217e+00
    perf/of3t_f64softmax/devgrad_f64.sh sitef64      # the code path
    perf/of3t_f64softmax/devgrad_f64.sh sitef64pc    # BREAK control, permuted cotangent
    perf/of3t_f64softmax/devgrad_f64.sh renorm       # AMENDMENT 1 Arm A, apbgrad's repair alone
    perf/of3t_f64softmax/devgrad_f64.sh sitef64rn    # AMENDMENT 1 Arm B, repair on the host path

`sitef64` sets `TT_BIO_HOST_F64_SOFTMAX_AB=all` through `device_gradient.py --softmax-site-f64`,
before the first `tt_bio` import — the selector is resolved at module construction, so an
environment set afterwards decides nothing. The arm patches no verb: it runs the shipped call
sites with the site flag on, which is the difference between this row and of3t-softgrad.

Each arm writes its own start and end epoch and `clockwin.sh <card> <start> <end>` reads the
AICLK out of `qbcard/cardtel.tsv` for that window. The column is resolved from the header by
name: `of3t_residual/clockwin.sh` hardcodes card 0's index, and on card 3 that would print a
plausible number for the wrong card.

## Scoring

    perf/of3t_softgrad/bars.py --f64 <grads_f64_043.pt> --bf16 <arm4_bf16_autocast/grads_f64.pt> \
        --bf16-sha ff78d7bc… --arms label=…pt          -> BARS_PER_TENSOR.json
    perf/of3t_f64softmax/controls.py --a <ours.pt> --b <softgrad sm64.pt>  -> CONTROLS.json
    perf/of3t_softgrad/cost.py --logs label=…log                           -> COST_ON_THE_REAL_ARM.json
    perf/of3t_softmax/softmax_cost.py --shapes 1x16x384x384 --card 3       -> softmax_cost_qb2c3.json

`softmax_cost.py` gained a fourth arm, `host_f64`, so the path's cost and its accuracy come out
of one run on the same shapes as the other three. That arm is the FORWARD round trip; under a
tape the backward pays a second one, and the both-ways figure is the scope cost.

## Inference

    perf/of3t_f64softmax/inference_blast_radius.py --model openfold3 --fixture … \
        --base-tree <detached worktree at 4fda3ccf6> --tree <this worktree> --card 3

The base tree is made with `git worktree add --detach <path> 4fda3ccf6` and removed afterwards.
Three arms — base, off, on — and the `on` arm is the control: a digest that could not see the
softmax would report byte-identity whatever the path did.

## AMENDMENT 1

`wk/of3t-apbgrad` is merged for its 19 lines in `tt_bio/taped_ttnn.py`, so the provenance of the
repair stays intact rather than being transcribed. `--softmax-bw-renorm` sets
`TT_BIO_SOFTMAX_BW_RENORM=1` before the first `tt_bio` import and the run publishes the value the
module actually read, because a flag consumed at import time is a flag a late `os.environ` write
cannot reach.

`tenstorrent.host_f64_softmax` honours the same flag. That is deliberate: it makes Arm B a
property of the code rather than of a harness patch, and without it the flag would stop at the
tape verb and never reach the host path at all — the host path does not call `ttnn.softmax`, so
`_v_softmax` is never entered and the renormalisation could not fire even in principle. A
structural no-op would not have answered the amendment's question, which is numerical.

    perf/of3t_f64softmax/rowsum_probe.py    -> ROWSUM_PROBE.json

measures both halves of the mechanism on the DiT's own softmax shape: the row sums each softmax
returns, and the `d_logits` row sums the repair exists to make vanish, with the repair off and on.
It is an accuracy probe, not a timing, so it carries no clock claim.

    controls.py --a <host_f64.pt>  --b <host_f64_renorm.pt>   -> CONTROLS_ARM_B.json
    controls.py --a <shipped.pt>   --b <bw_renorm.pt>         -> CONTROLS_RENORM_FIRES.json

The second pair is the firing census for Arm A. A lever that changed nothing would come back 547
of 547 bit-identical, which is exactly the shape of a relabelled shipped arm.

## Reading the numbers

`ours_vs_their_bf16` and `ours_vs_float64` have different denominators and A27 forbids conflating
them. The sqrt(2) in A26's reachable bar applies only to the first column, where both sides carry
error; against a float64 reference the bar is the threshold itself.
