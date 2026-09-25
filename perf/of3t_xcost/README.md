# of3t-xcost — what the exactness costs, and how much of that is not arithmetic

`exact_training`'s host float64 softmax and layer norm are 95.2 % of the OF3T backward
(`of3t-bwattrib`) and the fidelity they buy is load-bearing in full (`of3t-exactscope`). The
only question left is the price. These three scripts answer the half of it that needs no card.

**The additive axis is verb SELF-time by (verb, shape, dtype).** It sums to `verb_wall_s`, and
`verb_wall_s` plus the non-verb remainder is the step. The closure table in
`perf/of3t_bwattrib/out/` is INCLUSIVE of the verbs its closures issue, so the two cannot be
added, and it has a `closure: None` bucket holding the 108 recompute softmaxes that
`triangle_attention`'s own backward runs.

| script | opens a device? | what it measures |
|---|---|---|
| `hostprice.py` | no | the host float64 softmax forward and backward at `[384,4,384,384]`, swept over chunk size. The rate is flat, so the monolithic call is not paying for its 1.81 GB temporaries. Every chunked arm is `torch.equal` to the monolithic one. |
| `layoutprice.py` | no | the host tilize and untilize inside `from_torch` / `to_torch`, isolated by building host-only ttnn tensors. 0.0737 s and 0.1342 s per call at the production shape: 44.90 s per backward, 6.36 %. |
| `crossprice.py` | **yes, one card** | the crossing split into down / math / up / drain at `tt_bio.autograd.host_f64_softmax_values`, the one function both the raw and the taped exact softmax funnel through. |

`crossprice.py` is the arm that has not run. Three arms, three separate processes, one device
context each:

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-xcost \
      python3 perf/of3t_xcost/crossprice.py --arm base \
        --out perf/of3t_xcost/out/CROSS_384_base.json

then `--arm xsync` and `--arm rm`. `xsync` syncs the device immediately before the `to_torch`
and times the drain as its own leg: if `down_s` collapses onto `drain_s`, the 168.51 s was the
`_scores` matmul draining into a blocking call and the bus is innocent. `rm` crosses in
ROW_MAJOR with the layout conversion on the device, and checks the first `--check` calls
bit-identical against the production route before it is allowed to claim a second.

Read `state/of3t-xcost.md` for the sizing and the two candidates that are already dead.
