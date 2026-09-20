# of3t-traj20 — what ran, and how to re-run it

PROTOCOL §7's assembled 20-step trajectory, at OpenFold3's real parameter scope. The result
lives in `~/.coworker/state/of3t-traj20.md`; this file is the operating record.

## Re-running

    cd <worktree>
    env OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 \
      /home/ttuser/ptxft-venv/bin/python3 perf/of3t_traj20/traj20.py --arm scaled

Pin the thread count. Five arms at each library's default put a 16-core host at load average 53
and cut the per-rung rate from 21 s to 200 s: the work is memory-bound over 368 M elements, so
oversubscription buys nothing. `traj20.py` now sets 4 threads itself if the environment does
not. One arm is about 7 minutes and about 40 GB resident; three at a time fit in 249 GB.

Inputs are `/home/ttuser/of3t/bundle_min/{w0_r0_rebuild,grads_f64_r0}.pt` and upstream's own
`grad_manager.py` and `lr_schedulers.py`, copied to `/home/ttuser/of3t_traj20/upstream/` and
hashed into every result file. `ptxft-venv` is the only python on qb2 with torch.

    perf/of3t_traj20/table.py    perf/of3t_traj20/traj20_*.json   # the arm table
    perf/of3t_traj20/detail.py   perf/of3t_traj20/traj20_*.json   # every rung of every arm
    perf/of3t_traj20/gainladder.sh                                # the loop-gain ladder

## The arms

`shipped` and `scaled` are the two runs §7a requires. `wd0`, `wired` and `avg` close the first,
the first two, and the first three wiring divergences, so each one attributes exactly what it
closes. `miswire` restores D11's off-by-one and `stale` feeds step k-1's gradient; `avgstale`
is `stale` on the healthy arm, which is the only place it is not saturated.

## What a successor should pick up

With all four divergences closed the k = 2 divergence is 8.715e-03, 17.3x the fp32 differencing
floor, and the super-linear shape survives at +1.267. Two candidates and this row did not
separate them: a fifth wiring divergence, or the closed loop amplifying each stack's own fp32
rounding. The discriminator is cheap — re-run `avg` with both sides in float64 and see whether
the residual moves with the dtype. `EVERY_RUNG.txt` holds the rung-by-rung data either way.

The other open thread is §7b itself. Its shape bar passed the deliberately mis-wired arm
(exponent -0.707) and failed the healthiest one (+1.267), because every arm converges on a
common ceiling near 1e-01 by k = 20 and the exponent measures how far below that ceiling the
arm started. Any future row quoting a growth exponent from a saturating trajectory owes the
early rungs beside it.


## What `of3t-wirefix` did with this

Both open threads above are answered, and the answer to the first is not either of the two
candidates this file named.

**The residual was a fifth wiring divergence, not the loop amplifying rounding.** `recipes.py`
never passed `betas`, so `AdamW`'s class default `(0.9, 0.999)` shipped where OpenFold3 runs
`(0.9, 0.95)`. Closing it takes k=2 from 8.715e-03 to 4.9873e-06 and k=20 from 9.741e-02 to
8.2071e-06, and the growth exponent from +1.267 to +0.194.

The discriminator this file proposed, re-running `avg` in float64, was replaced. `AdamW`
refuses a non-fp32 master at construction for a measured reason, so a float64 arm would have
meant changing a shipped guard to run an experiment. The `ulp` arm answers the same question
without touching shipped code: it runs one path on both sides and perturbs one side's
accumulated gradient by a relative `2**-24` per element per step, formed in float64 and
rounded back. It reads 3.4668e-06 at k=2 falling to 1.8575e-06 at k=20, exponent **-0.146**,
so the closed loop **damps** rounding over 20 steps rather than amplifying it. That is what
ruled the rounding candidate out and pointed at a constant.

**The §7b thread.** The saturation confound is real and it is what made the exponent order the
arms backwards here. It does not reach the post-fix arm: `fixed5` sits four orders below the
~1e-01 ceiling at every rung and never approaches it, so its +0.194 is the shape of the curve.
Its r² of 0.093 is the more honest number, and it says the curve is flat scatter rather than a
law. Any future row quoting a growth exponent still owes the r² and the early rungs beside it.

    perf/of3t_traj20/wirefix_report.py > perf/of3t_traj20/WIREFIX_RUNGS.txt
    perf/of3t_traj20/schedule_family_shipped.py    # the schedule family on the shipped default
