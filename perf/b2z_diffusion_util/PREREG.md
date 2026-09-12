# Pre-registration — diffusion-step core-utilization census

Written before this branch's own device run existed. Ordering stated honestly: the first
number in this pass came from re-analysing `b2z-kernel-cycle-census`'s already-committed
`ops_perf_step_qb2c0.csv.gz` (same card, same build, 3 reps of the same grabbed call), which
carried `CORE COUNT` and `AVAILABLE WORKER CORE COUNT` per program and had simply never been
asked this question. So the prediction below is NOT blind to that re-analysis. It is the
falsifiable claim for the independent run: a fresh grab, 5 reps, and a second instrument
(`ttnn.ReadDeviceProfiler` + `get_all_programs_perf_data()`) that never touches the tracy
host log.

PREDICTED (independent run, qb2 card 0, BH p300c):

1. Duration-weighted mean core utilization of one `Diffusion` call lands in **73-78 %**.
   Falsifier: outside that band means one of the two region-finding rules is wrong, and
   neither number may be quoted until they are reconciled.
2. The API's `num_available_cores` reads **110**, matching the tracy CSV's
   `AVAILABLE WORKER CORE COUNT` and the 11x10 grid of a p300c Blackhole processor.
   Falsifier: any other value; the brief asks for the measured grid, not the assumed one.
3. The two instruments agree on the per-program core counts they share to within 1 program.
   Falsifier: a systematic offset means the API counts something else (e.g. cores including
   dispatch/idle cores) and the utilization number has to be recomputed on one instrument.
4. The grid-starvation hypothesis **holds** for the diffusion step where it was refuted for
   the pairformer block, but the prize is small in fold terms: perfect linear re-spreading of
   every program to the full grid is worth ~1.0-1.1 s/fold, ~1.05x, and the real prize is
   much less than that because the under-filled programs are small and latency-bound, not
   core-bound. Falsifier for the "holds" half: >90 % duration-weighted utilization.
