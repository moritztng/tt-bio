# Not the result. One structure compared against a 48-structure reference.

`device_gradient_043.json` was taken with `--structs 0`. It differentiated ONE noised structure
and scored it against BUNDLE-MIN-043's gradient, which is a full training step accumulating all
48. So it compared 1/48 of the reference's scope and read median 9.778e-01 against a zero-model
baseline of 1.0 — a number that looks like a refutation and is an artifact of the scope.

`of3t-orchestrator` caught it and predicted the value from first principles before seeing any
re-run. With `G = sum of 48 g_k`, the `g_k` roughly independent and of similar magnitude,
`||g_0 - G||^2 ~ 47||g||^2`, so the ratio is `sqrt(47/48) = 0.9895`. Measured 0.9778. A
single-sample gradient against a 48-sample sum lands near 0.99 whether the port is right or
wrong, which is also why all 547 tensors sat over the bar uniformly.

`device_gradient.py:326` carries the check that would have caught it — the accumulation probe is
supposed to grow structure over structure, and "one that stays flat means the run is measuring
the LAST structure alone". With one structure there is one probe value and nothing to compare it
to, so the guard could not fire. The result is
`perf/of3t_rebase/device_gradient_043all.json`, `--structs all`, whose probe grows 1.2510e-04 to
3.3378e-03 across the 48, 26.68x, rising at 40 of 47 steps.

Moved here rather than labelled in place: a label beside a wrong number does not stop the number
being quoted.
