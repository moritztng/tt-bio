# Cold-start kernel cache

tt-metal JIT-compiles every kernel to a device binary the first time it sees a
given op/shape. The compiled binary is cached on disk and reused by any later
process on the same host — `tt_bio/main.py` points `TT_METAL_CACHE` at
`~/.cache/tt-metal-cache-tt-bio/ttnn-<version>` (an operator-set
`TT_METAL_CACHE` always wins). The version in the path is read from the
`ttnn` package actually resolved at runtime, so a version bump — or running
against a different venv — gets its own cache instead of silently reusing a
binary built by a different tt-metal.

This does not help steady-state serving: the long-lived serve worker already
builds its model once and stays resident. It helps a **fresh process** —
ad-hoc CLI calls, `predict --devices` fan-out workers, and the first job after
a restart or deploy.

## Measured (ESMFold2, L=76, Blackhole P150, `--recycling_steps 3 --sampling_steps 20`)

Reproduce: run `tt-bio predict --model esmfold2` twice, once against an empty
`TT_METAL_CACHE` dir and once warm.

| cache state | wall clock | reported compute stage |
|---|---|---|
| empty (first-ever compile) | ~177s | ~116s |
| warm (2nd+ process, same shapes) | ~52-100s | **12.2s** |

The compute stage drops ~9x once the cache is warm. Wall clock drops less
(~2-3x) because most of the remaining time is process startup, weight
loading, and device open/close — none of which the kernel cache touches.

## Correctness

Same seed, empty vs warm cache: output `.cif` files are byte-identical
(matching md5sum). The cache stores compiled binaries only — it cannot change
model output.
