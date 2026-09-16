# The 0.93 s the trace cannot reach is probably featurization and output writing

The clock-immune term is **3.952 s measured**; the diffusion trace reaches about **3.02 s** of it
if the term is per-call dispatch. That leaves **0.93 s** unowned, and it was the ladder's only
unowned item.

The one artifact that measured host *stages* at this model's shapes is
`perf/b2x_host_residual/hostpath_512_pc.json`:

| stage | pc CPU seconds | |
|---|---|---|
| `diffusion_conditioning` | 1.0896 | now device-resident, `TT_BIO_DEVICE_CONDITIONING` defaults True |
| `prepare` | 0.2893 | still host |
| `input_embedder` | 0.1357 | still host |
| `write_result` | 0.1150 | still host |
| `rel_pos` | 0.0720 | still host |
| `to_batch` | 0.0001 | still host |

Drop the stage that has since moved to the device and **0.612 s of named, measured host work
remains — about two thirds of the 0.93 s.** Feature preparation, the input embedder,
relative-position encoding and writing the CIF. That is real work, not overhead, which is why a
trace cannot touch it and why the remainder is the hard part of the fixed cost.

It also validates the decision that was already taken: `diffusion_conditioning` was by far the
largest host stage at 1.0896 s, with `pairwise_conditioner/transition` alone at 1.597 s inclusive
across 10 calls in its subtree, and it is exactly the one that got moved onto the device.

## Why this is a hypothesis and not a result

- Measured on **pc's CPU** (12 cpus, 6 torch threads), not qb2's. The seconds do not transfer;
  only the structure and rough scale do.
- Older tree, 2026-09-11, and relative-position gathering has shipped as a host lever since, so
  `rel_pos` is probably smaller now.
- That census ran CPU-only with synthetic tensors at the model's shapes, by its own note. It is a
  cost model of the host path, not an observation of a real fold.
- The 0.93 s is itself derived — a measured 3.952 s minus a derived 3.02 s. Subtracting two
  derived numbers does not produce a measurement, and the 66 % agreement could be coincidence.

**`c10-fixed-cost`'s 298 aa arm is the discriminator**, because featurization and CIF writing
scale with target size while per-call dispatch scales with call count. If the fixed term turns out
to be size-dependent, this reading is supported; if it is flat, the remainder is something else.

    python3 remainder.py                      # prints remainder.json
    python3 -m pytest test_remainder.py -q    # 7 controls
