# BoltzGen diffusion-batch-size threshold: investigated, not lowered (dead end)

**Goal:** `tt-bio gen`'s `--diffusion_batch_size` defaults to 1 for `--num_designs < 100`
(`_default_diffusion_batch_size` in `tt_bio/boltzgen/cli/boltzgen.py`), so the batching
path never fires for JapanFold's actual production range (`num_designs <= 10`, see
`DESIGN_PARAMS`/`max_designs` in the aiand-bio platform's `catalog.py`). Every real
design job runs its samples one at a time. This investigated whether lowering or
dropping that threshold is a safe throughput win for `num_designs=1..10`.

## Mechanism: it is the Boltz-2 "multiplicity" pattern, not distinct-structure batching

`diffusion_batch_size` flows to `data.cfg.multiplicity`/`diffusion_samples` overrides
on the `design` pipeline step (`boltzgen.py:1324-1327`), which land on
`Boltz.forward(..., diffusion_samples=N)` → `structure_module.sample(..., multiplicity=N)`
(`tt_bio/boltzgen/model/models/boltz.py:558`, `.../model/modules/diffusion.py:291`). One
trunk pass produces a single `(s_trunk, s_inputs, diffusion_conditioning)` per dataset
item; `multiplicity=N` diffusion **samples of that identical conditioning** are then
generated in one batched device call — the same shared-conditioning "N samples of one
structure" trick used by Boltz-2 (`AtomDiffusion.sample`, `repeat_interleave(multiplicity)`),
**not** the newly-enabled distinct-per-slot-conditioning path that
[[boltz2-throughput-loop]] proved lossy. This checks out mechanically: `--diffusion_batch_size`'s
own help text already documents that all designs in one batch share the same (randomly
sampled) binder length, confirming shared conditioning across the batch.

Given [[boltz2-throughput-loop]] found Boltz-2's multiplicity path bit-exact, the working
hypothesis going in was that BoltzGen's batching — same mechanism, same kind of shared
`AtomDiffusion`/`TTDiffusionModule` device backend — would transfer as lossless. **It does
not**; see below.

## Measured throughput (real, but see parity verdict)

`examples/binder.yaml` (canonical validated target, per `docs/boltzgen-designability.md`),
`--steps design` only, single checkpoint, `sampling_steps=30` (reduced for iteration
speed — not production quality), `--num_designs 10`, TT_VISIBLE_DEVICES=2 (an idle card;
card 0 was occupied by another worker's job):

| `--diffusion_batch_size` | batches | wall time | speedup |
|---|---|---|---|
| 1 (today's default) | 10×1 | 107 s | 1.0× |
| 2 | 5×2 | 69 s | 1.55× |
| 10 | 1×10 | 51 s | 2.1× |

A real, consistent speedup — batching would help if it were lossless.

## Parity verdict: LOSSY — do not ship

Per-slot RNG-isolated parity harness: `scripts/boltzgen_batch_parity.py`. Reuses the real
`AtomDiffusion.sample()` verbatim; monkeypatches `torch.randn` so slot *i* always draws
from its own dedicated generator regardless of batch size (same technique as Boltz-2's
`parity_batched.py`, see [[boltz2-throughput-loop]]). Fixed-length design spec (single
length, not `binder.yaml`'s `80..120` range) to remove the random-length draw as a
confound, so the *only* variable between a standalone (`multiplicity=1`) and batched
(`multiplicity=2`) run of the same seed is on-device batch-size-dependent kernel numerics.

**Determinism control:** two standalone (`multiplicity=1`) runs at the same seed are
bit-identical (`raw_maxdiff=0.0`) — the unbatched diffusion path is fully deterministic.

**Batched vs standalone**, 30 sampling steps, identical per-slot noise (two independent
script runs):

| run | slot0 Kabsch RMSD | slot1 Kabsch RMSD |
|---|---|---|
| 1 | 0.54 Å | 2.82 Å |
| 2 | 1.22 Å | 0.79 Å |

Non-zero, substantial (comparable to or exceeding BoltzGen's own **≤2 Å strict**
designability bar, at only 30 of 500 production sampling steps), and **not even
reproducible run-to-run at fixed multiplicity=2** — unlike the bit-exact-reproducible
`multiplicity=1` path. So batching in BoltzGen's `AtomDiffusion`/`TTDiffusionModule`
device path is **not batch-invariant**, contradicting the Boltz-2 analogy: apparently
Boltz-2's proven-lossless multiplicity claim does not transfer to BoltzGen's use of the
same underlying mechanism. Given `AtomDiffusion.sample`'s per-step Euler-Maruyama loop
integrates noise over hundreds of steps, this kind of per-forward drift would plausibly
amplify further at full 500-step production sampling (mirrors the "chaotic amplification"
Boltz-2 measured for its own lossy distinct-conditioning path).

## Second, independent reason not to lower the threshold

Even setting numerics aside: real JapanFold design specs sample a random binder length
per design (`examples/binder.yaml`: `sequence: 80..120`; confirmed in aiand-bio's
`catalog.py` `DESIGN_PARAMS` via `lengthRange`, e.g. `"80..120"`). Because one
`diffusion_batch_size` group shares one trunk pass, every design *within* a batch shares
the *same* sampled length — already documented in `--diffusion_batch_size`'s own help
text. At production scale (`max_designs=10`), a full-batch default
(`diffusion_batch_size=10`) would collapse the entire job to **one length draw across
all 10 designs**, a real diversity regression for exactly the regime this task targeted.

## Verdict

**DEAD END — do not lower or drop the `num_designs < 100` threshold.** Both the
"sacred parity" bar (real, non-reproducible ~0.5–2.8 Å drift at reduced steps) and a
second independent product cost (collapsed length diversity at production scale) rule
it out. The existing default (`diffusion_batch_size=1` below 100 designs) is correct
as shipped. Making this lossless would need a batch-invariant device diffusion kernel
for `TTDiffusionModule` (tt-metal level) — out of scope, and per
[[boltz2-throughput-loop]] and the device-resident-diffusion dead end, not worth
pursuing further here.

Harness: `scripts/boltzgen_batch_parity.py`.
