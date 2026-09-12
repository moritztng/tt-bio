# The host residual with TT_BIO_DEVICE_CONDITIONING on — Wormhole, whglx card 11

Every second here is a **Wormhole** second (whglx, 8x9 grid, `wormhole_b0`, card 11, one chip).
The published cell is Blackhole and these seconds do not transfer to it; the attribution does.
Host-side Python is architecture-independent, which is the only reason this row is worth running
on Wormhole at all.

Commit `7ea19829` (`origin/main` 84da2a49 + `wk/b2z2-host-residual-zero`), HOST on, 512 aa
`cdk2x2_512` + its fixed 35-row a3m, 3 recycles, 200 sampling steps, 1 sample, seed 0,
templates off, timed at `predict_one`. Cold fold 73.195 s discarded. Two plain folds 39.558 /
39.455 s — an A/A spread of **0.103 s = 0.26 %**. Model load, outside the timed region: 6.11 s.
Raw: `base_hoston_whglx_c11.json`.

## The fold

| | s | % of fold |
|---|---|---|
| fold (`predict_one`) | **39.605** | 100 |
| `trunk` (device) | 26.556 | 67.1 |
| `denoise_device`, 200 calls (device) | 8.937 | 22.6 |
| `pairformer_conf` (device) | 1.033 | 2.6 |
| **host residual** | **3.079** | **7.77** |
| unattributed | 0.012 | 0.03 |

**3.079 s, attributed to 99.6 %.** It was 3.573 s on the same instrument before HOST
(`b2z2-host-residual-zero`, whglx card 6, 40.667 s fold) — but those two folds ran on a
differently-loaded box, so read the shares, not the difference. As a share the residual went
8.8 % -> 7.8 %.

## The five terms, by seconds

| # | term | s | where | removable? |
|---|---|---|---|---|
| 1 | confidence head, host part | **0.894** | `ConfidenceModule.forward` before its pairformer, plus `ConfidenceHeads` | **taken** by `b2z2-conf-head-device` |
| 2 | sampler's per-step host, 200 steps | **0.659** | `weighted_rigid_align` 0.341 (1.71 ms x 200), the loop's own Python 0.239, `randaug` 0.038, `denoiser` glue 0.040 | partly — align is an SVD per step |
| 3 | `z_init` pair assembly | **0.647** | `Boltz2.forward` exclusive 0.356 + `contact_cond` 0.182 + `rel_pos` 0.109 | **yes, to the device** |
| 4 | featurisation + parsing | **0.362** | `featurize` 0.244, `parse_yaml` 0.089, `tokenize` 0.023, `parse_a3m` 0.005 | host by nature |
| 5 | diffusion conditioning, atom track | **0.172** | `AtomEncoder` under `forward_atoms` | maybe |
| 6 | CIF writer | 0.154 | `write_result` | host by nature |
| 7 | input embedder | 0.140 | `InputEmbedder.forward` (0.098 of it its own atom encoder) | with #3 |
| 8 | distogram head | 0.038 | `DistogramModule` on the trunk's `z` | with #3 |

Rows 1, 3, 5, 7 and 8 are the same class of work the two device ports already deleted twice: a
channel map applied independently at each `(i, j)` over a `[1, n, n, c]` tensor, done in torch on
the host and then pushed across PCIe. Rows 2, 4 and 6 are not that.

## Row 3, which is what this row builds

`Boltz2.forward` builds the trunk's pair input on the host:

    z_init = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]
    z_init = z_init + self.rel_pos(feats)
    z_init = z_init + self.token_bonds(feats["token_bonds"].float())
    z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
    z_init = z_init + self.contact_conditioning(feats)

At 512 tokens each of those terms is `[1, 512, 512, 128]` fp32 = **134 MB**, and the result is
then uploaded to the device by `TrunkModule._build_static`. Every one of the five terms is a
per-`(i, j)` map: two broadcasts of a `[1, n, c]` projection, two table gathers (relative
position, bond type), one 1-channel linear (`token_bonds`) and one Fourier block
(`contact_conditioning`). `PairConditioningDevice` (HOST) and `ConfidencePairDevice`
(`b2z2-conf-head-device`) already run exactly this op set on the device for the other two pair
tracks in the model.

So the trunk's own input is the third instance of a pattern this campaign has built twice, and it
is the only one of the three that also deletes an upload the device then has to wait on.

## Correction to row 3's own price, measured before anyone builds on it

The five `[1, 512, 512, 128]` fp32 terms are **not** where row 3's 0.356 s of `Boltz2.forward`
exclusive goes. Timed directly on the same box (whglx, 32 torch threads, loadavg 12.9, four
repeats, no device involved):

| | s |
|---|---|
| `z_init_1(...)[:, :, None] + z_init_2(...)[:, None, :]` broadcast | 0.005 |
| the four `z = z + term` adds | 0.075 |
| `torch.zeros_like(z_init)` | 0.005 |
| **the pair arithmetic, total** | **0.085** |

So **0.27 s of the 0.356 s glue is something other than the z_init tensor arithmetic** — the
per-forward `disable_and_clear_program_cache()` / `enable_program_cache()` pair, the
`for m in self.modules()` static-cache reset, `pair_mask`, and the `.float()` copies of `s` and
`s_inputs` handed to the sampler. That term is unnamed and has to be split before it is priced.

Row 3's addressable seconds are therefore **rel_pos 0.109 + contact_cond 0.182 + pair arithmetic
0.085 + the two bond linears (inside the glue, not yet split) ~= 0.43 s**, not 0.647 s. Quoting
0.647 s for a device port of `z_init` overstates it by ~1.5x.

## And what the rest of the glue is — measured, not guessed

`--phases cacheclear` prices the one suspect that would have been free to fix. `Boltz2.forward`
clears and re-enables the device program cache on every call; on a populated cache that pair
costs **0.026 s** to clear and **0.000 s** to re-enable (two repeats, whglx card 11, loadavg 22).
So it is 7 % of the 0.356 s glue, not the glue.

That leaves `token_bonds` — a `[1, n, n, 1]` -> 128 linear — and `token_bonds_type`, an embedding
over `[1, n, n]`. Each materialises another 134 MB on the host, and both sit inside
`Boltz2.forward` rather than in a named region. Together with the 0.085 s of adds they account
for the glue, and all three are inside what a device `z_init` deletes.

(The instance-level brackets added for those two terms did not take: assigning a timing wrapper
over an `nn.Module` child raises in `nn.Module.__setattr__` and the harness swallowed it, so the
rows are absent rather than zero. Raw: `glue_hoston_whglx_c11.json`.)

## Row 3, built and measured

`TT_BIO_DEVICE_ZINIT` (`tenstorrent.PairAssemblyDevice.for_trunk`) builds the whole sum on the
device, inside the trunk's own static build, so nothing is uploaded. Measured on whglx card 11,
WH, commit `c44d6e83`, 6 interleaved paired folds in one process:

| | |
|---|---|
| median on / off | 39.099 / 39.525 s |
| ratio | **1.01090x** |
| paired mean | **+0.3685 s**, sd 0.1913, t = 4.72, **6/6 positive** |
| arm spreads | on 2.10 %, off 1.61 % — larger than the effect; the pairing is what carries it |
| device track added | **98.9 ms** (51 % of it one packed upload) |
| `cdk2x2_298` control | **0.298180 A** all-atom / 0.143312 A CA, A/A floor 0.000000 A, bar 0.35 A |

The fold sees 0.3685 s of the ~0.58 s the brackets predict, because part of the host assembly was
already overlapping queued device work.

## What is left of the residual after row 3

Rows 2 (the sampler's 200-step host, 0.659 s) and 4 (featurisation and parsing, 0.362 s) are
untouched and are now the largest host terms that nobody owns. Row 2 splits as
`weighted_rigid_align` 0.341 s over 200 calls and 0.239 s of loop Python; the align is a per-step
Kabsch superposition, so it is a device candidate but not a cheap one. Row 4 is parsing and
featurisation, which no device can help.
