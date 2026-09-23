# Diffusion samples and recycles: what they cost

`--diffusion_samples` and `--recycling_steps` multiply the work of a fold independently of
sequence length. This page gives the measured cost of raising either one, so you can size a job
before you submit it.

Measured on a Wormhole Galaxy (j10glx02), one chip per fold, AICLK 1000 MHz sampled during
every fold, on the size ladder's `cdk2x2_<tokens>` fixtures, single sequence, seed 0.
The box was shared with other jobs while these ran, so absolute seconds are pessimistic. The
ratios below are what carries over.

## Samples

Every structure model denoises its samples in chunks and runs confidence one sample at a time:

| model | chunk width |
|---|---|
| Boltz-2, Protenix-v1/v2, OpenDDE, OpenDDE-abag | `--max_parallel_samples`, default 5 |
| ESMFold2, ESMFold2-fast | as many as fit in free device memory, halved on a refusal |
| OpenFold3, OpenBind, RF3 | 1 |

**Memory** grows with the samples in one chunk and stops there. Past the chunk width, 25 or 100
samples need exactly the device memory that 5 do. Boltz-2 at 768 tokens adds 0.21 GiB per unit
of chunk width. For most models the largest allocation of the whole fold is in the trunk, not in
sampling, so the sample count never becomes what fails first. The one exception measured is
Boltz-2 at 1536 tokens, where the width-5 sampling chunk (8.08 GiB) overtakes the trunk
(7.21 GiB). It still fits.

**Time** is linear in the sample count: `T(S) = a + b * S`. The table gives `b` as a fraction of a
one-sample fold at production sampling steps.

| model | tokens | extra cost per sample |
|---|---|---|
| Boltz-2 | 256 / 768 | 0.41 / 0.18 |
| ESMFold2 | 256 / 768 | 0.12 / 0.15 |
| Protenix-v2 | 256 / 768 | 0.21 / 0.16 |
| OpenDDE | 256 / 512 | 0.17 / 0.17 |
| RF3 | 256 / 768 | 0.07 / 0.07 |
| OpenFold3 | 256 / 768 | 1.3 / 0.34 |

So 25 samples of Boltz-2 at 768 tokens take about 4.5 times as long as one sample, not 25 times.
OpenFold3 and OpenBind are the expensive ones because they denoise one sample at a time.

**Host memory** also grows with samples, because every sample's structure and confidence
outputs come back to the host. RF3 at 1088 tokens and 100 samples peaks at 12.4 GiB of host RSS.

**Largest measured points that fold**, at 25 samples unless noted: Boltz-2 1536, Protenix-v1
1536, Protenix-v2 1024, ESMFold2 1024, ESMFold2-fast 1152, OpenFold3 1024, OpenBind 960, OpenDDE
512, OpenDDE-abag 1024, RF3 1088 at 100 samples. Each of these is the model's token ceiling or
the largest rung walked; none of them is limited by the sample count.

## Recycles

Recycling re-runs the trunk on the same buffers, so device memory does not change with
`--recycling_steps` (Boltz-2 at 768 tokens peaks at 3.39 GiB at 3 and at 20). Time grows by one
trunk pass per cycle, about 14 s per cycle at 768 tokens for Boltz-2, ESMFold2 and Protenix-v2
on this box.

Boltz-2, ESMFold2, ESMFold2-fast, OpenFold3 and OpenBind run one more trunk cycle than asked
(20 recycles are 21 cycles) and accept 0. Protenix, OpenDDE and RF3 count cycles, so 20 is 20
and the smallest value they accept is 1.
