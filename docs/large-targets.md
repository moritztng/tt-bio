# Large targets on Wormhole

The pair representation every AlphaFold3-family model carries is an N×N tensor, so its memory
grows with the square of the token count. On a 12 GiB Wormhole chip the naive implementation
crosses the ceiling at roughly 850-1100 residues, depending on the model's pair channel width.
OpenDDE is the strictest case: its structural-token expander runs the refiner on about 1.9x the
residue count, and its pair tensor is 3.7x the residue-scale one.

The pair-track ops (triangle multiplication, triangle attention, the pair transition) are all
row-local along the token axis, so past a size threshold they run in row blocks and free every
intermediate that is not the input or the output. The peak then scales as three live pair
tensors instead of four. The MSA features of very deep MSAs are streamed from the host between
recycling cycles instead of staying resident. ESMFold2's pair initialisation is row-tiled the
same way.

Measured on the WH Galaxy (12 GiB chips) on the four targets the AbAg-XM campaign had to
exclude:

| target | residues | boltz2 | esmfold2 | protenix-v2 | opendde-abag |
|---|---|---|---|---|---|
| 9q7y | 853 | OK | OK | OK | OK |
| 9ivj | 891 | OK | OK | OK | OK |
| 9i3p | 980 | OK | OK | OK | OK |
| 9j4c | 1095 | OK | OK | OK | OK |

The last cell to fall, OpenDDE on 9j4c, was a lifetime bug rather than capacity: the structural
pair tensor and the sampler's pair bias stayed resident through the residue-axis confidence
stage that never reads them, so the confidence pair track started with under 2 GiB free. Both
are freed at the diffusion boundary now, which drops the confidence entry from 10.04 to
5.72 GiB.

Per-cell fold times and peak DRAM are in the release notes of the version that landed this.
Normal-size targets never enter the blocked path: it is gated on a token-count threshold
(1536 on Blackhole, 608 on the Wormhole Galaxy) that a 300-residue target does not reach, and the
before/after timings on the standard 117/298-residue benchmarks are unchanged within noise.
