# The same clash on Boltz-2, at a size the shipped code accepts

BoltzGen said the cliff is a property of the gate and the op, not of the model. Boltz-2 is the
test of that, because `b2z2-atoml1-size-curve` measured it clean and bit-exact to 1024 aa and
declared `TT_BIO_ATOM_L1` a GO. Its ladder stops exactly one bucket below the cliff.

Wormhole, whglx, cards 1 / 3 / 4, `perf/size512/fixtures/cdk2x2_{1024,1152}.yaml` with their own
a3m named in the yaml, shipped protocol (3 recycles, 200 sampling steps):

| fold | arm | result |
|---|---|---|
| 1024 aa | `TT_BIO_ATOM_L1=1` | ok, 214.6 s — reproduces the size-curve's GO on this box |
| 1152 aa | base (shipped default) | ok, 316.6 s |
| 1152 aa | `TT_BIO_ATOM_L1=1` | **FAILS after 264.8 s** |

    Statically allocated circular buffers in program 770 clash with L1 buffers on core range
    [(x=0,y=0) - (x=7,y=7)]. L1 buffer allocated at 1026048 and static circular buffer region
    ends at 1041696

Byte-identical to both BoltzGen crashes — same addresses, same core range, three different
program ids (770, 791, 999) across two models and three targets.

`tt_bio/size_limits.py` publishes **no ceiling for boltz2 on wormhole_b0 and never refuses**, and
its own note records 1024 / 1088 / 1152 / 1300 / 1408 / 1536 / 1664 aa all folding. 1152 aa is a
size a user gets today.
