Neighbour perf-gate legs for the 64 -> 32 token-bucket change, card 0, tt-quietbox2 p300c.

    protenix-v2     2.634 -> 3.03    +15.0%  PASS
    opendde         2.297 -> 2.611   +13.7%  PASS
    opendde-abag    2.276 -> 2.693   +18.3%  PASS
    openfold3       2.142 -> 1.989    -7.2%  PASS

GATE PASS. Boltz-2 scored separately (+15.9% PASS); RF3 has no p300c baseline and was A/B'd
directly instead, and came back perf-neutral.

Reading of the signs is in state/token-axis-bucketing-unify.md section 17. Short version: the three
improvements are the narrower pad paying off at the gate's 20-token fixture, and OpenFold3's loss is
a bucket being ADDED at a size where 20 pads to 32, the same single tile, so the pad/mask/slice buy
back no compute.
