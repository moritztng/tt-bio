# The GPU roof mirror

The b2z campaign rests on one assumption: that a Boltz-2 pairformer block ought to reach its
roof, and that missing it by 2.34x is a Tenstorrent software defect. This directory is the
control. It runs the same fold, the same three units and the same roof arithmetic on an NVIDIA
H100, against the H100's own measured roofs.

    setup.sh          install torch + tt-bio + the boltz2 checkpoint on a rented box
    gpu_roofs.py      the H100's own op-cost curve, HBM plateau and matmul roof, all measured
    gpu_decomp.py     the fold, then PairformerLayer / MSALayer / one diffusion step decomposed
    mirror.py         one estimator applied to both cards' curves, side by side

Result: the H100 misses its own two-parameter model by **2.332x** on the pairformer block where
Blackhole misses by **2.336x**. Working and consequences in
`~/.coworker/state/b2z-gpu-roof-mirror.md`.

This is a screen. No number in here is or can become a Tenstorrent performance claim.
