# Why the r = 0 rebuild had to be run four times

Four runs, two questions. PROTOCOL A13 asks whether the artifact is *reproduced* rather than
merely measured: tape it twice in two fresh processes and compare per tensor, not by file hash.

`nondet/` — runs A and B with dropout already in eval, kernels left as they come.
**Not reproduced**: median 2.631e-14 but worst **1.985** relative L2 over **55 of 4,147** tensors,
9 bit-identical. Every one of the 55 is a `layer_norm_z.bias`. Their gradient is a sum over tokens
that very nearly cancels, so a change of reduction order worth 1e-16 *absolute* reads O(1)
*relative*. The loss itself was not bit-identical on replay in run B (1.6073630623808797 against
1.60736306238088), which is the same cause seen at the top of the graph.

`det_selfdrawn/` — the same pair with `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. **Reproduced**: 4,147 of 4,147 bit-identical, worst 0.0,
median 0.0. Loss bit-identical on replay. Finite differences at h = 1e-4 over 16 stratified
entries: worst 5.669e-03, median 1.283e-04, against PROTOCOL 3d's 5.0e-02 bar.

Neither pair is the published artifact. Both let the step draw its own randomness, and that is
not the randomness any consumer holds: putting the Pairformer's Dropout in eval removes 61
consumers of the CUDA generator, so the next `torch.randn` in the diffusion head returns
something else. Measured, with nothing else changed, the diffusion noise levels go from
`[23.705, 5.853, 10.730, ...]` to `[2.612, 1.467, 5.936, ...]`, the loss from 1.6591175475821072
to 1.60736306238088 and the gradient global norm from 3.9083 to 4.3050. PROTOCOL 4a makes those
draws inputs to the update rule, so the published run replays `draws_recycles0.pt` instead
(`rebuild_r0_replay.sh`, runs C and D).
