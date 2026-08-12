# ESMFold2 512 aa, second perf round

Two harnesses, both card-2 qb2, benchlock held, ttnn 0.68.0.

`trimul_ops.py` times every op of `TriangleMultiplication` separately at the production shape
(L=512, c_z=256, chunk 32, group 8) and checks the nine timings against the whole-op wall, so the
transcription is verified rather than asserted: 15.219 ms against 14.755, ratio 1.0315. It also
prices the structural alternatives in the same session (`transpose_b` on the channel matmul, an
L1-resident row-blocked in-projection, a fused gated residual, one fused out projection).

`decomp2.py` is `../esm512/decomp.py` plus timers on host `nn.Linear` / `nn.LayerNorm` /
`nn.Embedding` / `F.linear`, on the vendor `ESMFold2Model.forward` and `ConfidenceHead.forward`, on
every `TorchWrapper` / `_Adapter` boundary and on `_from_torch` / `_to_torch`. It closes the 5.747 s
the previous round could not attribute down to 0.05 s, and it shows that remainder is host/device
transfer plus host torch, not device compute.

Results: `trimul_ops_512_c2.json`, `decomp2_512_c2.json`. Analysis and the build order live in
`~/.coworker/state/esmfold2-to-4x.md`.
