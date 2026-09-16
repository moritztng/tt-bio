# FLOP accounting contract

This CPU audit distinguishes exact matrix arithmetic from symbolic operations and missing evidence. The existing counter fails known-answer cases, so an exact whole-table claim is refused. No device is opened, and no timing, clock, instruction count or cycle saving is measured.

From the repository root, with standard-library Python:

```bash
python3 -m unittest discover -s perf/c10_flop_contract -p 'test_*.py' -v
python3 perf/c10_flop_contract/audit.py --out /tmp/flop-audit.json
python3 perf/c10_flop_contract/audit.py --summary --out /tmp/flop-coverage.json
python3 perf/c10_flop_contract/audit.py --require-exact-total --out /tmp/flop-audit.json
```

The tests pass when both valid arithmetic and rejection controls behave correctly. The last command exits **2** because exactness is unresolved. An ordinary audit exits 0 when it successfully writes the report; inspect its `verdict`, which is **STOP** for the counter defects below. Invalid captures exit 1. Positional paths select particular captures. There is no fold-total mode: separately captured parents and children overlap.

[coverage.json](coverage.json) is the reproducible per-class report without individual operation rows. The full report includes capture hashes, operation counters, ordered matrix shapes, legacy results and unresolved evidence for every row. [provenance.json](provenance.json) identifies the sources and copied dense control. Production consumers are unchanged.

## What is exact

For a dense contraction, one multiply-accumulate counts as two FLOPs, including the initial accumulation. With broadcast batch size B, the matrix component is exactly `2*B*M*N*K`. This convention is independent of precision, math fidelity, accumulator format and the number of machine instructions used to implement it.

The audit uses the ordered `MatmulDeviceOperation.input_tensors`, checks output and broadcast shapes, and accepts only a unique count across compatible transpose interpretations. It does not guess K from the largest candidate operand. Rank-one/vector operations, incompatible dimensions, missing roles and ambiguous counts remain unsupported.

Three quantities must stay separate:

| Quantity | Meaning |
|---|---|
| `shape_flops` | Exact arithmetic for captured tensor shapes, including any model-side bucketing already applied. It is not the original unpadded model work. |
| `tile32_flops` | Exact arithmetic under the explicit convention of rounding effective M, N and K to multiples of 32. This is conditional shape accounting, not proof of physically issued work. |
| `issued_flops`, `machine_instructions`, `cycles` | Unknown (`null`). Physical padded shapes, selected program and dynamic execution evidence are needed. |

`mm_generic.build` rounds M and N tile counts to core-grid dimensions and K to its block size. The trimul compute loops use those ranges and run two projection passes. These can introduce work beyond a simple 32-element rounding. Captured tensor shapes alone cannot certify the issued count.

`known_fused_matrix` checks the matrix formulas for source-identified `mm_generic`, triangle QKV/split projections and the two-projection trimul tail. Split destinations retain the full weight width; head-major storage requires an explicit effective matrix shape. `dense_attention_matrix` counts QKᵀ and PV separately, including rectangular attention and different value widths. These are formula controls, not inferred identities for historical `generic_op` nodes. Kernel eligibility and actual runtime descriptors remain separate evidence.

## Coverage

CPU replay covers **26 historical captures plus two dense controls**, with 7,978 captured operation occurrences across 30 labels. These occurrences overlap across files and are not a fold census. Of them, 2,034 have exact matrix-component counts, 3,797 are excluded from floating add/multiply arithmetic by an explicit movement/metadata convention, 2,091 are symbolic, and 56 generic calls are uncounted. **14 of the 26 historical captures have unbalanced spans.** Their within-capture subtotals describe candidate spans only; ownership is not certified.

| Captured label | Occurrences | Contract |
|---|---:|---|
| `linear` | 1,998 | Exact matrix component; bias/activation and program work unresolved |
| `matmul` | 36 | Exact matrix component; includes two dense controls |
| `add` | 193 | Symbolic, fusion attributes needed |
| `add_` | 220 | Symbolic, output alias and fusion attributes needed |
| `multiply` | 184 | Symbolic, fusion attributes needed |
| `multiply_` | 759 | Symbolic, output alias and fusion attributes needed |
| `layer_norm` | 638 | Symbolic reductions, variance, rsqrt and affine work |
| `softmax` | 4 | Symbolic reductions, exp and normalization |
| `cos` | 2 | Symbolic transcendental evaluations |
| `transformer.scaled_dot_product_attention` | 91 | Matrix roles/schedule and softmax unresolved |
| `generic_op` | 56 | Uncounted; identity cannot be recovered from rank alone |
| `deallocate` | 2,178 | Excluded movement/metadata arithmetic |
| `reshape` | 286 | Excluded movement/metadata arithmetic |
| `to_memory_config` | 248 | Excluded movement/metadata arithmetic |
| `Tensor.__getitem__` | 242 | Excluded movement/metadata arithmetic |
| `permute` | 218 | Excluded movement/metadata arithmetic |
| `slice` | 183 | Excluded movement/metadata arithmetic |
| `allocate_tensor_on_device` | 96 | Excluded movement/metadata arithmetic |
| `experimental.nlp_create_qkv_heads` | 93 | Excluded movement/metadata arithmetic |
| `unsqueeze` | 91 | Excluded movement/metadata arithmetic |
| `pad` | 51 | Excluded movement/metadata arithmetic |
| `to_layout` | 42 | Excluded movement/metadata arithmetic |
| `concat` | 23 | Excluded movement/metadata arithmetic |
| `experimental.nlp_concat_heads` | 17 | Excluded movement/metadata arithmetic |
| `squeeze` | 17 | Excluded movement/metadata arithmetic |
| `chunk` | 6 | Excluded movement/metadata arithmetic |
| `from_torch` | 2 | Excluded transfer/conversion arithmetic |
| `transpose` | 2 | Excluded movement/metadata arithmetic |
| `to_torch` | 1 | Excluded transfer/conversion arithmetic |
| `from_device` | 1 | Excluded movement/metadata arithmetic |

“Excluded” never means zero instructions or zero traffic. No coefficients convert exp, sigmoid, sqrt, rsqrt or cosine into FLOPs. Output element counts are labeled as elements. Missing output creation for in-place operations does not imply zero work. Every unresolved numeric total is `null`.

## Reproduced counter defects

Both stored 8192³ matmuls return **1,099,511,627,776** matrix FLOPs, agreeing with `2*8192^3`. That success does not certify the remaining operation classes.

| Case | Exact shape arithmetic | Existing counter |
|---|---:|---:|
| Captured `[1,4480,768]ᵀ @ [1,4480,512]` | 3,523,215,360 | 0, `matmul-unresolved` |
| Captured `[1,1] @ [1,256]` | 512 | 131,072 |
| Synthetic `[32,32] @ [32,64]` | 131,072 | 262,144 |
| Synthetic broadcast `[1,3,5] @ [7,5,11]` | 2,310 | 0, `matmul-unresolved` |
| Known generic `[32,32] @ [32,32]` with rank-two output buffer | 65,536 | 196,608 |
| Known noncausal generic attention, B=1, H=2, S=32, D=64 | 524,288 matrix FLOPs | 0, `layout` |

The first two occur at counters **2464 and 2510** in `cap_DiffusionModule__.json.gz`, and at **2426 and 2473** in its overlapping `cap_Diffusion__1x4480x3,1.json.gz` child. Do not add the parent and child discrepancies. Their reproductions are in `test_real_missing_transposed_and_scalar_matmul`; the other cases are emitted in the report's `legacy_controls`.

The matrix heuristic confuses an RHS with the activation whenever its leading volume matches the output rows. Generic inputs also include preallocated outputs, which the rank-two heuristic can count as extra weights. The B=1 attention heuristic cannot identify Q/K/V because it explicitly requires the first dimension to differ from one.

Other gaps have no justified scalar correction: all 979 captured in-place add/multiply calls receive legacy `noshape` zero; native SDPA receives one output-element count instead of its two matrix contractions and softmax work. Of 56 generic calls, the legacy counter labels 32 as matmul and 24 as layout. Neither assignment proves the kernel identity. The roof-table consumer adds its one-per-element estimates to matrix FLOPs despite their different meanings.

## Source boundary and device-owner handoff

The historical graphs identify revision `f072ae02f`; the audited production source is `71a306a8a`. Source hashes are in the report. Old capture coverage does not certify today's defaults. Current source enables head-major QKV/tail and QKVGB, dual-NOC input projections, persistent-mask SDPA and gated reblock. The SDPA gate epilogue and fused-QKV SDPA are off by default. Trimul F1 is enabled but normally accepts only the `(8,8)` key; `(4,4)` requires the explicit CZ128 opt-in. The MSA axis now uses a 64/128/256/512/1024 ladder by default, so the old 1024-row capture must not be substituted for a current-default census.

Plain reblock runs tile transposes, while gated reblock also runs sigmoid and multiply. Both appear as `generic_op`. SDPA can carry projections or a gate under compile-time options. No shape-only classifier can establish these distinctions. SFPU instruction costs additionally depend on the compiled lowering and selected mode.

Minimal changes proposed for the device owner, not applied here:

1. Replace the K heuristic with ordered operands, explicit transpose flags and broadcast/output validation. Preserve logical input shapes, tensor padded shapes and program iteration extents as separate fields. Reuse these negative controls.
2. Record a stable fused-kernel identity, source/build hash, operand roles, output aliases, defines and compile/runtime arguments. Preserve projection multiplicity and head-major reinterpretations. Do not recognize kernels by rank or treat unknown ones as layout.
3. Split matrix FLOPs, add/multiply conventions, symbolic transcendental/reduction evaluations and uncounted operations. Include native SDPA's two contractions when roles and dense/causal/chunk behavior are known; never hide its softmax work in that count. Refuse exact totals whenever a component is unresolved.
4. Capture complete named function spans and a disjoint fold partition with multiplicities. The parser's range fallback is useful diagnostically but cannot certify nested ownership. Do not sum inclusive parent/child reports or rescale them into a fold budget.
5. For issued-arithmetic or instruction claims, retain selected program descriptors, padded tile iterations, skipped/causal blocks and compiled-kernel provenance. A machine instruction census is a separate instrument, not a FLOP conversion.

Fresh captures, timing and pinning belong to the device owner. Any future performance claim requires AICLK sampled during the fold with min/max and coverage, a benchlock, warm interleaved arms in one process and calibrated instruments. Requested 1350 MHz is not a measurement; a loaded chip below 1200 MHz is a clock artifact. This audit measures CPU counts only, with cycles unmeasured.
