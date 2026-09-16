# Generic kernel work components

This CPU-only tool counts source-conditional work for identified generic wrappers. It does not measure hardware, certify an executed binary, or produce a model roofline.

Run from the repository root with Python 3.10 or later. Only the standard library is used; neither PyTorch nor TTNN is imported.

```sh
python3 -m unittest discover -s perf/c10_generic_work -p 'test_*.py' -v
python3 -m perf.c10_generic_work.controls --out /tmp/c10-work-controls
python3 perf/c10_generic_work/reduce.py \
  --contract /tmp/c10-work-controls/matmul_dense.json --out /tmp/matmul-work.json
python3 perf/c10_generic_work/reduce.py \
  --observer perf/c10_generic_work/evidence/reblock_on.jsonl \
  --sequence 1 --out /tmp/observed-refusal.json
```

Choose a new controls directory. The final command exits **2**, because the archived call lacks per-core runtime arguments and storage identity. Adding `--require-exact-total` to any invocation also exits **2**: every supported contract leaves total issued work unknown. An ordinary successful normalized reduction exits 0 and reports `SOURCE_CONDITIONAL`, which is not a measurement verdict.

[report.json](report.json) contains 15 synthetic controls and two archived-call refusals. To reproduce it, generate controls as above and compare their `report.json` with this one. Each synthetic control carries its full normalized descriptor and hash in the generated directory. [contracts.json](contracts.json) pins wrapper, kernel and local helper bytes. The upstream matrix arithmetic is reused from [c10_flop_contract](../c10_flop_contract/README.md), unchanged.

## Supported contracts

| Family | Logical component | Supported boundary |
|---|---|---|
| `matmul` | One flattened-left `A @ B`, 2 FLOPs/FMA | bf16, tile aligned, ordinary output, unbatched right operand, no transpose, bias, activation, ternary, split or head-major layout; 1×1 compute subblocks |
| `reblock` | `[1,N,N,C] → [1,C,N,N]` | bf16; C multiple of 32; ragged N allowed |
| `reblock_back` | Inverse permutation | bf16; N and C multiples of 32 |
| `reblock_gated` | Projection slice × sigmoid(gate slice), then permutation | Explicit disjoint channel slices; whole tensor or row block; aligned offsets; a ragged row block must end at the logical tail |
| `sdpa` | `QKᵀ` and `PV` | Audited triangle-attention kernel sources, noncausal, one phase, same Q/K/V batch and heads, equal Q/K/V head widths, tile aligned, chunk sizes divide sequences, explicit additive mask |
| `sdpa_gated` | SDPA plus output × sigmoid(gate) | DH=32, one Q chunk, explicit gate matching output |
| `sdpa_fused_qkv` | One QKV projection plus SDPA | Explicit internal QKV shape, square attention, DH=32, one Q chunk/head per worker, enough KV buffering; cannot combine with gated SDPA |

All operands are bf16 TILE and must have declared distinct storage. Mask broadcasting over batch and heads is supported. Persistent-mask contracts require batch-broadcast masks and one head/Q chunk per worker. These are arithmetic contracts, not memory-fit or eligibility checks. Small controls may bypass production eligibility windows; they do not assert that a model dispatches those shapes.

The matmul wrapper flattens the left prefix; it does **not** perform a broadcasted batch matmul or a tensor transpose. Transpose and broadcast shape arithmetic is tested through the shared counter, and unsupported wrapper uses are refused. Stock SDPA sources, ragged SDPA, causal/windowed/streaming attention, GQA, head-major/split matmuls, QKVGB and other projection variants remain unsupported. An unknown family is never treated as zero work. Fused-QKV and gated SDPA coverage does not assert that either optional path is enabled in a model.

## Input and identity

A normalized input has four fields:

```text
schema: 1
basis: "normalized_source_contract"
descriptor: {family, function, sources, operands, compile, extra_defines,
             ownership, assumptions}
descriptor_sha256: SHA256(canonical JSON of descriptor)
```

Canonical JSON uses sorted keys, comma/colon separators and no NaN. `controls.py` produces complete examples. `sources` must exactly equal the selected manifest entry, and every file is rehashed. The wrapper must also exist in the original observer's [role registry](../c10_generic_identity/roles.json). The file/function match applies to a conditional source contract; it is not a check of a live Python stack.

Each operand declares `role`, logical `shape`, `padded_shape`, `dtype`, `layout` and `storage`. Roles are ordered exactly as the wrapper passes them. Storage tokens assert non-overlapping allocations; reusing a token refuses the contract. Gated reblock counts only its selected input slices and written output rows, not the entire projection or preallocated output.

`compile` contains the complete set of normalized flags shown in the generated example for that family. These are reviewed semantic fields, **not** a native `ProgramDescriptor`. Unknown fields, missing fields, unsupported flags, extra defines and inconsistent shapes refuse. In particular, no ablation define is admitted. The required `assumptions` string is:

```text
listed source; no injected defines; API lowering and cached binary unverified
```

Ownership is `{id: <nonempty identifier>, scope: "one_generic_call", cover: "full_disjoint_logical_domain"}`. It asserts exactly one complete call, with the wrapper's disjoint work assignment. A normalized descriptor hash covers these assertions, the source map, roles, shapes and flags. It is **not** an observer execution hash, native cache hash, or binary hash. Normalized inputs are hypothetical until independently joined to a real call. The tool deliberately provides no aggregation of calls, parent spans or model captures.

For an observed call, `--observer` selects one JSONL archive and a sequence number. The tool verifies the archive framing, selected execution hash, embedded source hashes, wrapper-role evidence and return outcome, then reports missing inputs. It does not automatically normalize live descriptors. The archived smoke calls identify matmul/reblock sources and roles, but all their per-core runtime getters failed; no missing fields are completed from shapes or wrapper defaults. The report retains the archive, execution and descriptor hashes. Before model-table use, a separate reviewed join must establish actual runtime ownership, compile/define meanings, aliases and output regions for that execution. Cached shape labels and inclusive parent captures cannot supply that join.

## Output conventions

The report keeps five categories separate:

1. `logical_matrix_flops`: named primary contractions at **2/FMA**, including the initial accumulation add. For a proven permutation this is zero; for a refused generic call it is `null`.
2. `conventional`: scalar multiplies/adds, reduction additions/comparisons, and named exp/reciprocal/sigmoid evaluations. No FLOP coefficient is assigned to an elementary function. SDPA entries describe the dense real-arithmetic reference decomposition, not its online implementation. In particular, source code fuses scale into exp and normalizes an accumulated output instead of materializing normalized probabilities.
3. `source_schedule`: conditional loop counts. Matmul reports padded matrix extents and `matmul_block` calls with 1×1 subblocks. Its padded-domain FLOPs are a matrix convention, not issued arithmetic. Reblock reports tile transpose, sigmoid and multiply API calls, including padded lane slots. SDPA reports primary QK/PV chunk pairs only. Its `matmul_reduce` reduction, online rescaling and SFPU work stay unknown.
4. `issued_arithmetic` and `sfpu_flops`: always `null`. Source API calls do not equal executed instructions. Transitive TT APIs, compiler flags, lowering and cached code are not established by these file hashes.
5. `compulsory_logical_bytes`: unique required logical bf16 inputs plus the written logical output region, once per call under the distinct-storage assertion. Broadcast inputs count once. Virtual QKV intermediates in fused SDPA are not charged as input traffic. `physical_bytes` remains `null`: residency, rereads, spills, padding, transactions and memory-level traffic are unmeasured.

For example, the synthetic 32×32 matmul has 65,536 logical matrix FLOPs and 6,144 compulsory logical bytes. Its specified 2×2 grid and K block of two tiles expand the source matrix domain to 64×64×64, with eight `matmul_block` API calls. These are integer accounting controls, with no timing or performance implication.

## Evidence and limits

The two JSONL files are byte-for-byte archives from `5d8b761981dde4f0e3fbbf6fc13a5037bc252663`; [evidence/provenance.json](evidence/provenance.json) records their paths and hashes. Their embedded minimal-matmul source bytes supply the three snapshots under `sources/`. The SDPA writer snapshot was read without importing TTNN from the named k10 source checkout on `tt-quietbox2`. Source revision `1452925b033c6608726b731a81500bd3e19f7894` is context, not proof of the loaded or cached machine code. Local wrapper/kernel source is pinned against `71a306a8a`.

The archived instrument campaign targeted **1350 MHz**. This report uses source/identity fields only; it makes no hardware timing, utilization, roof, model-cycle-saving or fold-speedup claim. Historical `perf/roof_quiet/out_shipped_trimmed` rankings only selected which families to examine; their timings lack verification here.

Tests cover dense/rectangular/flattened matrix counts, transpose and batch-broadcast rejection, ragged and reverse permutations, gated slices and row tails, rectangular attention, mask broadcasting, both attention fusions, tampered identity, unsupported flags, aliases and exact-total refusal. Independent integer enumerations and float64 reference identities establish arithmetic only. They do not establish device accuracy or binary execution. No model math, defaults, dependencies, steps, recycles or production code change.

Parent review and an execution join are required before these components enter a model table. This CPU result does not complete Phase0. A later optimization still needs paired 512/298 structural accuracy, A/A controls and independent float64 transform references; the 512-residue bar is 0.60 Å beside its 1.84 Å seed floor.
