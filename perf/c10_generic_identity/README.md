# Generic program identity capture

Record the program descriptors passed to `ttnn.generic_op` without reading tensors or changing the call. This opt-in measurement tool helps identify work that a graph labels only `generic_op`. It does not count FLOPs, bytes, instructions or cycles.

Run the CPU controls from the repository root, with a new output directory:

```bash
python3 -m perf.c10_generic_identity.cpu_controls --out /tmp/c10-identity-controls
```

Use the context manager in an existing measurement driver:

```python
import ttnn
from perf.c10_generic_identity.observer import Capture

with Capture(
    ttnn,
    "/tmp/generic-identity.jsonl",  # must not exist
    provenance_files=[
        "/path/to/tt-metal/build_Release/CMakeCache.txt",
        "/path/to/tt-metal/build_Release/lib/libtt_metal.so",
    ],
    config={"fixture": "your fixture", "steps": steps, "recycles": recycles},
) as capture:
    result = existing_measurement_driver()
```

The observer imports no TTNN module and opens no device. The example driver owns its usual device lifecycle. Installation changes only the supplied module's `generic_op` attribute, for the duration of the context. Use one dispatch thread. Nested contexts each record a call once and restore the previous observer on exit. Do not combine this with another tool that replaces the same attribute inside the context.

Each call receives a sequence number within a unique capture session. JSONL contains a header, deduplicated source payloads, call snapshots, outcomes and a footer. A valid file ends with a footer. `observation_ok` reports whether the observer ran successfully; `identity_complete` remains false because compiled-code provenance and replay coverage are incomplete. Every call lists its `unavailable_fields`. An empty captured sequence is not evidence of an empty model.

The snapshot includes:

- Ordered operands with logical/padded shapes, dtype, layout and repeated Python-object aliases. Storage aliases and buffer addresses are not queried.
- Kernel source strings, absolute source paths and SHA-256 hashes of observed bytes. Source payloads use base64, including inline source. Relative paths remain unresolved rather than assuming the compiler's search order.
- Core ranges, compile arguments, named compile arguments, defines, exposed compute/data-movement config, CB formats and semaphore descriptors.
- Per-core runtime values and lengths, plus common runtime values, freshly read on every call. Positions are preserved; argument meanings remain unknown unless separately audited. The audited coordinate-only view has no storage-order getter. Native-only view/vector classes are resolved through `ttnn._ttnn.program_descriptor`. The observer enumerates declared core ranges and checks against the binding's entry count. Duplicate or outside-range entries produce an explicit gap.
- A binding cache hash when available, a per-call snapshot hash and a descriptor object address. The cache hash is not an execution identity: it excludes runtime values and hashes source paths rather than file contents in the audited version. Object addresses may be recycled after destruction.
- The loaded caller's code hash, its observed source bytes, selected environment settings, Python/binding identity and hashes of supplied provenance files. Supply relevant build/config files and explicit driver settings. Git revision, dirty source and binary identity are separate evidence.

Operand roles are **declared wrapper roles**, not proof of kernel reads or writes. They are assigned only when both the file hash and loaded code match an audited wrapper and its local operands match the dispatched objects by identity. The registry covers `mm_generic.generic_minimal_matmul`, `sdpa_generic.sdpa`, and the forward, reverse and gated reblock wrappers at the recorded revision. All other callers remain unknown, regardless of tensor rank. SDPA's gated wrapper appends its gate after the output, so the last operand is not generally the mathematical output. Fused-QKV and gated SDPA have different operand lists.

Sources are reread and rehashed on every call. Payload deduplication uses the content hash, never a cached path or timestamp. This detects same-size edits with restored mtime. An observed source file can still differ from an already-compiled cached program; the observer does not certify which machine instructions ran. Files must be regular files, and embedded sources are limited to 16 MiB. Missing files, opaque types, unavailable getters and unstable reads are explicit gaps. Arbitrary objects are never serialized with `repr`.

Observation does CPU metadata traversal, file reads, hashing, base64 encoding and flushed JSON writes before dispatch. Runtime and artifact cost can be substantial and are unmeasured. Do not report captured folds as bare-fold performance measurements. There are no explicit device synchronizations, tensor readbacks, device queries or tensor references stored in the capture. The audited source reads stored tensor specifications for device metadata, without synchronization. Getter behavior on the installed binary still needs validation.

Observer failures are printed to stderr and recorded where the sink remains writable. The dispatched call still receives the original argument objects exactly once and returns its original object or raises its original exception. After restoring the original callable, a clean context exit raises `CaptureError` if observation failed. If the body already raised, its original exception wins and `capture.errors` carries the observer failure. A failed sink may prevent the footer from being written; reject that file. Output files are created exclusively and never overwritten.

The audited k10 source revision is `1452925b033c6608726b731a81500bd3e19f7894`, with tracked local changes recorded in [source_audit.json](source_audit.json). Its Python binding does not expose `KernelDescriptor.opt_level` or `ProgramDescriptor.custom_program_hash`, although both exist in C++. These fields remain unavailable. CB backing addresses, implicit Reader/Writer defaults, transitive includes, loaded ELF dependencies, effective JIT flags and source-to-build linkage are not resolved automatically. `MeshProgramDescriptor`, device trace replays and cached aliases imported before patching are outside this observer's coverage. Outcomes now record the native host operation counter when its getter is available. This is not itself a profiler ID; the census reducer validates the deployment's encoding, dispatch count and graph join.

The committed [CPU report](controls/report.json) and [synthetic capture](controls/synthetic.jsonl) test mechanics only. Installed-binding construction was not run: read-only `bwrap` isolation on qb2 failed before Python at namespace setup. No unisolated import or device access followed.

The device worker must first validate the installed binding's getters and runtime view against constructed descriptors. Then run a small paired live call with the observer off/on, checking exact dispatch count, shapes, output accuracy and changed buffer arguments on a cached descriptor, and inspect missing fields. Verify source/build provenance on that worker's actual deployment and establish a join to the graph/profiler sequence. This does not complete a fold census or Phase0. Future timings require a calibrated bare/profiler comparison and AICLK min/max and coverage sampled during each fold; a 1350 MHz request alone is not a measurement.

The burst census validated this installed binding with live matmul and reblock controls: use `CoreRange.start/end`, the exported Python names, rather than C++ member names. See [the live evidence](../c10_burst_census/runs/smoke2/analysis.json). These controls certify the recorded source/runtime inputs and address rebinding, not cached machine-code provenance.
