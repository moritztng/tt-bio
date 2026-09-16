# CPU prerequisite review — export budget and cycle stream

Both rows were accepted by the parent on 2026-09-16, on CPU, with no device opened.

**`c10-export-budget` @ `640d893ec`.** The parent's own replay reproduces the child's published
`census_windows.json` and `packaging.json` exactly once the scratch directory name is normalised:
259,634 distinct raw call ids, 1,514 invocations, 13 windows, a 2,970,301-byte metadata archive over
26,969,586 original bytes, with every embedded input and output hash equal. Only the manifest's own
bytes differ, because it records its own absolute paths. All four negative controls (oversized
input, file cap, address-space cap, elapsed timeout) return STOP.

Replay scratch (`replay/`, `tools/`) is gitignored and was deleted after review; pc was at 94 % of
its root filesystem. The child's code and results are on its branch. The 2.9 GB census trace that
motivated the row was never tested and is gone — the worker's worktree was reclaimed when it
concluded, and only what it wrote under `artifacts/` or committed survived.

**`c10-cycle-stream` @ `c26c9ee75`.** Its 20 tests pass when rerun from the branch in a temp tree,
and all 16 published chunks in `~/.coworker/artifacts/c10-cycle-stream/verified` verify against their
hashed footers.

Neither row measured model cycles, and neither changed production code. The archives were captured
at a during-window sampled 1350 MHz; this review adds no new clock or timing evidence.

Record: `review.json`.
