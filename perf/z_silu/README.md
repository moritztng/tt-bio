# z-silu-lowering-fix — probe harnesses

Two throwaway probes that settled the deployability of a patched ttnn silu kernel. Plan, measurements
and the execution steps live in `~/.coworker/state/protenix-trunk--z-silu-lowering-fix.md`.

`jit_probe.py` — one fused `activation="silu"` matmul at production compute-kernel config. Used with
an `#error` injected into a private copy of the wheel's `ckernel_sfpu_silu.h` to prove the kernel is
JIT-compiled at runtime rather than baked into `_ttnn.so`.

`which_probe.py` — runs the fused and the bare matmul over the same inputs and reports whether they
are `torch.equal`. With the silu kernel patched to the identity they become equal, which is how the
fused path's silu was pinned to `metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` rather than to the
`tt_llk` copy of the same filename.

Both need a private runtime root so the shared install at `/home/ttuser/tt-bio/env` is never touched:

    WT=/home/ttuser/.coworker/wt/protenix-trunk--z-silu-lowering-fix
    Z=$WT/perf/z_silu; H=$Z/pkg/ttnn        # rsync -a <site-packages>/ttnn/ $H/
    TT_VISIBLE_DEVICES=2 TT_METAL_RUNTIME_ROOT=$H TT_METAL_HOME=$H TT_METAL_CACHE=$Z/kcache_pkg \
    TT_MESH_GRAPH_DESC_PATH=$H/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    /home/ttuser/tt-bio/env/bin/python which_probe.py

`TT_METAL_HOME` on its own is not enough: it is the data root, and the includes still resolve to the
installed wheel. `TT_METAL_RUNTIME_ROOT` is the one that moves the kernel sources.
