#!/usr/bin/env python3
"""Minimal reproduction of the extra-MSA backward crash, with no model and no BindCraft 2.

`predictor(extra_msa=True)` dies in the backward with "Buffer is not allocated". Reaching that
through a BC2 round costs ~10 minutes a try, because round 1 is the JAX compile. The mechanism
does not need any of that: it is `autograd.checkpoint` re-running a callable that CONSUMES a
tensor it closed over.

`_recompute` does `y = fn(*inner)` -- it duplicates the declared `inputs` but reuses everything
`fn` captured, and its docstring says captured parameters "are LEAVES" and need no duplication.
That holds only while the forward leaves them allocated. `af2._residual` deallocates both its
arguments, and `ExtraMsaOnDevice.extra_msa` captures `const` from outside the checkpoint and
passes it there, so the forward frees it and the recompute asks for it again.

Two arms, same shapes, one variable: whether the checkpointed callable frees its captured tensor.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch                                                           # noqa: E402
import ttnn                                                            # noqa: E402
from tt_bio import autograd as ag, tenstorrent as T                    # noqa: E402

N = 32


def arm(dev, free_the_capture: bool):
    mk = lambda: ttnn.from_torch(torch.randn(N, N), dtype=ttnn.bfloat16,     # noqa: E731
                                 layout=ttnn.TILE_LAYOUT, device=dev)
    x = ag.Tensor(mk(), requires_grad=True)
    const = ag.Tensor(mk(), requires_grad=False)

    def consuming(t, c):
        out = ag.matmul(t, c)
        if free_the_capture:
            ttnn.deallocate(c.value)     # what af2._residual does to both its inputs
        return out

    z = ag.checkpoint(lambda t, c=const: consuming(t, c), x)
    seed = ttnn.from_torch(torch.ones(N, N), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev)
    ag.backward([z], [seed])
    return x.grad is not None


def main():
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    out = {}
    for free in (False, True):
        label = "frees its capture (the extra-MSA shape)" if free else "control, keeps it"
        try:
            got = arm(dev, free)
            out[free] = f"OK, grad arrived={got}"
        except Exception as e:                                          # noqa: BLE001
            # str(e) on a TT_THROW is a C++ backtrace; the line that names the fault is the
            # first one, not the last.
            head = next((l.strip() for l in str(e).splitlines() if l.strip()), "")
            info = next((l.strip() for l in str(e).splitlines()
                         if l.strip() and not l.strip().startswith(('---', 'backtrace', 'info'))
                         and 'so(' not in l), head)
            out[free] = f"RAISED {type(e).__name__}: {info[:110]}"
        print(f"{label:42s} -> {out[free]}", flush=True)
    ttnn.close_device(dev)
    ok = out[False].startswith("OK") and out[True].startswith("RAISED")
    print("REPRO_CONFIRMED" if ok else "REPRO_INCONCLUSIVE", flush=True)


if __name__ == "__main__":
    main()
