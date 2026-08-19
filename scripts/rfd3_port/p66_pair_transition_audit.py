#!/usr/bin/env python3
"""p66 -- which real `Transition` call diverges, and by how much.

The p64 screen found maxabs 0.0 and torch.equal between the whole-tensor and the L1-chunked path
at the production pair shape on random weights, and then the 3-timestep fold's CIF digest moved
anyway (perf/p65/smoke3.json). That is `tt-bio-ttnn-slice-not-a-view-and-allocation-order-
sensitivity` exactly: a chunkwise equality test can pass while the fold digest does not.

So audit every real call. `Transition.__call__` computes the shipped result AND each candidate
variant on the live activations and weights, prints max|diff| per call site, and returns the
SHIPPED result so the trajectory stays on the shipped path and later calls are still audited
against shipped inputs.

Variants, narrowest first, so the divergence is attributed to one op and not to the bundle:

  chunk_dram   row-chunk on dim 1, every intermediate still in DRAM   (slice + concat only)
  chunk_l1     row-chunk, rms_norm/fc1/multiply in L1, fc2 -> DRAM
  chunk_l1fc2  row-chunk, fc2 -> L1 as well                          (what was built)
  whole_l1     no chunking, intermediates in L1 where they fit        (memory_config alone)
"""
import json
import os
import pathlib
import sys

import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                          # noqa: E402
from tt_bio.rfd3 import model as M                                     # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN                          # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p66/audit.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CKPT = "/home/ttuser/.boltz/rfd3/weights"
L1 = ttnn.L1_MEMORY_CONFIG
SEEN = {}


L1_SEL_VARIANTS = [
    ("chunk_dram", ()),                       # chunking alone: measured 0.0, the control
    ("l1_norm", ("norm",)),
    ("l1_fc1", ("fc1",)),
    ("l1_mul", ("mul",)),
    ("l1_mul_inplace", ("mul", "inplace")),
    ("l1_fc2", ("fc2",)),
    ("l1_norm_mul_inplace", ("norm", "mul", "inplace")),
    ("l1_norm_fc2_mul_inplace", ("norm", "fc2", "mul", "inplace")),
    ("l1_all", ("norm", "fc1", "fc2", "mul", "inplace")),
]


def swiglu_sel(self, x, sel):
    """`Transition._swiglu` with L1 placement selectable per op, so a divergence lands on one op.

    `inplace` is the multiply writing into `fc1`'s buffer (what the shared Transition's
    `multiply_` does); without it the product gets its own L1 buffer and costs a third resident.
    """
    mc = lambda k: {"memory_config": L1} if k in sel else {}
    xn = ttnn.rms_norm(x, weight=self.norm_w, epsilon=1e-6,
                       compute_kernel_config=self.compute_kernel_config, **mc("norm"))
    a = ttnn.linear(xn, self.fc1_w, activation="silu",
                    compute_kernel_config=self.compute_kernel_config, dtype=self.dtype,
                    core_grid=M.BATCH_INVARIANT_GRID, **mc("fc1"))
    b = M._tuned_linear(xn, self.fc2_w, ckc=self.compute_kernel_config, dtype=self.dtype,
                        core_grid=M.BATCH_INVARIANT_GRID,
                        mem=L1 if "fc2" in sel else None)
    ttnn.deallocate(xn)
    kw = dict(mc("mul"))
    if "inplace" in sel:
        kw["output_tensor"] = a
    m = ttnn.multiply(a, b, **kw)
    ttnn.deallocate(b)
    out = M._tuned_linear(m, self.fc3_w, ckc=self.compute_kernel_config, dtype=self.dtype,
                          core_grid=CORE_GRID_MAIN)
    ttnn.deallocate(m)
    return out


def chunked(self, x, h, sel):
    H = x.shape[1]
    parts = []
    for st in range(0, H, h):
        c = x[:, st:min(st + h, H)]
        parts.append(swiglu_sel(self, c, sel))
        ttnn.deallocate(c)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=1)
    for pp in parts:
        ttnn.deallocate(pp)
    return out


def audited(self, x):
    ref = self._swiglu(x, None)
    if not (len(x.shape) == 4 and x.shape[2] >= M._PAIR_TRANSITION_MIN_W):
        return ref
    hidden = int(self.fc1_w.shape[-1])
    h = max(1, min(int(x.shape[1]), M._PAIR_TRANSITION_CHUNK_ELEMS
                   // (int(x.padded_shape[2]) * hidden)))
    key = "%s|H=%d|h=%d" % ("x".join(str(d) for d in x.shape), hidden, h)
    row = SEEN.setdefault(key, {"calls": 0})
    row["calls"] += 1
    if row["calls"] > 1:
        return ref
    for name, sel in L1_SEL_VARIANTS:
        try:
            got = chunked(self, x, h, sel)
            row[name] = M._mm_maxabs(got, ref)
            ttnn.deallocate(got)
        except Exception as e:
            row[name] = "ERR: " + str(e).splitlines()[0][:110]
        print("  [p66] %-30s %-24s %s" % (key, name, row[name]), flush=True)
    return ref


def main():
    M.Transition.__call__ = audited
    M._PAIR_TRANSITION_L1 = False
    specs = json.loads(pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json").read_text())
    os.system("rm -rf /tmp/rfd3_p66")
    rfd3_design.run_design(specs, "/tmp/rfd3_p66", checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1,
                           verbose=False)
    print("\n%-30s %s" % ("call site", "max|variant - shipped|"))
    for k, v in SEEN.items():
        print("%-30s calls=%-3d %s" % (k, v["calls"], "  ".join(
            "%s=%s" % (n, v[n]) for n, _ in L1_SEL_VARIANTS if n in v)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sites": SEEN, "num_timesteps": STEPS, "seed": 42,
                               "host": "qb2", "card": 2}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
