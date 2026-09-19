#!/usr/bin/env python3
"""Drop the path-probe markers, keep one that only speaks when the lever is asked for.

The three MM*-PATH markers did their job (they proved create_program_mcast_in1_descriptor in the
1D factory is the live builder for key A).  They print on every program build for everyone using
this shared tree, so they go.  MM1D-FLAG stays, gated on the env var being set at all, so a
co-tenant who never asks for the lever sees nothing.
"""
from pathlib import Path

F1 = Path("/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/"
          "matmul_multicore_reuse_mcast_1d_program_factory.cpp")
F2 = Path("/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/"
          "matmul_multicore_reuse_mcast_2d_program_factory.cpp")

s = F1.read_text()
out = []
dropped = 0
for line in s.split("\n"):
    if "MM1D-PATH" in line:
        dropped += 1
        continue
    if "MM1D-FLAG" in line:
        out.append('    if (writer_on_in0_env_str != nullptr) {')
        out.append('        fprintf(stderr, "MM1D-FLAG writer_on_in0=%d\\n", (int)writer_on_in0);')
        out.append('    }')
        continue
    out.append(line)
F1.write_text("\n".join(out))
print("1D: dropped %d path markers, flag marker gated" % dropped)

s = F2.read_text()
out = [l for l in s.split("\n") if "MM2D-PATH" in l and False or "MM2D-PATH" not in l]
print("2D: dropped %d path markers" % (len(s.split("\n")) - len(out)))
F2.write_text("\n".join(out))
