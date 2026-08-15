#!/usr/bin/env python3
"""npz -> flat binary, so the C++ replay harness needs no npz reader.

Layout: 13 int64 geom, then mdl(mdlX*mdlY*mdlZ*2 f32, interleaved re/im), eul(on*9),
tx(tn), ty(tn), img_r(is), img_i(is), w(is).  Same order the bridge received them.
"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
d = np.load(src)
g = d["geom"].astype(np.int64)
with open(dst, "wb") as fh:
    fh.write(g.tobytes())
    for k in ("mdl", "eul", "tx", "ty", "img_r", "img_i", "w"):
        a = np.ascontiguousarray(d[k], dtype=np.float32)
        fh.write(a.tobytes())
print(src, "->", dst, dict(zip(
    "mdlX mdlY mdlZ mdlInitY mdlInitZ maxR maxR2_padded padding_factor imgX imgY "
    "orientation_num translation_num image_size".split(), g.tolist())))
