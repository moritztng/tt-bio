"""dfd-bucket: which matmul, linear and gather shapes a dock asks the card for, and from where.

    python scripts/diffdock_port/bucket_shapes.py <out.json> <cap> dock <protein.pdb> --ligand ... --seed 0

*cap* is ``conv.BLOCK_CAP`` in tiles for this run, 0 for none.

Runs ``tt-bio`` with the three shape-keyed ops wrapped. Writes one record per distinct
(op, input shapes, call site) with its call count, so two complexes can be compared set against
set: a signature only one of them has is a kernel the other did not warm.
"""
import json
import sys
import traceback
from collections import Counter

import ttnn

seen = Counter()


def wrap(name):
    real = getattr(ttnn, name)

    def call(*args, **kwargs):
        shapes = tuple(tuple(a.shape) for a in list(args) + [kwargs.get("bias")] if hasattr(a, "shape"))
        site = next((f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}"
                    for f in reversed(traceback.extract_stack(limit=12)[:-1]) if "diffdock/" in f.filename), "?")
        seen[(name, shapes, site)] += 1
        return real(*args, **kwargs)

    setattr(ttnn, name, call)


if __name__ == "__main__":
    out = sys.argv[1]
    from tt_bio.diffdock import conv
    if hasattr(conv, "BLOCK_CAP"):
        conv.BLOCK_CAP = int(sys.argv[2]) or None
    sys.argv = ["tt-bio"] + sys.argv[3:]
    for op in ("matmul", "linear", "embedding"):
        wrap(op)
    from tt_bio.main import cli
    try:
        cli()
    finally:
        json.dump([{"op": o, "shapes": s, "site": w, "calls": n} for (o, s, w), n in seen.items()],
                  open(out, "w"))
