#!/usr/bin/env python3
"""trainarm.py with every sample's pre-clip gradient hashed per parameter, in backward order.

The combined tree's step 0 loss is bit-identical run to run and its step 1 loss is not, so the
update differs and the forward does not. Two runs of this, diffed by `gradhash_diff.py`, name
the parameters whose gradient differs; the one nearest the loss in backward order is where the
nondeterminism enters.

    gradhash.py --hash-out <json> <trainarm argv...>
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
i = sys.argv.index("--hash-out")
OUT = Path(sys.argv[i + 1])
del sys.argv[i:i + 2]

import numpy as np                                                    # noqa: E402
from tt_bio.train import optim                                        # noqa: E402
from tt_bio.train.tensors import to_host                              # noqa: E402

_orig = optim.AdamW.clip_and_accumulate
samples = []


def clip_and_accumulate(self, disabled=()):
    rows = {}
    for name, t in self.params.items():
        if t.grad is None:
            continue
        g = np.ascontiguousarray(to_host(t.grad).astype(np.float32))
        rows[name] = [hashlib.sha1(g.tobytes()).hexdigest()[:16], float(np.linalg.norm(g)),
                      list(g.shape), int(np.isnan(g).sum())]
    samples.append(rows)
    OUT.write_text(json.dumps(samples) + "\n")
    return _orig(self, disabled)


optim.AdamW.clip_and_accumulate = clip_and_accumulate
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_p10trainout"))
import trainarm                                                       # noqa: E402

raise SystemExit(trainarm.main())
