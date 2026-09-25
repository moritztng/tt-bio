#!/usr/bin/env python3
"""of3t-infab: does the training walk still find the confidence heads once inference stops uploading them?

    PYTHONPATH=TREE walkcheck.py OUT.json

Builds `OpenFold3Forward(...).model` (no forward, no backward) and records the walked device tensor
count, the confidence-head tensors among them and `late_device_weights()`.
"""
import json
import sys
from pathlib import Path

import tt_bio
from tt_bio.tenstorrent import walk_device_weights
from tt_bio.train.openfold3 import OpenFold3Forward

fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=0)
m = fwd.model
walked = [p for p, _o, _k, _t in walk_device_weights(m)]
head = [p for p in walked if "confidence_head" in p]
out = {"tt_bio": tt_bio.__file__, "walked": len(walked), "confidence_head": len(head),
       "confidence_head_paths": head,
       "late_device_weights": [list(map(str, x)) for x in m.confidence_head.late_device_weights()]}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("WALK", out["tt_bio"], out["walked"], out["confidence_head"], out["late_device_weights"], flush=True)
