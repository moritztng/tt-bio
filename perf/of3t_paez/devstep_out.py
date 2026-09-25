#!/usr/bin/env python3
"""of3t-paez: confpfe's devstep.py, plus the step's own head outputs on host.

    devstep_out.py --outputs-out O.pt <every perf/of3t_confpfe/devstep.py argument>

Writes every output the forward returns (pred_xyz, the logits) as host float64 before
confpfe's wrapper runs, so the objective's pae term can be evaluated at the device's own labels.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    argv = sys.argv[1:]
    i = argv.index("--outputs-out")
    path = argv[i + 1]
    del argv[i:i + 2]
    import torch
    import ttnn
    from tt_bio.train.openfold3 import OpenFold3Forward

    def host(t):
        t = getattr(t, "value", t)
        return (t if torch.is_tensor(t) else torch.Tensor(ttnn.to_torch(t))).double().cpu()

    orig = OpenFold3Forward.__call__

    def call(self, batch):
        got = orig(self, batch)
        torch.save({k: host(v) for k, v in got.items()}, path)
        return got

    OpenFold3Forward.__call__ = call
    sys.argv = [str(HERE.parent / "of3t_confpfe" / "devstep.py")] + argv
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
