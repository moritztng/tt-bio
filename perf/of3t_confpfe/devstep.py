#!/usr/bin/env python3
"""of3t-confpfe: of3t-denoise's devstep.py, plus the structure the confidence heads saw.

    devstep.py --repr-out R.pt <every perf/of3t_denoise/devstep.py argument>

The step is unchanged. After the forward, the token-scope structure the rollout handed the
confidence heads (`repr_coords`) and the rolled-out atoms are written to R.pt, so a float64
reference can be evaluated at the device's own structure and the confidence heads' error split
into what the structure carries and what the heads' arithmetic does.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    i = argv.index("--repr-out")
    repr_out = argv[i + 1]
    del argv[i:i + 2]

    import torch
    from tt_bio.train.openfold3 import OpenFold3Forward

    orig_call = OpenFold3Forward.__call__

    def call(self, batch):
        got = orig_call(self, batch)
        atoms = getattr(self, "rollout_coords", None)
        torch.save({"repr_x": self.repr_coords.double(),
                    "atoms": None if atoms is None else atoms.double()}, repr_out)
        return got

    OpenFold3Forward.__call__ = call
    # Loaded under its own name: it imports fullstep64's module as `devstep`.
    spec = importlib.util.spec_from_file_location(
        "denoise_devstep", HERE.parent / "of3t_denoise" / "devstep.py")
    dn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dn)
    sys.argv = [spec.origin] + argv
    return dn.main()


if __name__ == "__main__":
    raise SystemExit(main())
