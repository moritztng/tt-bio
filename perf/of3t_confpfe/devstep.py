#!/usr/bin/env python3
"""of3t-confpfe: of3t-denoise's devstep.py, plus what the confidence heads saw.

    devstep.py --repr-out R.pt [--conf-dump C.pt [--forward-only]]
               <every perf/of3t_denoise/devstep.py argument>

The step is unchanged. After the forward, the token-scope structure the rollout handed the
confidence heads (`repr_coords`) and the rolled-out atoms are written to R.pt, so a float64
reference can be evaluated at the device's own structure.

`--conf-dump` also writes the confidence head's device inputs (s_input, s_trunk, z_trunk) and
every output it returns, as host float32, so upstream's head can be run in float64 on exactly
those inputs (bisection step 3). `--forward-only` stops after the forward: it is a diagnostic,
not an arm, and writes no gradient.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pop(argv, flag, value=True):
    if flag not in argv:
        return None
    i = argv.index(flag)
    got = argv[i + 1] if value else True
    del argv[i:i + (2 if value else 1)]
    return got


def main() -> int:
    argv = sys.argv[1:]
    repr_out = _pop(argv, "--repr-out")
    conf_dump = _pop(argv, "--conf-dump")
    forward_only = _pop(argv, "--forward-only", value=False)

    import torch
    import ttnn
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from tt_bio.train.openfold3 import OpenFold3Forward

    host = lambda t: torch.Tensor(ttnn.to_torch(getattr(t, "value", t))).float().cpu()
    seen = {}
    if conf_dump:
        orig_fd = OF3ConfidenceHead.forward_device

        def forward_device(self, si_input_d, si_trunk_d, zij_trunk_d, oh_d, **kw):
            out = orig_fd(self, si_input_d, si_trunk_d, zij_trunk_d, oh_d, **kw)
            seen["inputs"] = {"s_input": host(si_input_d), "s_trunk": host(si_trunk_d),
                              "z_trunk": host(zij_trunk_d)}
            seen["dtypes"] = {k: str(getattr(t, "value", t).dtype) for k, t in
                              (("s_input", si_input_d), ("s_trunk", si_trunk_d),
                               ("z_trunk", zij_trunk_d))}
            seen["outputs"] = {k: host(v) for k, v in out.items()}
            return out

        OF3ConfidenceHead.forward_device = forward_device

    orig_call = OpenFold3Forward.__call__

    def call(self, batch):
        got = orig_call(self, batch)
        atoms = getattr(self, "rollout_coords", None)
        torch.save({"repr_x": self.repr_coords.double(),
                    "atoms": None if atoms is None else atoms.double()}, repr_out)
        if conf_dump:
            torch.save({**seen, "repr_x": self.repr_coords.double()}, conf_dump)
        if forward_only:
            print(f"FORWARD-ONLY: wrote {repr_out} and {conf_dump}; no backward", flush=True)
            raise SystemExit(0)
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
