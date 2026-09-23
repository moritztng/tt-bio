#!/usr/bin/env python3
"""of3t-denoise: fullstep64's device step with the one-step denoise arm on.

    devstep.py [--denoise] <every fullstep64 devstep.py argument>

`perf/of3t_fullstep64/devstep.py` unchanged, with three things added around it:

  * `--denoise` reaches `trainfwd_run.py`, so the adapter's own switch turns the arm on;
  * the sigma and the noise the adapter drew are recorded (value, sha256 of the float32 noise),
    which is what the float64 reference must reproduce;
  * after the forward, every device tensor the model holds is compared, by identity, with the
    set the registering walk made into leaves. Any tensor minted after registration is listed
    by path (`unregistered_after_forward`), which is D256's measurement and the guard's control.

The grad dump is then read back for finiteness and max|g| per top-level module.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS64 = HERE.parent / "of3t_fullstep64"


def main() -> int:
    argv = sys.argv[1:]
    denoise = "--denoise" in argv
    argv = [a for a in argv if a != "--denoise"]
    out = Path(argv[argv.index("--out") + 1])
    grad_out = argv[argv.index("--grad-out") + 1]
    sys.path[0:0] = [str(FS64), os.path.join(os.getcwd(), "perf/of3t_trainfwd")]

    import torch
    import trainfwd_run
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import walk_device_weights
    from tt_bio.train.openfold3 import OpenFold3Forward

    extra = {"denoise": denoise}
    orig_main = trainfwd_run.main

    def main_with_denoise():
        if denoise:
            sys.argv.append("--denoise")
        return orig_main()

    trainfwd_run.main = main_with_denoise

    orig_call = OpenFold3Forward.__call__

    def call(self, batch):
        got = orig_call(self, batch)
        late = [p for p, _o, _k, t in walk_device_weights(self.model)
                if ag.parameter_for(t) is None]
        extra["unregistered_after_forward"] = {"n": len(late), "paths": late}
        extra["denoise_sigma"] = self.denoise_sigma
        return got

    OpenFold3Forward.__call__ = call


    sys.argv = [str(FS64 / "devstep.py")] + argv
    import devstep
    rc = devstep.main()

    rec = json.loads(out.read_text())
    g = torch.load(grad_out, weights_only=False)
    mods = {}
    for name, v in g.items():
        v = v.double()
        key = ".".join(name.split(".")[:2]) if name.startswith(("sampler.", "trunk.")) \
            else name.split(".")[0]
        e = mods.setdefault(key, {"n": 0, "nonfinite_tensors": 0, "max_abs": 0.0, "sq": 0.0})
        e["n"] += 1
        fin = torch.isfinite(v)
        if not bool(fin.all()):
            e["nonfinite_tensors"] += 1
            e.setdefault("nonfinite_names", []).append(name)
        vf = v[fin]
        if vf.numel():
            e["max_abs"] = max(e["max_abs"], float(vf.abs().max()))
            e["sq"] += float((vf ** 2).sum())
    extra["by_module"] = mods
    extra["nonfinite_total"] = sum(e["nonfinite_tensors"] for e in mods.values())
    rec["of3t_denoise"] = extra
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DENOISE " + json.dumps({k: extra[k] for k in ("denoise", "denoise_sigma",
                                                          "nonfinite_total")}
                                  | {"unregistered": extra.get("unregistered_after_forward",
                                                                {}).get("n")}), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
