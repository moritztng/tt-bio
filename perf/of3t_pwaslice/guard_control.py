#!/usr/bin/env python3
"""of3t-pwaslice: count raw slices of registered weights under the tape, and fire the guard.

    guard_control.py --batch B.pt --out F.json

One taped forward of the OpenFold3 training adapter. Every `ttnn.Tensor.__getitem__` whose
receiver is a registered parameter is recorded with the weight's walk path and the first tt-bio
frame (SITES). On the pre-fix tree every one of them is a constant, and `check_registered` at the
end of the forward must raise and name the sites (CONTROL). On the fixed tree the same count
shows where the slices are, and the guard must stay silent.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    batch, out = Path(argv[argv.index("--batch") + 1]), Path(argv[argv.index("--out") + 1])
    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tt
    from tt_bio.tenstorrent import walk_device_weights
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    names = {id(t): p for p, _o, _k, t in walk_device_weights(fwd.model)}
    seen = Counter()
    inner = tt._param_getitem

    def counting(self, *args, **kwargs):
        if ag.parameter_for(self) is not None:
            seen[(names.get(id(self), "?"), tt._tt_bio_site())] += 1
        return inner(self, *args, **kwargs)

    tt._param_getitem = counting
    import subprocess
    rec = {"batch": str(batch), "denoise": fwd.denoise,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()}
    try:
        fwd(OpenFold3Dataset(batch).batch([0]))
        rec["fired"] = False
    except RuntimeError as e:
        rec["fired"] = True
        rec["message"] = str(e)
    rec["raw_param_slices"] = sum(seen.values())
    rec["by_weight_and_site"] = [{"weight": w, "site": s, "calls": n}
                                 for (w, s), n in sorted(seen.items())]
    rec["weights"] = sorted({w for w, _ in seen})
    rec["sites"] = sorted({s for _, s in seen})
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("fired", "raw_param_slices", "sites")}
                     | {"weights": len(rec["weights"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
