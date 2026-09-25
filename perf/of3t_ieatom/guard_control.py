#!/usr/bin/env python3
"""of3t-ieatom: fire the checkpoint-side guard in `check_registered`, or show it silent.

    guard_control.py --batch B.pt --out F.json

One taped forward of the OpenFold3 training adapter. `check_registered` raises when a checkpoint
key the step trains was never uploaded into a registered leaf, and names the keys. On the pre-fix
tree that must be exactly the 93 `input_embedder.atom_attn_enc` tensors BIJECTION_DN.json files as
`upstream_unplaced`; on the fix it must stay silent. The lineage census (keys reached per module)
is written either way, so a silent guard is shown to have looked.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    batch, out = Path(argv[argv.index("--batch") + 1]), Path(argv[argv.index("--out") + 1])
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import walk_device_weights
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    bij = json.loads((HERE.parent / "of3t_denoise/BIJECTION_DN.json").read_text())
    the93 = sorted(k for k in bij["upstream_unplaced"] if k.startswith("input_embedder."))
    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    rec = {"batch": str(batch), "denoise": fwd.denoise,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
           "git_dirty": subprocess.run(["git", "status", "--porcelain", "tt_bio"],
                                       capture_output=True, text=True).stdout.strip()}
    try:
        fwd(OpenFold3Dataset(batch).batch([0]))
        rec["fired"], named = False, []
    except RuntimeError as e:
        rec["fired"], rec["message"] = True, str(e)
        m = re.search(r"train as constants: (\[.*\])$", str(e), re.S)
        named = sorted(eval(m.group(1))) if m else []
    live = {p: t for p, _o, _k, t in walk_device_weights(fwd.model)}
    reached = {k for p, ks in fwd._lineage.items()
               if p in live and ag.parameter_for(live[p]) is not None for k in ks}
    rec["claimed"] = len(fwd._claimed)
    rec["reached"] = len(reached & fwd._claimed)
    rec["leaves_with_lineage"] = len(fwd._lineage)
    rec["leaves"] = len(live)
    rec["reached_by_module"] = dict(Counter(".".join(k.split(".")[:2]) for k in reached))
    rec["named"] = len(named)
    rec["named_equals_the_93"] = named == the93
    rec["named_not_in_93"] = sorted(set(named) - set(the93))
    rec["the_93_not_named"] = sorted(set(the93) - set(named))
    rec["named_keys"] = named
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("fired", "claimed", "reached", "named",
                                          "named_equals_the_93")}), flush=True)
    if rec["fired"] and not named:
        print(rec["message"][:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
