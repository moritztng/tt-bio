#!/usr/bin/env python3
"""Read back the reference tree `perf/of3t_trajbar/trajbar.py` actually imports.

`of3t-trajbar`'s bar -- 1.762065e-01 at k = 20, the denominator GO condition 3 divides by --
was taken by a script that RECORDS `TW.REF_TREE` and, read statically, never reads
`openfold3.__file__` back. The composition's D149 ratchet refuses it for exactly that. This
script settles what the ratchet cannot see, by reproducing that import sequence and asking the
interpreter instead of the source:

  * the same two `sys.path.insert(0, p)` calls, in the same loop, over the same two directories
    in the same order as `trajbar.py:47-51`;
  * `import trajwide as TW`, which is the same module object `trajbar.py` gets;
  * `TW.refpath.install()`, which is `trajbar.run_arm`'s first act;
  * `TW.build_theirs(torch.float32)`, which is what `install_mixed`'s `build_mixed` calls into
    on the bf16-mixed arm, and the only place `TW.REF_TREE` is ever assigned.

Then it prints `openfold3.__file__`, the parameter count `run_theirs` records as
`n_parameters`, and the missing/unexpected counts from loading `of3-p2-155k.pt`. 0.4.3 loads
this checkpoint at 0 missing / 0 unexpected; 0.5.0 loads it at 1 missing / 24 unexpected, so
the counts are a second, independent witness to which tree answered.

It also digests the resolved package tree under `of3t-trunk043ref/tree_digest.py`'s
A24-AMENDMENT rule -- sha256 over the sorted per-file sha256 of every `.py` under the package
root -- because this campaign now has TWO 0.4.3 trees on disk, `of3t_refprec/of3pkg043` and the
restored `of3t-campaign-refs/of3pkg043`, and which one a checkout gets depends on which branch
it is. A path is not a version. The digest is.

CPU only. No card. Building the module and loading the checkpoint is the whole cost, ~1 min.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# --------------------------------------------------------------- trajbar.py:47-51, verbatim
#
# Same two directories, same loop, same constant index. `trajbar.py` writes `(TWDIR, HERE)` and
# inserting both at 0 leaves them on sys.path as [HERE, TWDIR] -- the reverse of what is
# written. That reversal is what the D149 ratchet flags, and reproducing it rather than
# correcting it is the point: this has to be the import sequence the bar was measured under,
# not a tidied one.
HERE = os.path.dirname(os.path.abspath(__file__))
TBDIR = os.path.join(os.path.dirname(HERE), "of3t_trajbar")
TWDIR = os.path.join(os.path.dirname(HERE), "of3t_trajwide")
for p in (TWDIR, TBDIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import trajwide as TW                                                    # noqa: E402

WITNESS = [
    "core/model/latent/base_blocks.py",
    "core/model/latent/pairformer.py",
    "core/model/layers/triangular_attention.py",
    "core/model/layers/attention_pair_bias.py",
    "core/model/primitives/attention.py",
    "entry_points/parameters.py",
]


def tree_digest(package_root):
    """`of3t-campaign-refs/tree_digest.py` rule, reimplemented so this script needs nothing
    outside the repo. Checked against the manifest's published value in the output."""
    import pathlib
    root = pathlib.Path(package_root)
    files = sorted(root.rglob("*.py"))
    per = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"package_root": str(root), "n_py_files": len(files),
            "tree_sha256": hashlib.sha256("".join(sorted(per.values())).encode()).hexdigest(),
            "witness_files": {w: per.get(w) for w in WITNESS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)     # run_bar.sh runs the arms at 10
    ap.add_argument("--label", default="branch",
                    help="which tree this checkout is, for the record: branch | composition")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)

    t0 = time.time()
    rec = {"label": a.label, "cwd": os.getcwd(), "executable": sys.executable,
           "trajwide_module": TW.__file__, "refpath_module": TW.refpath.__file__,
           "OF3PKG_constant": TW.refpath.OF3PKG, "REFDEPS_constant": list(TW.refpath.REFDEPS),
           "REF_TREE_before_build": TW.REF_TREE}
    print(f"trajwide  {TW.__file__}")
    print(f"refpath   {TW.refpath.__file__}")
    print(f"OF3PKG constant   {TW.refpath.OF3PKG}")
    print(f"TW.REF_TREE before build_theirs: {TW.REF_TREE!r}")

    TW.refpath.install()                                   # trajbar.run_arm's first act
    rec["sys_path_head_after_install"] = sys.path[:6]

    import torch
    # build_theirs is where `REF_TREE = refpath.assert_resolved()` happens, and it is on the
    # bf16-mixed arm's path: install_mixed's build_mixed calls the original with float32.
    m, own, inc, moved = TW.build_theirs(torch.float32)

    import openfold3
    resolved = os.path.realpath(os.path.dirname(os.path.dirname(
        os.path.abspath(openfold3.__file__))))
    n_parameters = len(dict(m.named_parameters()))

    rec.update({
        "REF_TREE_after_build": TW.REF_TREE,
        "openfold3___file__": openfold3.__file__,
        "resolved_tree": resolved,
        "resolved_equals_REF_TREE": os.path.realpath(TW.REF_TREE or "") == resolved,
        "openfold3_version": getattr(openfold3, "__version__", None),
        "n_parameters": n_parameters,
        "n_elements": int(sum(p.numel() for p in m.parameters())),
        "state_dict_entries_loaded": len(own),
        "missing_keys": len(inc.missing_keys), "unexpected_keys": len(inc.unexpected_keys),
        "layer_norm_z_realigned": moved,
        "checkpoint": TW.CKPT, "checkpoint_bytes": os.path.getsize(TW.CKPT),
        "tree": tree_digest(os.path.join(resolved, "openfold3")),
        "wall_s": round(time.time() - t0, 1),
    })

    print(f"REF_TREE resolved: {resolved}")
    print(f"openfold3.__file__ = {openfold3.__file__}")
    print(f"parameters: {n_parameters}   missing: {len(inc.missing_keys)}   "
          f"unexpected: {len(inc.unexpected_keys)}")
    print(f"tree digest: {rec['tree']['tree_sha256']} over {rec['tree']['n_py_files']} .py files")
    print(f"[{rec['wall_s']}s] {a.label}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
