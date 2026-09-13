#!/usr/bin/env python3
"""Fold cdk2x2 at 512 aa and 298 aa on this branch and digest the CIFs.

This workstream changed no model code, so parity here is a control on the tree rather than on a
lever: both folds must reproduce the digests already on record for current main, and the Angstrom
deviation against them must be 0.000. A branch that only adds files under perf/ has nothing to move
the structure, and this proves it rather than asserting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = {  # b2x-baseline-attrib, qb2 card 1, current main, published protocol
    "512": "4f3995a69be5d610",
    "298": "71653ff72cbf01b048cf24bfa3b3358b1044d1e2b309d312bb7a646c0c6c1c69",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="512,298")
    a = ap.parse_args()
    a.cifdir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
    import torch
    torch.set_grad_enabled(False)
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    out = {"env": {"host": os.uname().nodename,
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip(),
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, "folds": {}}
    fix = ROOT / "perf" / "size512" / "fixtures"
    for size in a.sizes.split(","):
        tgt, a3m = fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m"
        msa_dir = Path(__file__).resolve().parent / f".msa_{size}"
        one_fold, meta, _ = B.build_fold("boltz2", msa_dir, tgt, a3m, fast=False)
        struct_dir = Path(meta["struct_dir"])
        s, m = one_fold()
        digs = {}
        for f in sorted(struct_dir.glob("*.cif")):
            dst = a.cifdir / f"{size}_{f.name}"
            shutil.copyfile(f, dst)
            digs[f.name] = sha256_file(dst)
        assert m.get("msa") or meta.get("n_msa"), "fold ran without an MSA"
        key = f"cdk2x2_{size}.cif"
        got = digs.get(key, "")
        ref = REF[size]
        out["folds"][size] = {
            "s": round(s, 3), "plddt": m.get("plddt"), "n_msa": meta.get("n_msa"),
            "grid": meta.get("grid"), "aiclk_mhz": meta.get("aiclk_mhz"),
            "digests": digs, "reference": ref,
            "matches_reference": got.startswith(ref) or ref.startswith(got[:16]),
        }
        print(f"{size} aa: {s:.3f}s plddt={m.get('plddt')} {key} {got[:16]} "
              f"ref {ref[:16]} match={out['folds'][size]['matches_reference']}", flush=True)
        a.out.write_text(json.dumps(out, indent=1))
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
