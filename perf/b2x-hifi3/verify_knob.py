#!/usr/bin/env python3
"""TT_BIO_TRUNK_MATH_FIDELITY reaches Boltz-2's trunk, and nothing else.

Two claims, one fold each, and the digest is what decides both -- the fold is bit-exact within an
arm, so one fold is enough:

  unset      the trunk's wrappers must read HiFi4 and the CIF must be 4f3995a69be5d610, the
             digest current main produces. Production unchanged, proved by bytes.
  hifi3      the trunk's wrappers must read HiFi3 and the CIF must be bf5455fb4977a142, the
             digest the in-process flip produced in hifi_ab.py. Same math through a different
             door.

Prints the fidelity of EVERY compute-kernel-config object in the model, so a leak into the
sampler or the confidence head is visible rather than assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-sha16", default=None)
    ap.add_argument("--expect-trunk-fidelity", required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

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

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{a.size}"
    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)

    rows = []
    for name, mod in state.model.named_modules():
        if isinstance(mod, T.TorchWrapper) and getattr(mod, "compute_kernel_config", None):
            rows.append({"name": name, "class": type(mod).__name__,
                         "n_blocks": getattr(mod, "n_blocks", None),
                         "fidelity": str(mod.compute_kernel_config.math_fidelity)})
    trunk = [r for r in rows
             if r["class"] == "MSAModule" or (r["class"] == "PairformerModule"
                                              and r["n_blocks"] == 64)]
    other = [r for r in rows if r not in trunk]
    fid = sorted({r["fidelity"] for r in trunk})
    print(json.dumps(rows, indent=1), flush=True)

    fold_s, m = one_fold()
    cif = sorted(Path(meta["struct_dir"]).glob("*.cif"))[0]
    sha = hashlib.sha256(cif.read_bytes()).hexdigest()
    res = {
        "env": os.environ.get("TT_BIO_TRUNK_MATH_FIDELITY"),
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "trunk_fidelity": fid, "ckc_map": rows,
        "non_trunk_fidelity": sorted({r["fidelity"] for r in other}),
        "fold_s": round(fold_s, 3), "cif_sha256": sha, "cif_sha256_16": sha[:16],
        "expect_sha16": a.expect_sha16,
        "fidelity_ok": fid == [f"MathFidelity.{a.expect_trunk_fidelity}"],
        "sha_ok": (a.expect_sha16 is None) or sha[:16] == a.expect_sha16,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "ckc_map"}, indent=1), flush=True)
    T.cleanup()
    return 0 if (res["fidelity_ok"] and res["sha_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
