#!/usr/bin/env python3
"""Fold one size and hash the structure, so a branch can be shown to change nothing.

The gated reader's ragged path -- the tile-padding rows -- never runs at 512 aa, where every row
group is full. 298 aa is the size that exercises it, and it is the size this repo's forward kernels
carry an explicit ragged branch for. Run this on the branch and on the merge base and compare the
hash and the call census.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"


def main():
    size = int(sys.argv[2])
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import reblock_permute as rbp
    _E.set_progress(lambda *a, **k: None)

    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="trix-gb-struct-", dir="/tmp"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{size}.yaml", (FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("trix-gatedbank-struct", cfg)
    for nm in ("STATS", "STATS_BACK", "STATS_GATED"):
        s = getattr(rbp, nm)
        s[0] = s[1] = 0
    t = time.perf_counter()
    state.predict_one(FIX / f"cdk2x2_{size}.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t

    files = sorted(q for q in struct_dir.rglob("*") if q.is_file())
    hashes = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()
              for q in files if q.suffix in (".cif", ".pdb", ".mmcif")}
    out = {
        "size": size, "fold_s": round(wall, 3),
        "structures": hashes,
        "reblock_permute": list(rbp.STATS),
        "reblock_permute_back": list(rbp.STATS_BACK),
        "reblock_permute_gated": list(rbp.STATS_GATED),
        # absent on the merge base, which is the point of running this script on both
        "knobs": {k: getattr(rbp, k, None) for k in ("CT_STREAM", "ROW_BATCH", "CT_BUFS")},
        "aiclk": {str(n): int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text())
                  for n in range(4)},
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
