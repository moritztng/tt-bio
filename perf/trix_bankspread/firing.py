#!/usr/bin/env python3
"""Which channel-move op does a 512 aa fold actually call?

`trix-transaction` measured `reblock_permute` and carried a fold saving out of it. The fold A/B in
`fold_ab_512_qb2c2.json` read `reblock_permute.STATS[0] == 0` on all nine folds, so this counts
every variant in one fold and settles it: served, declined, and for the gated op the shape it is
called at.
"""
from __future__ import annotations

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
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import reblock_permute as rbp
    import tt_bio.tenstorrent as TT
    _E.set_progress(lambda *a, **k: None)

    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="trix-firing-", dir=str(REPO / "perf" / "trix_bankspread")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / "cdk2x2_512.yaml", (FIX / "cdk2x2_512.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("trix-bankspread-firing", cfg)

    # record the shape every gated call runs at, so the next pass knows its Ct without guessing
    shapes: dict = {}
    orig = rbp.reblock_permute_gated

    def traced(gp, *gate, **kw):
        k = (int(gp.shape[1]), int(gp.shape[2]), int(gp.shape[3]))
        shapes[str(k)] = shapes.get(str(k), 0) + 1
        return orig(gp, *gate, **kw)

    rbp.reblock_permute_gated = traced
    TT._reblock.reblock_permute_gated = traced

    for nm in ("STATS", "STATS_BACK", "STATS_GATED"):
        s = getattr(rbp, nm)
        s[0] = s[1] = 0
    rbp.REJECTS.clear()
    t = time.perf_counter()
    state.predict_one(FIX / "cdk2x2_512.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t

    out = {
        "fold_s": round(wall, 3), "size": 512,
        "reblock_permute_served_declined": list(rbp.STATS),
        "reblock_permute_back_served_declined": list(rbp.STATS_BACK),
        "reblock_permute_gated_served_declined": list(rbp.STATS_GATED),
        "gated_shapes_N_N_C": shapes,
        "rejects": {str(k): v for k, v in rbp.REJECTS.items()},
        "trimul_mask_after_move": bool(TT._TRIMUL_MASK_AFTER_MOVE),
        "aiclk": int(Path("/sys/class/tenstorrent/tenstorrent!2/tt_aiclk").read_text().strip()),
        "loadavg1": round(os.getloadavg()[0], 2),
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
