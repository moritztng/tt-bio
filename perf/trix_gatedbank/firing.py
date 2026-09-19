#!/usr/bin/env python3
"""Which channel-move ops does a 512 aa fold call, at which shapes, with which arguments?

Step 1 of `trix-gatedbank`, and it is non-negotiable: `trix-bankspread` measured a whole chain on
`reblock_permute`, which serves zero calls. This counts `reblock_permute`, `reblock_permute_gated`
and `reblock_permute_back` in one fold and records, for every gated call, the arguments that decide
the kernel's index arithmetic -- the wide channel count, the slice width, the two slice offsets and
the row offset -- not just the op name. A per-op ratio and an in-fold share are only multipliable
if they name the same executed program, and an op-class name is not that identity.
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


def clocks():
    out = {}
    for n in range(4):
        p = Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk")
        try:
            out[str(n)] = int(p.read_text().strip())
        except Exception:
            out[str(n)] = -1
    return out


def main():
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
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
    grid = dev.compute_with_storage_grid_size()
    work = Path(tempfile.mkdtemp(prefix="trix-gb-firing-", dir=str(REPO / "perf" / "trix_gatedbank")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{size}.yaml", (FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
    cfg = H.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("trix-gatedbank-firing", cfg)

    gated: dict = {}
    back: dict = {}
    fwd: dict = {}
    o_gated, o_back, o_fwd = rbp.reblock_permute_gated, rbp.reblock_permute_back, rbp.reblock_permute

    def t_gated(xw, p_slice, g_slice, slice_c, memory_config=None, device=None, out=None, row_off=0):
        N = int(out.shape[2]) if out is not None else int(xw.shape[1])
        Ct = (int(out.shape[1]) if out is not None else slice_c) // 32
        Ctw = int(xw.shape[3]) // 32
        Nt = (N + 31) // 32
        Nrt = (int(xw.shape[1]) + 31) // 32
        k = json.dumps({
            "xw": [int(d) for d in xw.shape],
            "out_C_N": [int(out.shape[1]), int(out.shape[2])] if out is not None else [slice_c, N],
            "slice_c": int(slice_c), "p_off_tiles": p_slice // 32, "g_off_tiles": g_slice // 32,
            "row_off": int(row_off), "Nt": Nt, "Nrt": Nrt, "Ct": Ct, "Ctw": Ctw,
            "row_stride_pages": Nt * Ctw, "row_stride_mod8": (Nt * Ctw) % 8,
            "pg_gap_tiles": abs(p_slice // 32 - g_slice // 32),
            "pg_gap_mod8": abs(p_slice // 32 - g_slice // 32) % 8,
            "num_groups": Nrt * Nt * Ct,
            "buffer_type": str((memory_config or xw.memory_config()).buffer_type),
        }, sort_keys=True)
        gated[k] = gated.get(k, 0) + 1
        return o_gated(xw, p_slice, g_slice, slice_c, memory_config=memory_config,
                       device=device, out=out, row_off=row_off)

    def t_back(x, memory_config=None, device=None):
        C, N = int(x.shape[1]), int(x.shape[2])
        Nt, Ct = N // 32, C // 32
        k = json.dumps({"x": [int(d) for d in x.shape], "Nt": Nt, "Ct": Ct,
                        "num_groups": Nt * Nt * Ct,
                        "buffer_type": str((memory_config or x.memory_config()).buffer_type)},
                       sort_keys=True)
        back[k] = back.get(k, 0) + 1
        return o_back(x, memory_config=memory_config, device=device)

    def t_fwd(x, memory_config=None, device=None):
        k = json.dumps({"x": [int(d) for d in x.shape]}, sort_keys=True)
        fwd[k] = fwd.get(k, 0) + 1
        return o_fwd(x, memory_config=memory_config, device=device)

    for mod in (rbp, TT._reblock):
        mod.reblock_permute_gated = t_gated
        mod.reblock_permute_back = t_back
        mod.reblock_permute = t_fwd

    for nm in ("STATS", "STATS_BACK", "STATS_GATED"):
        s = getattr(rbp, nm)
        s[0] = s[1] = 0
    rbp.REJECTS.clear()
    clk0 = clocks()
    t = time.perf_counter()
    state.predict_one(FIX / f"cdk2x2_{size}.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t
    clk1 = clocks()

    out = {
        "size": size, "fold_s": round(wall, 3),
        "host": os.uname().nodename, "grid": [grid.x, grid.y],
        "reblock_permute_served_declined": list(rbp.STATS),
        "reblock_permute_back_served_declined": list(rbp.STATS_BACK),
        "reblock_permute_gated_served_declined": list(rbp.STATS_GATED),
        "gated_calls": {k: v for k, v in sorted(gated.items())},
        "back_calls": {k: v for k, v in sorted(back.items())},
        "fwd_calls": {k: v for k, v in sorted(fwd.items())},
        "rejects": {str(k): v for k, v in rbp.REJECTS.items()},
        "trimul_mask_after_move": bool(TT._TRIMUL_MASK_AFTER_MOVE),
        "aiclk_before": clk0, "aiclk_after": clk1,
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
