#!/usr/bin/env python3
"""Isolated `torch.equal` probe on the two numerics FIX-2 chooses between (state doc §5.2).

On an idle device both paths place successfully, so this compares the tuned
`MatmulMultiCoreReuseProgramConfig` against `program_config=None` -- FIX-2's fallback -- with
nothing else changing. The throwing class at 512 aa is `AttentionPairBias` q@k^T: 16 heads of
padded width 32 over 512 tokens, so `batch=16, m_tiles=16, k_tiles=1, n_tiles=16`.

Also prints the modelled circular-buffer footprint so the 995 840 B in the throw can be checked
against the closed form from `state/protenix-trunk--z-crashband-fix.md` §2:

    CB(p) = 2*(p + Nt)*block_w*2048 + p*Nt*(2048 + 4096)   plus 111 104 B of fixed per-core ttnn
                                                           overhead

    TT_VISIBLE_DEVICES=2 PYTHONPATH=$WT python3 perf/z_fix2_crossmodel/bmm_equal_probe.py
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn

OVERHEAD = 111_104          # measured on qb1 13x10 @ 0.67.4; this probe re-checks it on 11x10


def cb_model(p, nt, block_w, elem_bytes=2):
    tile, acc = 1024 * elem_bytes, 4096
    return 2 * (p + nt) * block_w * tile + p * nt * (tile + acc)


def main() -> int:
    import tt_bio.tenstorrent as T
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(REPO), f"stale tt_bio: {T.__file__}"

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    bank = int(ttnn.get_max_worker_l1_unreserved_size())
    # The exact config openfold3 builds in `worker.py:413`, so `packer_l1_acc` and
    # `fp32_dest_acc_en` match the real call and the comparison is the one FIX-2 actually makes.
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "cores": g.x * g.y, "bank_bytes": bank,
           "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
           "overhead_const": OVERHEAD, "classes": []}

    # (label, batch, N tokens, padded head width) -- the q@k^T class at 298 aa and at 512 aa, plus
    # its attn@v sibling, which models far smaller and is not the thrower.
    CLASSES = [
        ("q@kT 512aa", 16, 512, 512, 32),
        ("q@kT 298aa(pad320)", 16, 320, 320, 32),
        ("attn@v 512aa", 16, 512, 32, 512),
    ]
    torch.manual_seed(0)
    for label, batch, m, n, k in CLASSES:
        mt, kt_, nt = -(-m // 32), -(-k // 32), -(-n // 32)
        cfg = T._batched_matmul_config(batch, mt, kt_, nt, 2)
        row = {"label": label, "batch": batch, "shape_a": [1, batch, m, k],
               "shape_b": [1, batch, k, n], "m_tiles": mt, "k_tiles": kt_, "n_tiles": nt,
               "cfg": None}
        if cfg is not None:
            bw = int(cfg.in0_block_w)
            p = int(cfg.per_core_M)
            row["cfg"] = {"in0_block_w": bw, "per_core_M": p, "per_core_N": int(cfg.per_core_N),
                          "out_subblock_h": int(cfg.out_subblock_h),
                          "out_subblock_w": int(cfg.out_subblock_w)}
            row["cb_modelled"] = cb_model(p, nt, bw)
            row["cb_plus_overhead"] = cb_model(p, nt, bw) + OVERHEAD
            row["cores_engaged"] = batch * mt // p
            row["cb_fits_idle_bank"] = row["cb_modelled"] <= bank

        ta = torch.randn(1, batch, m, k, dtype=torch.bfloat16)
        tb = torch.randn(1, batch, k, n, dtype=torch.bfloat16)
        kwa = dict(layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qa = ttnn.from_torch(ta, **kwa)
        qb = ttnn.from_torch(tb, **kwa)
        try:
            if cfg is None:
                row["torch_equal"] = None
                row["note"] = "chooser declined; no tuned config to compare"
            else:
                a_t = ttnn.to_torch(ttnn.matmul(qa, qb, compute_kernel_config=ckc,
                                                program_config=cfg))
                b_t = ttnn.to_torch(ttnn.matmul(qa, qb, compute_kernel_config=ckc))
                row["torch_equal"] = bool(torch.equal(a_t, b_t))
                d = (a_t.float() - b_t.float()).abs()
                row["max_abs_diff"] = float(d.max())
        except Exception as e:                                          # noqa: BLE001
            row["error"] = str(e)
        finally:
            ttnn.deallocate(qa)
            ttnn.deallocate(qb)
        out["classes"].append(row)
        print(json.dumps(row), flush=True)

    p = Path(__file__).resolve().parent / "bmm_equal_probe.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
