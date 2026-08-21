#!/usr/bin/env python3
"""Does the persistent-mask SDPA fit L1 when a core owns TWO q chunks instead of one?

The 1024 aa TriangleAttention cliff (`sizes-recheck-opendde` Defect A, 72.680 s / 11.14 % of the
fold) is a work-split ceiling, not a q_chunk ceiling. `triatt_sdpa.sdpa` forces
`q_pf == q_num_chunks` because the hoisted mask fill assumes one q chunk per core, so at 1024 aa
the split granularity is `n_heads * q_num_chunks` = 12 * 4 = 48 cores and only 96 of 130 are given
a batch range. The makespan is then ceil(1024 / floor(130/48)) = 512 rows per core; with two q
chunks per core the granularity is 24, floor(130/24) = 5, and the makespan is 205 rows x 2 chunks
= 410 q256-units. 512/410 = 1.249x, and nothing about the arithmetic per (row, q chunk) changes,
so it is bit-exact by construction.

The only thing that can kill it is L1: the mask CB has to hold `q_per_core * k_num_chunks *
Sq_chunk_t * Sk_chunk_t` tiles instead of one q chunk's worth. This screen answers exactly that,
and times the two splits at the real shape while it is there. The B arm computes WRONG numbers --
the shipped reader fills only the first q chunk's mask blocks -- but it allocates the real CBs and
issues the real per-core work, so its wall is the wall the fixed kernel would have.

Arms, at OpenDDE's tri-attention shape (batch = S, n_heads = 12, d = 32, bf16 interleaved DRAM):
  A  q_chunk 256, split (2,12,4), mask CB 256 tiles  -- what main runs today at 1024 aa
  B  q_chunk 256, split (5,12,2), mask CB 512 tiles  -- the proposal
  C  q_chunk 512, split (5,12,2), mask CB 512 tiles  -- the alternative; expected to refuse
Run at 768 too, where the same proposal is expected to refuse (mask CB 576 tiles), which is what
makes the lever 1024-only and therefore incapable of moving the 512 aa page cell.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WARM, REPS = 2, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,768")
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.sdpa_generic as SG
    import tt_bio.triatt_sdpa as PM
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    cores = g.x * g.y
    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "grid": [g.x, g.y], "cores": cores,
           "l1_bank_bytes": int(T._l1_bank_bytes()), "rows": []}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)

    ckc = (ttnn.MathFidelity.HiFi2, True, False, False)
    H, DH = a.heads, a.dh

    for S in [int(s) for s in a.sizes.split(",")]:
        mk = lambda shp: ttnn.allocate_tensor_on_device(          # noqa: E731
            ttnn.Shape(shp), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        q = mk([S, H, S, DH]); k = mk([S, H, S, DH]); v = mk([S, H, S, DH])
        bias = mk([1, H, S, S]); out = mk([S, H, S, DH])
        k_chunk = T._sdpa_chunks_shipped(S, S)[1]
        arms = []
        for q_chunk in sorted({256, 512, T._sdpa_chunks_shipped(S, S)[0]}):
            qnc = -(-S // q_chunk)
            for q_pf in sorted({qnc, max(1, qnc // 2)}):
                b_pf = cores // (H * q_pf)
                if b_pf < 1:
                    continue
                arms.append((q_chunk, q_pf, b_pf))
        for q_chunk, q_pf, b_pf in arms:
            split = (b_pf, H, q_pf)
            p = SG.plan(q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, 1.0, split)
            mask_tiles = p["q_per_core"] * p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            cb_tiles = (p["q_tiles"] + p["k_tiles"] + p["v_tiles"] + mask_tiles + 3
                        + p["qk_tiles"] + 2 * p["out_im_tiles"] + 5 * p["statistics_tiles"]
                        + p["out0_t"])
            row = {"S": S, "q_chunk": q_chunk, "k_chunk": k_chunk, "split": list(split),
                   "cores_used": b_pf * H * q_pf, "q_per_core": p["q_per_core"],
                   "batch_per_core": p["batch_per_core"], "mask_cb_tiles": mask_tiles,
                   "cb_tiles_total": cb_tiles, "cb_bytes_total": cb_tiles * 2048,
                   "fill_preconditions_ok": bool(
                       p["nh_per_core"] == 1 and p["bcast_batch"] and not p["use_padded_mask"]
                       and p["NKH"] == H and p["NVH"] == H)}
            def run():
                SG.sdpa(dev, q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, 1.0, split=split,
                        kernel_dir=PM.KERNEL_DIR, mask_cb_tiles=mask_tiles,
                        defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})
            try:
                for _ in range(WARM):
                    run()
                ttnn.synchronize_device(dev)
                ts = []
                for _ in range(REPS):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    run()
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) * 1e3)
                row["ms"] = round(st.median(ts), 4)
                row["ms_all"] = [round(t, 4) for t in ts]
            except Exception as exc:                              # noqa: BLE001
                row["error"] = str(exc).strip().split("\n")[-1][:200]
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
        for t in (q, k, v, bias, out):
            ttnn.deallocate(t)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
