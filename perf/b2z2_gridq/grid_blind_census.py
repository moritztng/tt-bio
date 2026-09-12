#!/usr/bin/env python3
"""Which of the engine's work-partitioning pickers actually change when the grid does?

`_capped_sdpa_chunk_size` was convicted by reading it. That does not scale and it is not
evidence: a constant can be grid-blind on the page and grid-aware in effect (something
downstream rescales it), or grid-aware on the page and blind in effect (the grid term is
dominated by a cap). So this asks the question by EXPERIMENT instead.

Every picker below is called at the shapes the fleet's models actually run, under three
simulated grids -- Wormhole 8x9 = 72, Blackhole 11x10 = 110, Blackhole 13x10 = 130 -- and the
returns are diffed. A picker whose answer never moves across a 1.8x change in core count is
grid-blind BY MEASUREMENT. Whether that is a defect is a second question the exposure column
answers: a picker is only exposed if the thing it decides is how many cores run the op.

One grid per process, because `_apply_grid_thresholds` returns early on a full-size grid and
therefore cannot restore a baseline it has already tightened -- switching grids inside one
process leaves the small-grid values in place. Production calls it once, so that is not a bug
there; it is a trap for this script.

    python3 grid_blind_census.py --grid 8,9  --out out/g72.json
    python3 grid_blind_census.py --grid 11,10 --out out/g110.json
    python3 grid_blind_census.py --grid 13,10 --out out/g130.json
    python3 grid_blind_census.py --join out/g72.json out/g110.json out/g130.json

No device is opened. Every picker here is a pure function of shapes and module state.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def set_grid(T, gx: int, gy: int) -> None:
    """What `_configure_active_compute_grid` does to module state, without a device."""
    import ttnn
    T.CORE_GRID_MAIN = ttnn.CoreGrid(y=gy, x=gx)
    T.COMPUTE_GRID_MAIN = (gx, gy)
    T._apply_grid_thresholds((gx, gy), None)
    for fn in ("_sdpa_program_config", "_sdpa_program_config_for_lengths", "_grid_q_chunk",
               "_triangle_mul_program_config", "_capped_sdpa_chunk_size",
               "_dividing_sdpa_chunk_size", "_sdpa_chunks_shipped", "_trimul_chunk_size",
               "_tri_att_q_chunks"):
        f = getattr(T, fn, None)
        if hasattr(f, "cache_clear"):
            f.cache_clear()


# (probe id, what it decides, class, callable(T, modules) -> jsonable)
# `cls` is the VERIFIED class of what the number decides, not a guess from its name:
#   OCCUPANCY  -- it sets how many cores run the op. A grid-blind one here can starve the grid.
#   inner-loop -- it sets a loop INSIDE the kernel (k_chunk, in0_block_w). Blindness costs
#                 arithmetic or a fused-kernel refusal, never a core.
#   bytes      -- it bounds an L1 or DRAM peak. Its blocks run one after another and each one
#                 still spreads over the whole grid, so blindness costs headroom, not cores.
#   host       -- it blocks work on the CPU. The Tensix grid is not its resource at all.
#   guard      -- it exists to avoid a hang, and is not a tuning parameter.
PROBES: list[tuple[str, str, str, object]] = [
    # ---- SDPA: q_chunk decides work units = batch*heads*q_chunks, one chunk per core ----
    ("sdpa.capped_q_chunk", "SDPA q_chunk: work units = batch*heads*q_chunks, one per core", "OCCUPANCY",
     lambda T, M: {s: T._capped_sdpa_chunk_size(s) for s in (320, 512, 768, 1024, 1536)}),
    ("sdpa.grid_q_chunk", "THIS ROW'S LEVER -- control, must move with the grid", "OCCUPANCY",
     lambda T, M: {f"{s}/{h}": T._grid_q_chunk(s, h, T._capped_sdpa_chunk_size(s),
                                               T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1])
                   for s in (320, 512, 768, 1024, 1536) for h in (8, 16)}),
    ("sdpa.dividing_k_chunk", "k_chunk: the SDPA inner loop, not a core count", "inner-loop",
     lambda T, M: {s: T._dividing_sdpa_chunk_size(s) for s in (320, 448, 512, 640, 896)}),
    ("sdpa.shipped_band", "the (64,64) band; its q half is occupancy on a saturated path", "OCCUPANCY",
     lambda T, M: {f"{s}": list(T._sdpa_chunks_shipped(s, s)) for s in (288, 320, 352, 384, 512)}),
    ("sdpa.tri_att_q_ladder", "tri-att q_chunks: occupancy, but L1-bound and never starved", "OCCUPANCY",
     lambda T, M: {s: list(T._tri_att_q_chunks(s, s)) for s in (320, 512, 768, 896, 1024)}),
    ("sdpa.fp32_softmax_cores", "cores the ragged fp32-softmax shard may use", "OCCUPANCY",
     lambda T, M: T._fp32_softmax_core_budget()),
    # ---- matmul program configs: per_core_M/N is how the work is spread ----
    ("mm.trimul_per_core", "per_core_M/N of the triangle-mult matmul", "OCCUPANCY",
     lambda T, M: {s: [T._triangle_mul_program_config(s // 32).per_core_M,
                       T._triangle_mul_program_config(s // 32).per_core_N]
                   for s in (320, 512, 1024)}),
    ("mm.trimul_in0_block_w", "in0_block_w: the matmul K loop, not a core count", "inner-loop",
     lambda T, M: {s: T._trimul_in0_block_w(s // 32) for s in (320, 512, 544, 1024)}),
    # ---- chunk pickers whose resource is L1 bytes ----
    ("chunk.trimul", "channel chunk of the triangle multiplication", "bytes",
     lambda T, M: {f"{s}/{h}": T._trimul_chunk_size(s, h) for s in (320, 512, 1024)
                   for h in (128, 256)}),
    ("chunk.transition_h", "H chunk of the pair transition", "bytes",
     lambda T, M: [T.TRANSITION_H_CHUNK_SIZE, T.TRANSITION_H_CHUNK_SIZE_FAST,
                   T.TRANSITION_H_CHUNK_SIZE_BIG, T.TRANSITION_W_CHUNK_SIZE]),
    ("chunk.seq_len_more", "SEQ_LEN_MORE_CHUNKING and the transition thresholds", "bytes",
     lambda T, M: [T.SEQ_LEN_MORE_CHUNKING, T.TRANSITION_BATCH_CHUNKING_THRESHOLD,
                   T.TRANSITION_W_CHUNKING_THRESHOLD]),
    ("rows.pair_row_tile", "rows per block of a pair [B,L,L,*] op", "bytes",
     lambda T, M: {s: T.pair_row_tile(s) for s in (320, 512, 1024, 1536)}),
    ("rows.msa_row_tile", "rows per block of an MSA [B,L,M,*] op", "bytes",
     lambda T, M: {f"{s}/{m}": T.msa_row_tile(s, m) for s in (512, 1024) for m in (1024, 8192)}),
    ("rows.row_block", "rows per block from the L1 residency budget", "bytes",
     lambda T, M: {b: T.row_block(b, T.l1_resident_budget_bytes())
                   for b in (1 << 16, 1 << 20, 1 << 22)}, True),
    ("rows.pwa_depth_block", "MSA depth block of the pair-weighted average", "bytes",
     lambda T, M: {f"{d}/{t}": T.pwa_depth_block(d, t, 64) for d in (1024, 8192)
                   for t in (512, 1024)}, True),
    ("budget.l1_resident", "L1 a row-blocked op may hold", "bytes",
     lambda T, M: T.l1_resident_budget_bytes(), True),
    ("budget.atom_pair", "DRAM an atom-pair section may hold", "bytes",
     lambda T, M: T.atom_pair_budget_bytes(), True),
    ("budget.trimul_l1", "trimul L1 residency thresholds", "bytes",
     lambda T, M: [T.TRIANGLE_MULT_L1_MAX_SEQ, T.TRIANGLE_MULT_L1_MAX_SEQ_FAST,
                   T.TRIANGLE_MULT_L1_CHUNK_BUDGET]),
    # ---- other models on the same engine ----
    ("esmc.pair_ffn_row_block", "ESMC pair-FFN row block", "bytes",
     lambda T, M: [M["esmc"]._PAIR_FFN_ROW_BLOCK, M["esmc"].PAIR_FFN_ROW_BLOCK_SEQ,
                   M["esmc"].PAIR_FFN_ROW_BLOCK_MIN]),
    ("esmfold2.msa_row_blocks", "ESMFold2 MSA row blocks", "bytes",
     lambda T, M: {f"{s}/{m}": list(M["esmfold2"]._msa_row_blocks(s, m))[:1] or [0]
                   for s in (512, 1024) for m in (1024, 8192)}),
    ("esmfold2.opm_fallback_rows", "ESMFold2 outer-product-mean fallback rows", "bytes",
     lambda T, M: {s: M["esmfold2"]._opm_fallback_rows(s) for s in (512, 1024)}),
    ("boltz2.row_block", "Boltz-2 host row block", "bytes",
     lambda T, M: {b: M["boltz2"]._row_block(b) for b in (1 << 12, 1 << 16, 1 << 20)}),
    ("rfd3.attn_row_block", "RFD3 HOST neighbour row block (CPU threads) + pair-transition H chunk", "host/bytes",
     lambda T, M: [M["rfd3"]._ATTN_ROW_BLOCK, M["rfd3"]._PAIR_TRANSITION_H_CHUNK]),
    ("protenix.paircond", "whether to force a core grid at all: a multicast DEADLOCK guard", "guard",
     lambda T, M: [M["protenix"].PAIRCOND_MM_NARROW_MAX_TILES]),
    # ---- exposure: the work units the shipped pick produces, against the cores present ----
    ("exposure.sdpa_units", "work units the shipped q_chunk leaves, per shipped SDPA shape", "OCCUPANCY",
     lambda T, M: {f"{n}": _units(T, q, w) for n, q, w in SDPA_SHAPES}),
]

# (label, padded q_len, work = batch * heads) for every SDPA shape the fleet's models run.
# `work` is what the ttnn kernel multiplies by the chunk count to get its core demand.
SDPA_SHAPES = [
    ("boltz2.diffusion.token", 512, 16),
    ("boltz2.diffusion.atom", 32, 560),
    ("boltz2.trunk.tri_att", 512, 512 * 4),
    ("esmc.attn.L512", 512, 20),
    ("esmfold2.attn.L512", 512, 32),
    ("saprot.attn.L512", 512, 20),
]


def _units(T, q_len, work):
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    padded = T._padded_sdpa_len(q_len)
    shipped = T._capped_sdpa_chunk_size(q_len)
    rule = T._grid_q_chunk(q_len, work, shipped, cores)
    su = work * -(-padded // shipped)
    ru = work * -(-padded // rule)
    return {"shipped_chunk": shipped, "shipped_units": su,
            "shipped_occupancy": round(min(su, cores) / cores, 3),
            "shipped_passes": -(-su // cores),
            "rule_chunk": rule, "rule_units": ru,
            "rule_occupancy": round(min(ru, cores) / cores, 3),
            "rule_passes": -(-ru // cores)}


def run(gx: int, gy: int, allow_device: bool) -> dict:
    import tt_bio.tenstorrent as T
    set_grid(T, gx, gy)
    mods = {}
    for name, path in (("esmc", "tt_bio.esmc"), ("esmfold2", "tt_bio.esmfold2"),
                       ("boltz2", "tt_bio.boltz2"), ("rfd3", "tt_bio.rfd3.model"),
                       ("protenix", "tt_bio.protenix")):
        try:
            mods[name] = __import__(path, fromlist=["x"])
        except Exception as exc:                                    # noqa: BLE001
            mods[name] = None
            print(f"  ! {name} not importable: {str(exc)[:90]}", flush=True)
    out = {"grid": [gx, gy], "cores": gx * gy, "small_grid": T._IS_SMALL_GRID,
           "device_probes": allow_device, "probes": {}}
    for probe in PROBES:
        pid, what, cls, fn = probe[:4]
        if len(probe) > 4 and probe[4] and not allow_device:
            # These read the part's own L1/DRAM, which means opening a chip. Skipped unless the
            # caller has pinned one: a census that quietly opens a device is how an unpinned open
            # takes a card somebody else is holding, and on a Galaxy it brings up all 32.
            out["probes"][pid] = {"what": what, "cls": cls,
                                  "value": {"skipped": "needs a pinned device"}}
            continue
        try:
            val = fn(T, mods)
        except Exception as exc:                                    # noqa: BLE001
            val = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        out["probes"][pid] = {"what": what, "cls": cls, "value": val}
    return out


def join(paths: list[Path]) -> dict:
    runs = [json.loads(p.read_text()) for p in paths]
    ids = list(runs[0]["probes"])
    rows = []
    for pid in ids:
        vals = [json.dumps(r["probes"][pid]["value"], sort_keys=True) for r in runs]
        moved = len(set(vals)) > 1
        rows.append({"probe": pid, "what": runs[0]["probes"][pid]["what"],
                     "cls": runs[0]["probes"][pid]["cls"],
                     "grid_aware": moved,
                     "values": {f"{r['cores']}": r["probes"][pid]["value"] for r in runs}})
    blind_par = [r for r in rows if not r["grid_aware"] and r["cls"] == "OCCUPANCY"]
    print(f"\n  {len(rows)} pickers probed on {[r['cores'] for r in runs]} cores")
    print(f"  {sum(r['grid_aware'] for r in rows)} move with the grid, "
          f"{sum(not r['grid_aware'] for r in rows)} do not")
    print(f"  OF THOSE, {len(blind_par)} decide how many cores run an op -- the exposed class\n")
    for r in rows:
        mark = "grid-aware" if r["grid_aware"] else "GRID-BLIND"
        print(f"  {mark} [{r['cls']:10s}] {r['probe']:28s} {r['what']}")
    print("\n  The exposed class is GRID-BLIND x OCCUPANCY. Everything else is grid-blind about "
          "a resource\n  that is not the grid, which is not a defect.")
    return {"rows": rows, "grids": [r["cores"] for r in runs]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--join", nargs="*", type=Path)
    ap.add_argument("--allow-device-open", action="store_true",
                    help="run the probes that read the part's own L1/DRAM; requires a pin")
    a = ap.parse_args()
    if a.join:
        res = join(a.join)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
        return 0
    gx, gy = (int(v) for v in a.grid.split(","))
    if a.allow_device_open and not os.environ.get("TT_VISIBLE_DEVICES"):
        print("refusing: --allow-device-open with no TT_VISIBLE_DEVICES pin", file=sys.stderr)
        return 2
    res = run(gx, gy, a.allow_device_open)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"  {gx}x{gy} = {gx * gy} cores, small_grid={res['small_grid']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
