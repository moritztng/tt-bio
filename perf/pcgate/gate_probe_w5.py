#!/usr/bin/env python3
"""E8, part 2 -- the one gate left whose budget is a SEARCH BOUND rather than a veto.

W5 (`wk/perfwar-l1-residency-298aa`) revives the tri-attention L1-residency gate that main leaves
dead, and its `_l1_resident_row_block` searches the row block downwards until
`_l1_resident_matmul_config` admits it. There the budget does not merely veto a fixed config, it
picks one -- so an optimistic budget picks a LARGER row block than actually fits, which is the
shape of bug E6 found in the chunked Transition.

W5 already carries an output term (`2 * tile` for the L1-resident result plus the per-core share of
the sibling result), derived independently from a crash rather than from E6. What it does not carry
is E6's other half: the base is still `ttnn.get_max_worker_l1_unreserved_size()`, the device
constant, which on this card is 1532416 B -- larger than the whole 1461760 B per-bank L1.

This probe runs W5's code as the production path on a real 298 aa fold and, at every call into the
gate, re-evaluates the same search with the base replaced by the LIVE
`largest_contiguous_bytes_free_per_bank`, recording whether the chosen row block moves.
"""
import argparse, json, statistics, sys, tempfile
from collections import defaultdict
from pathlib import Path

W5 = Path(__file__).resolve().parent / "_w5ref"
sys.path.insert(0, str(W5))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

MAX_SAMPLES = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    assert str(W5) in T.__file__, f"probe is not running W5 code: {T.__file__}"

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    num_cores = gx * gy
    device_l1 = int(ttnn.get_max_worker_l1_unreserved_size())

    def free_per_bank():
        return int(ttnn.get_memory_view(dev, ttnn.BufferType.L1)
                   .largest_contiguous_bytes_free_per_bank)

    def admits(m_tiles, k_tiles, n_tiles, elem_bytes, full_k, budget):
        """W5's `_l1_resident_matmul_config` predicate with its L1 test taken against `budget`."""
        if k_tiles >= num_cores or m_tiles < n_tiles * 8 or n_tiles > 64:
            return False
        per_core_M = next(
            (p for p in range(max(1, -(-m_tiles // num_cores)), m_tiles + 1) if m_tiles % p == 0), 0)
        if not per_core_M or -(-m_tiles // per_core_M) > num_cores:
            return False
        in0_block_w = k_tiles if full_k else 1
        tile = 1024 * elem_bytes
        fixed = per_core_M * n_tiles * (2 * tile + 4096) + 128 * 1024
        fixed += m_tiles * n_tiles * tile // num_cores
        if fixed + in0_block_w * (per_core_M + n_tiles) * tile >= budget:
            return False
        if 2 * m_tiles * n_tiles * tile > 0.6 * num_cores * device_l1:
            return False
        return True

    def row_block(rows, cols, k_tiles, n_tiles, elem_bytes, budget):
        col_tiles = -(-cols // 32)
        for r in range(rows, 0, -1):
            if admits(r * col_tiles, k_tiles, n_tiles, elem_bytes, True, budget):
                return r
        return 0

    acc = defaultdict(lambda: dict(calls=0, samples=0, free=[], prod=set(), live=set(), note=""))

    real_rb = T._tri_att_row_block

    def rb_spy(x, w, dtype):
        r = real_rb(x, w, dtype)
        key = f"tri_att_row_block {tuple(int(d) for d in x.shape)}@{tuple(int(d) for d in w.shape)}"
        e = acc[key]
        e["calls"] += 1
        e["gate"] = "_l1_resident_row_block (SEARCH bound)"
        e["prod"].add(r)
        if e["samples"] < MAX_SAMPLES and dtype == ttnn.bfloat16:
            e["samples"] += 1
            f = free_per_bank()
            e["free"].append(f)
            if T._l1_resident_linear_config(x, w, dtype) is not None:
                e["live"].add(0)  # whole tensor already fits; blocking would only add dispatches
                e["note"] = "whole-tensor path, no row block"
            else:
                rows, cols = int(x.shape[0]), int(x.shape[1])
                _m, kt, nt = T._linear_tiles(x, w)
                e["live"].add(row_block(rows, cols, kt, nt, 2, f))
        return r

    real_lin = T._l1_resident_linear

    def lin_spy(x, w, dtype, ckc, full_k=False):
        cfg = T._l1_resident_linear_config(x, w, dtype, full_k=full_k)
        key = f"l1_resident_linear {tuple(int(d) for d in x.shape)}@{tuple(int(d) for d in w.shape)}"
        e = acc[key]
        e["calls"] += 1
        e["gate"] = "_l1_resident_matmul_config (VETO)"
        e["prod"].add(cfg is not None)
        if e["samples"] < MAX_SAMPLES and dtype == ttnn.bfloat16:
            e["samples"] += 1
            f = free_per_bank()
            e["free"].append(f)
            mt, kt, nt = T._linear_tiles(x, w)
            e["live"].add(admits(mt, kt, nt, 2, full_k, f))
        return real_lin(x, w, dtype, ckc, full_k=full_k)

    T._tri_att_row_block = rb_spy
    T._l1_resident_linear = lin_spy

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="pcgate-w5-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml", Path(B.FIXTURES) / "prot300.a3m")
    one_fold()

    rows = []
    for k, e in sorted(acc.items(), key=lambda kv: -kv[1]["calls"]):
        rows.append(dict(shape=k, gate=e.get("gate"), calls=e["calls"], samples=e["samples"],
                         free_min=min(e["free"]) if e["free"] else None,
                         free_median=int(statistics.median(e["free"])) if e["free"] else None,
                         production=sorted(str(v) for v in e["prod"]),
                         live_budget=sorted(str(v) for v in e["live"]),
                         changed=bool(e["live"]) and sorted(str(v) for v in e["live"]) !=
                                 sorted(str(v) for v in e["prod"]),
                         note=e["note"]))
    out = dict(model=a.model, n_aa=298, branch="wk/perfwar-l1-residency-298aa",
               device_l1_unreserved=device_l1, grid=[gx, gy],
               any_decision_changed=any(r["changed"] for r in rows), rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
