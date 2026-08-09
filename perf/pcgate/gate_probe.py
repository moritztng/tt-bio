#!/usr/bin/env python3
"""E8 -- does an output-aware, live-L1 budget change what any static program-config gate picks?

E6 root-caused a real defect: a gate that compares a circular-buffer total against a free-L1 number
without subtracting the op is own output is optimistic, because ttnn allocates the output tensor
before the program factory places a single CB. It named the generalisation -- every static gate in
tt_bio budgets against either the device-level constant or a free-L1 number with no output term.

There are exactly two such gates left: `_tri_att_qkv_l1_config` (output L1) and
`_pair_proj_program_config` (output DRAM). Both budget against
`ttnn.get_max_worker_l1_unreserved_size()`, the device constant.

This probe runs a real 298 aa fold with the production path untouched and, at every call into
either gate, evaluates the same predicate under three budgets:

  naive      -- what ships: ttnn.get_max_worker_l1_unreserved_size()
  live       -- largest_contiguous_bytes_free_per_bank at the moment of the call
  live_out   -- live minus the op is own output, per bank (E6 is fix)

and records whether the admitted/declined decision differs. The fold itself always takes the
production branch, so nothing is perturbed beyond the memory-view drain the probe adds.
"""
import argparse, json, statistics, sys, tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

MAX_SAMPLES = 12  # memory_view drains the pipeline; bound how often we ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--target", default="examples/prot300.yaml")
    ap.add_argument("--a3m", default="prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    num_cores = gx * gy
    mv0 = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    banks = int(mv0.num_banks)
    device_l1 = int(ttnn.get_max_worker_l1_unreserved_size())

    acc = defaultdict(lambda: dict(calls=0, samples=0, need=None, out_bpb=None,
                                   free=[], dec_naive=None, dec_live=set(),
                                   dec_live_out=set(), note=""))

    def free_per_bank():
        return int(ttnn.get_memory_view(dev, ttnn.BufferType.L1)
                   .largest_contiguous_bytes_free_per_bank)

    def out_bytes_per_bank(m_tiles, n_tiles, elem_bytes):
        tiles = m_tiles * n_tiles
        return -(-tiles // banks) * 1024 * elem_bytes

    # ---- gate 1: the tri-attention qkv projection, output in L1 -------------------------------
    def qkv_predicate(m_tiles, k_tiles, n_tiles, elem_bytes, budget):
        """`_tri_att_qkv_l1_config` with its L1 test taken against `budget`."""
        if k_tiles >= num_cores or m_tiles < n_tiles * 8 or n_tiles > 64:
            return "shape"
        per_core_M = next(
            (p for p in range(max(1, -(-m_tiles // num_cores)), m_tiles + 1) if m_tiles % p == 0), 0)
        if not per_core_M or -(-m_tiles // per_core_M) > num_cores:
            return "shape"
        tile = 1024 * elem_bytes
        fixed = per_core_M * n_tiles * (tile + 4096) + 128 * 1024
        per_block = (per_core_M + n_tiles) * tile
        if fixed + k_tiles * per_block > budget:
            return "l1"
        if 2 * m_tiles * n_tiles * tile > 0.6 * num_cores * device_l1:
            return "aggregate"
        return f"admit(pcm={per_core_M})"

    def qkv_need(m_tiles, k_tiles, n_tiles, elem_bytes):
        per_core_M = next(
            (p for p in range(max(1, -(-m_tiles // num_cores)), m_tiles + 1) if m_tiles % p == 0), 0)
        if not per_core_M:
            return None
        tile = 1024 * elem_bytes
        return (per_core_M * n_tiles * (tile + 4096) + 128 * 1024
                + k_tiles * (per_core_M + n_tiles) * tile)

    real_qkv = T._qkv_l1_config

    def qkv_spy(x, w, dtype):
        cfg = real_qkv(x, w, dtype)
        xs, ws = list(x.shape), list(w.shape)
        m = 1
        for d in xs[:-1]:
            m *= int(d)
        k, n = int(xs[-1]), int(ws[-1])
        key = f"qkv {tuple(int(d) for d in xs)}@{tuple(int(d) for d in ws)}"
        e = acc[key]
        e["calls"] += 1
        e["gate"] = "_tri_att_qkv_l1_config"
        if dtype != ttnn.bfloat16:
            e["dec_naive"] = "declined(dtype)"
            e["note"] = "gate only sized for bf16"
            return cfg
        if m % 32 or k % 32 or n % 32:
            # the padded-tile-count bug W5 root-caused: M here is the LOGICAL flattened row count
            e["dec_naive"] = "declined(%32 shape guard)"
            e["note"] = (f"logical M={m} is not a multiple of 32 -- the gate never reaches its L1 "
                         f"test on this shape (W5, tt-bio-l1-residency-guard-dead-in-real-folds)")
            return cfg
        mt, kt, nt = m // 32, k // 32, n // 32
        e["need"] = qkv_need(mt, kt, nt, 2)
        e["out_bpb"] = out_bytes_per_bank(mt, nt, 2)
        e["dec_naive"] = qkv_predicate(mt, kt, nt, 2, device_l1)
        if e["samples"] < MAX_SAMPLES:
            e["samples"] += 1
            f = free_per_bank()
            e["free"].append(f)
            e["dec_live"].add(qkv_predicate(mt, kt, nt, 2, f))
            e["dec_live_out"].add(qkv_predicate(mt, kt, nt, 2, f - e["out_bpb"]))
        return cfg

    # ---- gate 2: the tall pair-track projection, output in DRAM --------------------------------
    def pp_need(m_tiles, k_tiles, n_tiles, in0_block_w, elem_bytes):
        per_core_M = -(-(-(-m_tiles // num_cores)) // 5) * 5
        if per_core_M > m_tiles or -(-m_tiles // per_core_M) > num_cores:
            return None, None
        tile = 1024 * elem_bytes
        need = (2 * in0_block_w * (5 + n_tiles) * tile
                + 5 * n_tiles * (tile + 4096) + 128 * 1024)
        return need, per_core_M

    def pp_tiles(x, w):
        xs, ws = list(x.shape), list(w.shape)
        if len(xs) < 2 or len(ws) != 2:
            return None
        batch = 1
        for d in xs[:-2]:
            batch *= int(d)
        m_tiles = batch * -(-int(xs[-2]) // 32)
        k_tiles = -(-int(xs[-1]) // 32)
        n_tiles = -(-int(ws[-1]) // 32)
        if k_tiles != -(-int(ws[-2]) // 32):
            return None
        bw = max((d for d in (k_tiles, 8, 4, 2, 1)
                  if d <= (T._PAIR_PROJ_BW or 0) and k_tiles % d == 0), default=1)
        return m_tiles, k_tiles, n_tiles, bw

    real_pp = T._pair_proj_linear

    def pp_spy(x, w, ckc, dtype):
        key = f"pairproj {tuple(int(d) for d in x.shape)}@{tuple(int(d) for d in w.shape)}"
        e = acc[key]
        e["calls"] += 1
        e["gate"] = "_pair_proj_program_config"
        t = pp_tiles(x, w)
        if T._PAIR_PROJ_BW is None:
            e["dec_naive"] = "declined(feature off)"
            e["note"] = "_PAIR_PROJ_BW is None -- the tuned config is not enabled in this build"
        elif t is None or x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
            e["dec_naive"] = "declined(shape/dtype)"
        else:
            mt, kt, nt, bw = t
            need, pcm = pp_need(mt, kt, nt, bw, 2)
            if need is None or mt < num_cores or kt % bw:
                e["dec_naive"] = "declined(shape)"
            else:
                e["need"] = need
                e["out_bpb"] = 0  # the result is written to DRAM, so it takes no L1 bank space
                e["dec_naive"] = f"admit(pcm={pcm},bw={bw})" if need <= device_l1 else "declined(l1)"
                if e["samples"] < MAX_SAMPLES:
                    e["samples"] += 1
                    f = free_per_bank()
                    e["free"].append(f)
                    d = f"admit(pcm={pcm},bw={bw})" if need <= f else "declined(l1)"
                    e["dec_live"].add(d)
                    e["dec_live_out"].add(d)
        return real_pp(x, w, ckc, dtype)

    T._qkv_l1_config = qkv_spy
    T._pair_proj_linear = pp_spy

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="pcgate-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / a.target, Path(B.FIXTURES) / a.a3m)
    one_fold()

    rows = []
    for k, e in sorted(acc.items(), key=lambda kv: -kv[1]["calls"]):
        rows.append(dict(
            shape=k, gate=e.get("gate"), calls=e["calls"], samples=e["samples"],
            need_bytes=e["need"], out_bytes_per_bank=e["out_bpb"],
            free_min=min(e["free"]) if e["free"] else None,
            free_median=int(statistics.median(e["free"])) if e["free"] else None,
            dec_naive=e["dec_naive"], dec_live=sorted(e["dec_live"]),
            dec_live_out=sorted(e["dec_live_out"]),
            changed=bool(e["dec_live_out"]) and any(d != e["dec_naive"] for d in e["dec_live_out"]),
            note=e["note"]))
    out = dict(model=a.model, n_aa=298, hardware=meta.get("hardware"),
               device_l1_unreserved=device_l1,
               l1_total_bytes_per_bank=int(mv0.total_bytes_per_bank),
               l1_banks=banks, grid=[gx, gy],
               any_decision_changed=any(r["changed"] for r in rows), rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
