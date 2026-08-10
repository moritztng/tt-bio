#!/usr/bin/env python3
"""y-permute-flip deliverable 3: esmfold2, plus the L1 co-residency ladder re-taken.

esmfold2 builds two `TriangleMultiplication` instances of its own (`PairUpdateBlock` at
esmfold2.py:138/141 and `MSAEncoderBlock` at 1102/1103) as well as reaching the shared one, and its
pair shapes are the least like protenix's. The census wraps `_channel_move` and records every call's
shape, destination buffer type and gate verdict in a real fold, at one length INSIDE the shipped L1
window (298 aa, prot300) and one OUTSIDE it (117 aa, prot.yaml).

The co-residency ladder re-takes the free-L1-per-bank figure with the merged L1 consumers and the
kernel's circular buffers live together, rather than inheriting it.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import torch
import ttnn

OUT = Path(__file__).resolve().parent
R: dict = {}
OUTPATH = None


def flush():
    if OUTPATH:
        Path(OUTPATH).write_text(json.dumps(R, indent=2, default=str) + "\n")


def load():
    return [round(v, 2) for v in os.getloadavg()]


def med(v):
    return sorted(v)[len(v) // 2]


_L1F = ("total_bytes_per_bank", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank",
        "largest_contiguous_bytes_free_per_bank", "num_banks")


def l1_stats(dev):
    try:
        v = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    except Exception as e:                                            # noqa: BLE001
        return {"error": repr(e)[:160]}
    return {k: int(getattr(v, k)) for k in _L1F if hasattr(v, k)}


def ladder(dev, RP):
    """Add each L1 consumer in turn and read the allocator; the kernel runs at the bottom rung."""
    rows = [{"live": "nothing", **l1_stats(dev)}]
    keep = []

    def add(name, shape):
        t = ttnn.from_torch(torch.zeros(*shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        keep.append(t)
        rows.append({"live": name, "mb": round(t.volume() * 2 / 1e6, 2), **l1_stats(dev)})
        return t

    try:
        add("one 48.82 MB pair tensor [1,298,320,256]", (1, 298, 320, 256))
        add("two of them (the merged L1-output path)", (1, 298, 320, 256))
        chunk = add("+ the trimul's own [1,298,298,64] chunk", (1, 298, 298, 64))
        RP.set_enabled(True)
        y = RP.reblock_permute(chunk, ttnn.L1_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        keep.append(y)
        rows.append({"live": "+ the kernel's output tensor, kernel RAN", "kernel_ran": True,
                     **l1_stats(dev)})
        RP.set_enabled(False)
    except Exception as e:                                            # noqa: BLE001
        rows.append({"live": "REFUSED", "error": repr(e)[:220]})
        RP.set_enabled(False)
    for t in keep:
        try:
            ttnn.deallocate(t)
        except Exception:                                             # noqa: BLE001
            pass
    return rows


def retake(dev, RP, cells, reps=40):
    """Re-take disputed window cells with more reps. A C=32 wall at the C=64 wall's value is not a
    window finding, it is a contaminated measurement, and the way to tell is to measure again."""
    rows = []
    for (N, C, bt) in cells:
        g = torch.Generator().manual_seed(1)
        t = torch.randn(1, N, N, C, generator=g).to(torch.bfloat16)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG if bt == "l1"
                            else ttnn.DRAM_MEMORY_CONFIG)
        mcfg = ttnn.L1_MEMORY_CONFIG if bt == "l1" else ttnn.DRAM_MEMORY_CONFIG

        def run(fn, n):
            best = None
            for _ in range(5):                     # 5 independent throughput blocks, take the best
                outs = [fn() for _ in range(2)]
                ttnn.synchronize_device(dev)
                for o in outs:
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                outs = [fn() for _ in range(n)]
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / n * 1e6
                for o in outs:
                    ttnn.deallocate(o)
                best = dt if best is None else min(best, dt)
            return best

        n = 4 if bt == "l1" else 8
        RP.set_enabled(True)
        k = run(lambda: RP.reblock_permute(x, mcfg), n)
        RP.set_enabled(False)
        st = run(lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mcfg), n)
        rows.append({"N": N, "C": C, "buffer": bt, "kernel_us": round(k, 2),
                     "stock_us": round(st, 2), "ratio_stock_over_kernel": round(st / k, 3),
                     "stat": "best of 5 throughput blocks", "load": load()})
        print("retake:", rows[-1], flush=True)
        ttnn.deallocate(x)
    return rows


def main() -> int:
    global OUTPATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "flip_esmfold2.json"))
    a = ap.parse_args()
    OUTPATH = a.out

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    R.update({"wheel": "0.67.4", "host": "qb1 (tt-quietbox)",
              "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1")),
              "baseline": "origin/main 7224ff34 + the constant refactor",
              "targets": {}})
    flush()

    # --- the census: every _channel_move call, its shape, its destination, the gate's verdict -----
    census: dict = {}
    _orig_cm = T._channel_move

    def _counting_cm(chunk, memory_config):
        key = (tuple(int(d) for d in chunk.shape), str(memory_config.buffer_type),
               bool(RP.eligible(chunk, memory_config)))
        census[key] = census.get(key, 0) + 1
        return _orig_cm(chunk, memory_config)

    T._channel_move = _counting_cm

    trimul_throws = []
    _orig_tm = T.TriangleMultiplication.__call__

    def _wrapped_tm(self, x, mask=None):
        try:
            return _orig_tm(self, x, mask)
        except Exception as e:                                        # noqa: BLE001
            trimul_throws.append(repr(e)[:300])
            raise

    T.TriangleMultiplication.__call__ = _wrapped_tm

    from tt_baseline import build_fold
    msa_dir = Path.home() / "w6_gate_msa"

    dev = None
    for label, yml, a3m in (("298aa_inside_window", "examples/prot300.yaml",
                             "scripts/gpu_vs_tt/fixtures/prot300.a3m"),
                            ("117aa_outside_window", "examples/prot.yaml",
                             "scripts/gpu_vs_tt/fixtures/prot117.a3m")):
        t0 = time.perf_counter()
        try:
            one_fold, meta, *_rest = build_fold("esmfold2", msa_dir, REPO / yml, REPO / a3m)
        except Exception as e:                                        # noqa: BLE001
            R["targets"][label] = {"error": repr(e)[:400]}
            flush()
            continue
        print(f"[{label}] model ready in {time.perf_counter() - t0:.1f}s", flush=True)
        if dev is None:
            dev = T.get_device()
            g = dev.compute_with_storage_grid_size()
            R["grid"] = {"compute_grid_main": list(T.COMPUTE_GRID_MAIN), "device_grid": [g.x, g.y]}
            R["l1_ladder"] = ladder(dev, RP)
            print("ladder:", R["l1_ladder"], flush=True)
            flush()
            R["window_retake"] = retake(dev, RP, [(288, 32, "l1"), (352, 32, "l1"),
                                                  (298, 32, "l1"), (320, 32, "l1"),
                                                  (298, 64, "l1")])
            flush()

        entry: dict = {"yaml": yml, "rounds": []}
        census.clear()
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        cb, mb = one_fold()
        entry["cold_base_s"] = round(cb, 3)
        entry["cold_base_calls"] = RP.STATS[0] - n0
        entry["census_base_arm"] = {f"{list(k[0])}|{k[1]}|eligible={k[2]}": v
                                    for k, v in census.items()}
        census.clear()
        RP.set_enabled(True)
        n0 = RP.STATS[0]
        cw, mw = one_fold()
        RP.set_enabled(False)
        entry["cold_wire_s"] = round(cw, 3)
        entry["cold_wire_calls_served"] = RP.STATS[0] - n0
        entry["census_wire_arm"] = {f"{list(k[0])}|{k[1]}|eligible={k[2]}": v
                                    for k, v in census.items()}
        entry["plddt_base"] = mb.get("plddt")
        entry["plddt_wire"] = mw.get("plddt")
        entry["load_after_cold"] = load()
        print(f"[{label}] census:", entry["census_wire_arm"], flush=True)
        print(f"[{label}] served:", entry["cold_wire_calls_served"], flush=True)
        R["targets"][label] = entry
        flush()

        # Only pay for a paired A/B where the kernel actually serves calls.
        if entry["cold_wire_calls_served"] > 0:
            for r in range(a.rounds):
                RP.set_enabled(False)
                tb, m_b = one_fold()
                lb = load()
                RP.set_enabled(True)
                n1 = RP.STATS[0]
                tw, m_w = one_fold()
                served = RP.STATS[0] - n1
                RP.set_enabled(False)
                row = {"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
                       "delta_ms": round((tb - tw) * 1e3, 1), "calls_served_wire": served,
                       "plddt_base": m_b.get("plddt"), "plddt_wire": m_w.get("plddt"),
                       "l1_out_refused_n": len(T._L1_OUT_REFUSED),
                       "trimul_throws_n": len(trimul_throws),
                       "load_base": lb, "load_wire": load()}
                entry["rounds"].append(row)
                print(f"[{label}] ab:", row, flush=True)
                flush()
            # An A/A pair on the same target, so this model has its own floor too.
            RP.set_enabled(False)
            ta, _ = one_fold()
            tb2, _ = one_fold()
            entry["aa"] = {"a_s": round(ta, 4), "b_s": round(tb2, 4),
                           "apparent_delta_ms": round((ta - tb2) * 1e3, 1), "load": load()}
            print(f"[{label}] aa:", entry["aa"], flush=True)
            d = [x["delta_ms"] for x in entry["rounds"]]
            entry["summary"] = {"delta_ms_paired_mean": round(sum(d) / len(d), 1),
                                "per_round": d,
                                "n_positive": sum(1 for v in d if v > 0),
                                "ratio": round(med([x["base_s"] for x in entry["rounds"]])
                                               / med([x["wire_s"] for x in entry["rounds"]]), 5)}
        else:
            entry["summary"] = {"verdict": "no eligible calls -- no A/B taken, the census is the "
                                           "answer"}
        entry["trimul_throws"] = list(trimul_throws)
        entry["l1_out_refused"] = sorted(str(k)[:160] for k in T._L1_OUT_REFUSED)
        R["targets"][label] = entry
        flush()

    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
