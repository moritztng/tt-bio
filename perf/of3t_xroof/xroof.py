"""Settle the crossing RATE: `of3t-xsplit`'s 1.265/2.013 GB/s against `of3t-stepqb2`'s 2.15/2.77.

Two rows measured the same crossings on the same board and reported rates 1.70x and 1.38x apart.
Neither row's SECONDS are in doubt -- three independent runs agree on 0.712-0.722 s per
`to_torch` of the score block. What differs is the BYTE COUNTER, and this instrument runs both
counters over the same call, in one process, so the comparison is not a cross-run one.

    counter A  PCIe bytes    what actually crosses the link: the device-side tensor, once.
    counter B  bwprof.Rec    `perf/of3t_bwattrib/bwprof.py:134-142` -- the output tensor's bytes
                             PLUS every ttnn argument's bytes, so a crossing is charged for both
                             of its ends; and `_itemsize` (`:80-87`) tests lowercase substrings
                             against `str(ttnn.float32) == "DataType.FLOAT32"`, so an fp32 device
                             tensor is charged 2 B/element instead of 4.

Counter B is imported from bwprof rather than re-implemented, so the claim is made with the
instrument's own code on the instrument's own objects.

And the crossing is split into its two real terms rather than left as one number, because a roof
over a PCIe link and a roof over a single-threaded host tile-shuffle are different roofs:

    to_torch(t)   = ttnn.from_device(t)            DMA, still tiled      -> the LINK
                  + host_tensor.to_torch()         untilize on the host  -> the HOST
    from_torch(h) = ttnn.from_torch(h, device=None)  tilize on the host  -> the HOST
                  + ttnn.to_device(t)                DMA                 -> the LINK

    xroof.py --out perf/of3t_xroof/out/xroof_384.json
"""
import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402

NS = time.perf_counter_ns
SCORE = (384, 4, 384, 384)


def _sysfs(dev_index=0):
    """Board class, serial and PCIe link, with no device opened. The class node is
    `tenstorrent!0`; `tenstorrent/0` is neither a file nor an error and every read returns
    None (of3t-stepqb2 recorded a board of `null` that way on its first try)."""
    base = Path(f"/sys/class/tenstorrent/tenstorrent!{dev_index}")
    r = {}
    for k, p in (("board_type", "tt_card_type"), ("serial", "tt_serial"),
                 ("aiclk_idle", "tt_aiclk"),
                 ("link_speed", "device/current_link_speed"),
                 ("link_width", "device/current_link_width"),
                 ("link_speed_max", "device/max_link_speed"),
                 ("link_width_max", "device/max_link_width")):
        try:
            r[k] = (base / p).read_text().strip()
        except Exception:                                                # noqa: BLE001
            r[k] = None
    return r


def _mem_avail_gib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / (1024 * 1024)
    return None


def _med(xs):
    return statistics.median(xs)


class Timed:
    """Median-of-n with the queue drained either side, plus the load at each repetition."""

    def __init__(self, ttnn, dev, reps):
        self.ttnn, self.dev, self.reps = ttnn, dev, reps

    def __call__(self, fn, *, keep=False):
        s, la, res = [], [], None
        self.ttnn.synchronize_device(self.dev)
        for _ in range(self.reps):
            t0 = NS()
            r = fn()
            self.ttnn.synchronize_device(self.dev)
            s.append((NS() - t0) / 1e9)
            la.append(round(os.getloadavg()[0], 2))
            if keep and res is None:
                res = r
            else:
                del r
                gc.collect()
        return {"s": [round(x, 6) for x in s], "median_s": round(_med(s), 6),
                "min_s": round(min(s), 6), "loadavg1": la}, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rep = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_holder": os.environ.get("TT_BIO_LEASE_HOLDER"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()],
        "mem_available_gib_start": round(_mem_avail_gib(), 2),
        "nproc": os.cpu_count(),
        "board": _sysfs()}, "config": {"reps": a.reps, "score_shape": list(SCORE)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(rep, indent=1, default=str))   # noqa: E731
    dump()

    with during() as clk:
        try:
            import torch
            import ttnn
            from tt_bio.tenstorrent import get_device
            from perf.of3t_bwattrib import bwprof            # counter B, their code

            # --- the byte counters, both of them ------------------------------------------
            #
            # A: the device-side tensor's bytes, once -- what the DMA engine moves.
            # B: bwprof's own Rec.bump, on the same (out, args) the real call saw.
            REC = bwprof.Rec()

            def counter_b(name, out_obj, args):
                _key, nb = REC.bump(name, out_obj, args)
                return nb

            rep["counter_b_source"] = {
                "file": "perf/of3t_bwattrib/bwprof.py",
                "bump": "Rec.bump:129-142 -- nb = output bytes + every ttnn argument's bytes",
                "itemsize": "_itemsize:81-87 -- lowercase substring test",
                "itemsize_on_this_build": {
                    s: bwprof._itemsize(d) for s, d in (
                        (str(ttnn.float32), ttnn.float32),
                        (str(ttnn.bfloat16), ttnn.bfloat16),
                        (str(ttnn.bfloat8_b), ttnn.bfloat8_b),
                        (str(ttnn.uint32), ttnn.uint32),
                        (str(torch.float32), torch.float32),
                        (str(torch.bfloat16), torch.bfloat16))},
                "true_itemsize": {"DataType.FLOAT32": 4, "DataType.BFLOAT16": 2,
                                  "DataType.BFLOAT8_B": 1.0625, "DataType.UINT32": 4,
                                  "torch.float32": 4, "torch.bfloat16": 2}}
            dump()

            dev = get_device()
            rep["env"]["arch"] = str(dev.arch())
            T = Timed(ttnn, dev, a.reps)
            dump()

            # =============================================================================
            # 1. CONVENTION -- the production shape, both counters, one call.
            # =============================================================================
            n_elem = 1
            for d in SCORE:
                n_elem *= d
            pcie_gb_f32 = n_elem * 4 / 1e9

            ht = torch.randn(SCORE, dtype=torch.float32)
            t_tile = ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.float32)
            ttnn.synchronize_device(dev)

            conv = {}

            def leg(name, fn, pcie_gb, *, bump_name):
                r, kept = T(fn, keep=True)
                nb_b = counter_b(bump_name, kept, _BUMP_ARGS[name])
                r["pcie_gb_per_call"] = round(pcie_gb, 6)
                r["pcie_gb_s"] = round(pcie_gb / r["median_s"], 4)
                r["bwprof_gb_per_call"] = round(nb_b / 1e9, 6)
                r["bwprof_gb_s"] = round(nb_b / 1e9 / r["median_s"], 4)
                r["inflation_x"] = round((nb_b / 1e9) / pcie_gb, 4)
                conv[name] = r
                del kept
                gc.collect()

            _BUMP_ARGS = {
                "A_to_torch_from_TILE": (t_tile,),
                "D_from_torch_to_TILE": (ht,),
            }
            leg("A_to_torch_from_TILE", lambda: ttnn.to_torch(t_tile), pcie_gb_f32,
                bump_name="to_torch")
            leg("D_from_torch_to_TILE",
                lambda: ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT, device=dev,
                                        dtype=ttnn.float32),
                pcie_gb_f32, bump_name="from_torch")
            rep["convention"] = conv
            dump()

            # =============================================================================
            # 2. SPLIT -- the link term and the host term, each on its own line.
            # =============================================================================
            split = {}

            def put(k, r, pcie_gb):
                r["pcie_gb_per_call"] = round(pcie_gb, 6)
                r["gb_s_at_pcie_bytes"] = round(pcie_gb / r["median_s"], 4) if pcie_gb else None
                split[k] = r

            r, host_tile = T(lambda: ttnn.from_device(t_tile), keep=True)
            put("G_from_device_DMA_only", r, pcie_gb_f32)

            r, _ = T(lambda: ttnn.to_torch(host_tile))
            put("H_host_untilize_no_DMA", r, 0.0)

            r, host_built = T(lambda: ttnn.from_torch(ht, layout=ttnn.TILE_LAYOUT,
                                                      dtype=ttnn.float32), keep=True)
            put("I_host_tilize_no_DMA", r, 0.0)

            r, _ = T(lambda: ttnn.to_device(host_built, dev))
            put("J_to_device_DMA_only", r, pcie_gb_f32)

            # xsplit's ROW_MAJOR legs, re-taken here so its table and this one share a process
            r, _ = T(lambda: ttnn.to_layout(t_tile, ttnn.ROW_MAJOR_LAYOUT))
            put("B_device_to_layout_ROW_MAJOR", r, 0.0)
            t_rm = ttnn.to_layout(t_tile, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(dev)
            r, _ = T(lambda: ttnn.to_torch(t_rm))
            put("C_to_torch_from_ROW_MAJOR", r, pcie_gb_f32)
            r, _ = T(lambda: ttnn.from_torch(ht, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                             dtype=ttnn.float32))
            put("E_from_torch_to_ROW_MAJOR", r, pcie_gb_f32)

            A = conv["A_to_torch_from_TILE"]["median_s"]
            D = conv["D_from_torch_to_TILE"]["median_s"]
            G = split["G_from_device_DMA_only"]["median_s"]
            H = split["H_host_untilize_no_DMA"]["median_s"]
            I = split["I_host_tilize_no_DMA"]["median_s"]
            J = split["J_to_device_DMA_only"]["median_s"]
            split["residual"] = {
                "out_leg_A_minus_G_plus_H_s": round(A - (G + H), 6),
                "out_leg_residual_pct": round(100 * (A - (G + H)) / A, 3),
                "ret_leg_D_minus_I_plus_J_s": round(D - (I + J), 6),
                "ret_leg_residual_pct": round(100 * (D - (I + J)) / D, 3),
                "note": "A and D are the shipped crossings; G/J are the DMA alone and H/I the "
                        "host layout pass alone. A small residual is what says the split holds."}
            split["link_roof_gb_s"] = {
                "device_to_host_DMA_only": round(pcie_gb_f32 / G, 4),
                "host_to_device_DMA_only": round(pcie_gb_f32 / J, 4)}
            split["host_layout_gb_s"] = {
                "untilize": round(pcie_gb_f32 / H, 4),
                "tilize": round(pcie_gb_f32 / I, 4)}
            rep["split"] = split
            dump()

            del host_tile, host_built, t_rm
            gc.collect()

            # =============================================================================
            # 3. ROOF -- a size and dtype sweep over the shapes the backward actually crosses.
            # =============================================================================
            SHAPES = [
                ("[128] bf16",            (128,),                  ttnn.bfloat16, torch.bfloat16),
                ("[1,64,384,128] bf16",   (1, 64, 384, 128),       ttnn.bfloat16, torch.bfloat16),
                ("[384,384,128] bf16",    (384, 384, 128),         ttnn.bfloat16, torch.bfloat16),
                ("[1,384,384,128] bf16",  (1, 384, 384, 128),      ttnn.bfloat16, torch.bfloat16),
                ("[1,16,384,384] bf16",   (1, 16, 384, 384),       ttnn.bfloat16, torch.bfloat16),
                ("[96,4,384,384] f32",    (96, 4, 384, 384),       ttnn.float32,  torch.float32),
                ("[384,4,384,384] f32",   SCORE,                   ttnn.float32,  torch.float32),
            ]
            roof = []
            for label, shp, tdt, pdt in SHAPES:
                ne = 1
                for d in shp:
                    ne *= d
                gb = ne * (4 if tdt is ttnn.float32 else 2) / 1e9
                try:
                    h = torch.zeros(shp, dtype=pdt)
                    t = ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=tdt)
                    ttnn.synchronize_device(dev)
                    rd, _ = T(lambda: ttnn.to_torch(t))
                    rdma, _ = T(lambda: ttnn.from_device(t))
                    ru, _ = T(lambda: ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev,
                                                      dtype=tdt))
                    row = {"shape_class": label, "shape": list(shp), "pcie_gb": round(gb, 6),
                           "to_torch_s": rd["median_s"],
                           "to_torch_gb_s": round(gb / rd["median_s"], 4),
                           "from_device_dma_s": rdma["median_s"],
                           "from_device_dma_gb_s": round(gb / rdma["median_s"], 4),
                           "from_torch_s": ru["median_s"],
                           "from_torch_gb_s": round(gb / ru["median_s"], 4),
                           "loadavg1": rd["loadavg1"][-1]}
                    # the 3.0x production family: an fp32 host tensor landing as bf16 on device
                    if tdt is ttnn.bfloat16:
                        h32 = h.to(torch.float32)
                        rc, _ = T(lambda: ttnn.from_torch(h32, layout=ttnn.TILE_LAYOUT,
                                                          device=dev, dtype=ttnn.bfloat16))
                        row["from_torch_f32src_s"] = rc["median_s"]
                        row["from_torch_f32src_gb_s"] = round(gb / rc["median_s"], 4)
                        del h32
                    roof.append(row)
                    del t, h
                    gc.collect()
                except Exception:                                        # noqa: BLE001
                    roof.append({"shape_class": label, "error": traceback.format_exc()[-800:]})
                dump()
            rep["roof"] = roof
            dump()

            # =============================================================================
            # 4. LIMITER -- is the isolated micro-benchmark paying for a fresh host allocation?
            #    xsplit read 1.187 GB/s isolated against 1.265 live, and a benchmark slower than
            #    production needs its limiter named.
            # =============================================================================
            lim = {}
            warm = [torch.empty(SCORE, dtype=torch.float32) for _ in range(2)]
            for w in warm:
                w.fill_(0)
            del warm
            gc.collect()
            r, _ = T(lambda: ttnn.to_torch(t_tile))
            lim["to_torch_after_touching_two_same_sized_buffers"] = r
            lim["to_torch_after_touching_gb_s"] = round(pcie_gb_f32 / r["median_s"], 4)
            t_alloc0 = NS()
            fresh = torch.empty(SCORE, dtype=torch.float32)
            fresh.fill_(1.0)
            lim["fresh_906mb_alloc_and_first_touch_s"] = round((NS() - t_alloc0) / 1e9, 6)
            del fresh
            gc.collect()
            t_cp0 = NS()
            src = torch.ones(SCORE, dtype=torch.float32)
            dst = src.clone()
            lim["host_memcpy_906mb_s"] = round((NS() - t_cp0) / 1e9, 6)
            lim["host_memcpy_gb_s"] = round(2 * pcie_gb_f32 / lim["host_memcpy_906mb_s"], 4)
            del src, dst
            gc.collect()
            rep["limiter"] = lim
            dump()

            rep["ok"] = True
        except Exception:                                                # noqa: BLE001
            rep["error"] = traceback.format_exc()[-4000:]
            rep["ok"] = False
        finally:
            rep["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
            rep["env"]["mem_available_gib_end"] = round(_mem_avail_gib(), 2)
            dump()
    rep["env"]["aiclk_during"] = clk.summary()
    keys = list(clk.summary())
    rep["env"]["aiclk_line"] = clk.line(keys[0]) if keys else clk.line()
    rep["env"]["loadavg_during"] = clk.load[-40:]
    dump()
    print(json.dumps({"ok": rep.get("ok"), "aiclk": rep["env"].get("aiclk_line"),
                      "convention": rep.get("convention"),
                      "split_roof": rep.get("split", {}).get("link_roof_gb_s"),
                      "residual": rep.get("split", {}).get("residual")},
                     indent=1, default=str))
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
