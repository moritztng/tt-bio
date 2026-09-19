#!/usr/bin/env python3
"""Does the recipe crop fit? Measured on the SHIPPED forward, at full pairformer depth.

The campaign inherited "384 tokens does not fit in card DRAM" from a table
(`state/ptxft-build.md` MEMORY) that measured a differentiable TWIN of the pairformer, not
the module the fold runs. `state/ptx/LEDGER.md` K1 leaves it open in both directions and
tells this row to inherit neither story, so the arm here is the shipped
`tenstorrent.Pairformer` on the real remapped protenix-v2 weights, at the depth the
checkpoint ships (48) and at the widths it ships (c_z 256, 8 triangle heads of 32).

The forward is UNTAPED, which is what the refused allocation was measured under: under
per-block checkpointing the block forward carries no tape, so what fails is one block own
working set. That is why this arm can run today without waiting on `ptx-fastpath`.

Per size it reports:

  verdict       PASS, or OOM classified by the ENGINE own `size_limits.classify_device_oom`
                so the refusal is read as shape / full / fragmented rather than guessed.
  peak          device DRAM and L1 high-water over every `dram_peak` tag the shipped
                modules already carry (trimul, tri_att, transition, per block), plus the
                allocator own `total_bytes_allocated_per_bank`.
  blocks        the per-block DRAM curve, which is what separates "one block is too big"
                from "the chain accumulates".
  fragmentation the smallest `largest_contiguous_bytes_free_per_bank` and the allocator
                free-block count at the high-water. `of3-1024aa-oom-allocation-count-not-size`
                is the governing warning: an OOM can be a count problem rather than a size
                problem and the fix differs completely. `MemoryView.block_table` does not
                convert through this wheel, so the free-block census is the observable
                (same instrument and same limitation as `perf/b2z2_size_ladder/ladder.py`).

Masks are None at every size run here, which is not a simplification: 256 / 384 / 512 are
all multiples of the token bucket, so the shipped trunk passes None too at these lengths.

    trunk_profile.py --sizes 256,384,512 --blocks 48 --out out/trunk_qb1c1.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during  # noqa: E402

CKPT = Path(os.environ.get("PTX_V2_CKPT", "/home/ttuser/protenix_ckpt/protenix-v2.pt"))


def _free_block_census(path: Path) -> dict:
    """Free-block count and largest class from the allocator own detailed report."""
    try:
        txt = path.read_text()
    except OSError:
        return {}
    dram = txt.split(",L1", 1)[0]
    blocks = re.findall(r"Size class \d+: \(\d+ - \d+\) blocks: (.*)", dram)
    n = sum(len([b for b in line.split() if b.strip()]) for line in blocks)
    return {"dram_free_blocks": n}


class Probe:
    """Stands in for `tenstorrent.dram_peak`, keeping its contract and adding L1 and counts."""

    def __init__(self, ttnn, dev, reports: Path):
        self.ttnn, self.dev, self.reports = ttnn, dev, reports
        self.reset()

    def reset(self):
        self.samples = 0
        self.peak = {"dram": 0, "l1": 0}
        self.peak_tag = {"dram": None, "l1": None}
        self.minfree = {"dram": None, "l1": None}
        self.census = {}
        self.trace = []
        self.probe_s = 0.0

    def read(self, key="dram"):
        bt = self.ttnn.BufferType.DRAM if key == "dram" else self.ttnn.BufferType.L1
        mv = self.ttnn.get_memory_view(self.dev, bt)
        return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

    def __call__(self, tag=None):
        if tag is None:
            return self.peak["dram"]
        t0 = time.perf_counter()
        try:
            for key, bt in (("dram", self.ttnn.BufferType.DRAM), ("l1", self.ttnn.BufferType.L1)):
                mv = self.ttnn.get_memory_view(self.dev, bt)
                used = int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)
                lcf = mv.largest_contiguous_bytes_free_per_bank
                lcf = min(lcf) if isinstance(lcf, (list, tuple)) else int(lcf)
                if self.minfree[key] is None or lcf < self.minfree[key]:
                    self.minfree[key] = lcf
                if used > self.peak[key]:
                    self.peak[key], self.peak_tag[key] = used, tag
                    if key == "dram":
                        self.ttnn.dump_device_memory_state(self.dev, "ptxcrop_")
                        self.census = _free_block_census(
                            self.reports / "ptxcrop_detailed_memory_usage.csv")
                        self.census["largest_contiguous_free_per_bank_b"] = lcf
                if key == "dram" and tag.startswith("pairformer "):
                    self.trace.append([tag, used, lcf])
            self.samples += 1
        except Exception:
            pass                                   # a diagnostic must never break a fold
        self.probe_s += time.perf_counter() - t0
        return self.peak["dram"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="256,384,512")
    ap.add_argument("--blocks", type=int, default=0, help="0 = whatever the checkpoint ships")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device, Pairformer
    from tt_bio import protenix_weights as PW
    from tt_bio.protenix import n_blocks
    from tt_bio import size_limits as SL
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    ttnn.device.EnableMemoryReports()
    reports = REPO / "generated" / "reports"
    probe = Probe(ttnn, dev, reports)
    TT.dram_peak = probe
    for name, mod in list(sys.modules.items()):
        if name.startswith("tt_bio.") and getattr(mod, "dram_peak", None) is not None:
            mod.dram_peak = probe

    mvd = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    mvl = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    g = dev.compute_with_storage_grid_size()
    out = {
        "doc": __doc__,
        "env": {
            "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
            "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
            "tt_bio_file": _TB.__file__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg": os.getloadavg(),
            "dram_total_b": int(mvd.total_bytes_per_bank) * int(mvd.num_banks),
            "dram_banks": int(mvd.num_banks),
            "dram_bytes_per_bank": int(mvd.total_bytes_per_bank),
            "l1_total_b": int(mvl.total_bytes_per_bank) * int(mvl.num_banks),
            "ckpt": str(args.ckpt),
            "fast_mode": TT._FAST_MODE,
        },
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    t0 = time.perf_counter()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    nb_ckpt = n_blocks(sd, "pairformer_stack")
    nb = args.blocks or nb_ckpt
    c_z = int(sd["layernorm_z_cycle.weight"].shape[0])
    comb, n_par = {}, 0
    for i in range(nb):
        blk = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in sd.items()
               if k.startswith(f"pairformer_stack.blocks.{i}.")}
        for k, v in PW.remap_pairformer_block(blk).items():
            comb[f"layers.{i}.{k}"] = v
            n_par += int(v.numel())
    out["env"].update({"ckpt_pairformer_blocks": nb_ckpt, "blocks": nb, "c_z": c_z,
                       "pairformer_params": n_par,
                       "ckpt_load_s": round(time.perf_counter() - t0, 2)})
    del ck
    dump()

    with during() as clk:
        t0 = time.perf_counter()
        pf = Pairformer(nb, 32, c_z // 32, 384 // 16, 16, True, comb, ckc)
        ttnn.synchronize_device(dev)
        out["weights_upload_s"] = round(time.perf_counter() - t0, 2)
        out["dram_after_weights_b"] = probe.read("dram")
        dump()

        for s in [int(x) for x in args.sizes.split(",") if x.strip()]:
            probe.reset()
            rec = {"tokens": s, "blocks": nb,
                   "dram_before_b": probe.read("dram"),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "loadavg": os.getloadavg()}
            st = zt = so = zo = None
            t = time.perf_counter()
            try:
                st = ttnn.from_torch(torch.randn(1, s, 384) * 0.5, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=dev)
                zt = ttnn.from_torch(torch.randn(1, s, s, c_z) * 0.5, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=dev)
                rec["dram_inputs_b"] = probe.read("dram")
                so, zo = pf(st, zt, None, None, None)
                ttnn.synchronize_device(dev)
                rec["verdict"] = "PASS"
                rec["z_out_shape"] = list(zo.shape)
            except Exception as exc:
                text = f"{exc}\n{traceback.format_exc()}"
                rec["verdict"] = "OOM" if SL.is_alloc_refusal(exc) else "FAIL"
                rec["oom_kind"] = SL.classify_device_oom(text)
                rec["oom_says"] = SL.describe_device_oom(text)
                m = re.search(r"Not enough space to allocate (\d+) B", text)
                rec["refused_b"] = int(m.group(1)) if m else None
                rec["error"] = str(exc)[:600]
            rec["wall_s"] = round(time.perf_counter() - t, 2)
            rec["peak_dram_b"] = probe.peak["dram"]
            rec["peak_dram_tag"] = probe.peak_tag["dram"]
            rec["peak_l1_b"] = probe.peak["l1"]
            rec["peak_l1_tag"] = probe.peak_tag["l1"]
            rec["min_largest_contiguous_free_dram_b"] = probe.minfree["dram"]
            rec["min_largest_contiguous_free_l1_b"] = probe.minfree["l1"]
            rec["census_at_peak"] = probe.census
            rec["probe_samples"] = probe.samples
            rec["probe_s"] = round(probe.probe_s, 2)
            rec["block_trace"] = probe.trace
            for t_ in (st, zt, so, zo):
                try:
                    ttnn.deallocate(t_)
                except Exception:
                    pass
            rec["dram_after_free_b"] = probe.read("dram")
            out["runs"].append(rec)
            dump()
            print("%5d aa  %-5s peak %6.2f GB  %7.2f s  %s" % (
                s, rec["verdict"], rec["peak_dram_b"] / 1e9, rec["wall_s"],
                rec.get("oom_kind") or ""), flush=True)
    out["aiclk"] = clk.summary()
    out["clock_line"] = clk.line(0)
    dump()
    print(json.dumps(out["aiclk"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
