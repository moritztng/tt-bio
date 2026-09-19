#!/usr/bin/env python3
"""What a TRAINED trunk tape costs at the recipe crop, measured instead of projected.

`state/ptx/LEDGER.md` K12: every memory number this campaign has is a LoRA adapter on a
frozen trunk, and `dryrun.plan(frozen_trunk=False)` returns UNMEASURED at every crop because
its only source is `perf/hall_grad/DECISION.md`, a feasibility memo whose 27.58 GB is a
projection of unbuilt work at 800 aa. This row was assigned that number. Three arms, all on
the SHIPPED `tenstorrent.Pairformer` at the checkpoint own depth and widths:

  retain   `ttnn.deallocate` is replaced by a keeper that holds the tensor instead of
           freeing it, so every intermediate the block produces stays resident and
           referenced. That is a hard UPPER bound on what any reverse-mode tape can retain
           inside one block, because a tape retains a subset of what the forward produces.
           Measured against the same block with the shipped deallocates, so the difference
           is the retained set and nothing else.

  boundary under per-block checkpointing -- proven equivalent to seven digits by
           `hallgrad-build`, so the mechanism is not speculative -- the trunk tape is the
           block-boundary (s, z) pairs and nothing inside a block survives the block. This
           arm allocates all `blocks` boundary pairs at the crop and reads the allocator, so
           the figure is the card answer and not an arithmetic one.

  leak     the self-closure class (`hallgrad-tape-self-closure-leak`): a node closure that
           reads its own output makes the cycle out -> node -> fn -> out, which CPython
           refcounting cannot collect, and the symptom is an OOM with a tape that fits
           several times over. `autograd._tape` documents the fix; this arm checks it on the
           card by building and dropping the same tape repeatedly and asserting DRAM comes
           back. A leak inflates the tape figure, so it is checked BEFORE the tape figure is
           believed.

    tape_profile.py --sizes 256,384 --out out/tape_qb1c1.json
"""
from __future__ import annotations

import argparse
import gc
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


def _keeper(kept, shipped_dealloc, ttnn):
    """A `ttnn.deallocate` that HOLDS the tensor instead of freeing it, DRAM only.

    DRAM only on purpose. An L1-resident intermediate is transient inside one op, and holding
    it past the op clashes with the next program's statically allocated circular buffers --
    which makes the shipped trimul retry at a narrower chunk width. That measures a different
    kernel rather than a bigger tape, so the L1 ones go to the real deallocate.
    """
    def keep(t, *a, **k):
        try:
            if t.memory_config().buffer_type == ttnn.BufferType.DRAM:
                kept.append(t)
                return None
        except Exception:
            pass
        return shipped_dealloc(t, *a, **k)
    return keep


def _bytes(ts) -> int:
    """Device bytes of a list of ttnn tensors, 0 for any this wheel will not measure."""
    n = 0
    for t in ts:
        try:
            n += int(t.volume()) * int(t.element_size())
            continue
        except Exception:
            pass
        try:
            v = 1
            for d in list(t.shape):
                v *= int(d)
            n += v * 2
        except Exception:
            pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="256,384")
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--retain-blocks", type=int, default=1)
    ap.add_argument("--leak-iters", type=int, default=6)
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
    from tt_bio import autograd as ag
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    def dram():
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

    def freeblocks():
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        lcf = mv.largest_contiguous_bytes_free_per_bank
        return min(lcf) if isinstance(lcf, (list, tuple)) else int(lcf)

    mvd = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
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
            "ckpt": str(args.ckpt),
        },
        "retain": [], "boundary": [], "leak": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    nb_ckpt = n_blocks(sd, "pairformer_stack")
    nb = args.blocks or nb_ckpt
    c_z = int(sd["layernorm_z_cycle.weight"].shape[0])
    n_par = sum(int(v.numel()) for k, v in sd.items() if k.startswith("pairformer_stack."))
    n_all = sum(int(v.numel()) for v in sd.values())
    out["env"].update({"ckpt_pairformer_blocks": nb_ckpt, "c_z": c_z,
                       "pairformer_params": n_par, "model_params": n_all})
    rb = args.retain_blocks
    comb = {}
    for i in range(rb):
        blk = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in sd.items()
               if k.startswith(f"pairformer_stack.blocks.{i}.")}
        for k, v in PW.remap_pairformer_block(blk).items():
            comb[f"layers.{i}.{k}"] = v
    del ck, sd
    gc.collect()
    dump()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    with during() as clk:
        pf = Pairformer(rb, 32, c_z // 32, 384 // 16, 16, True, comb, ckc)
        ttnn.synchronize_device(dev)
        out["dram_after_weights_b"] = dram()
        dump()

        # ---- arm 1: retained set of one shipped block, shipped deallocates vs held
        shipped_dealloc = ttnn.deallocate
        for n in sizes:
            rec = {"tokens": n, "blocks": rb}
            for arm in ("shipped", "held"):
                kept: list = []
                if arm == "held":
                    ttnn.deallocate = _keeper(kept, shipped_dealloc, ttnn)
                else:
                    ttnn.deallocate = shipped_dealloc
                gc.collect()
                base = dram()
                st = zt = so = zo = None
                t0 = time.perf_counter()
                try:
                    st = ttnn.from_torch(torch.randn(1, n, 384) * 0.5, dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=dev)
                    zt = ttnn.from_torch(torch.randn(1, n, n, c_z) * 0.5, dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=dev)
                    inputs = dram()
                    so, zo = pf(st, zt, None, None, None)
                    ttnn.synchronize_device(dev)
                    rec[arm] = {"verdict": "PASS", "base_b": base, "inputs_b": inputs,
                                "after_b": dram(), "kept_tensors": len(kept),
                                "kept_b": _bytes(kept),
                                "largest_contiguous_free_b": freeblocks(),
                                "wall_s": round(time.perf_counter() - t0, 2)}
                except Exception as exc:
                    text = f"{exc}\n{traceback.format_exc()}"
                    m = re.search(r"Not enough space to allocate (\d+) B", text)
                    rec[arm] = {"verdict": "OOM" if SL.is_alloc_refusal(exc) else "FAIL",
                                "base_b": base, "after_b": dram(),
                                "kept_tensors": len(kept),
                                "oom_kind": SL.classify_device_oom(text),
                                "refused_b": int(m.group(1)) if m else None,
                                "error": str(exc)[:400]}
                ttnn.deallocate = shipped_dealloc
                for t_ in (st, zt, so, zo):
                    try:
                        shipped_dealloc(t_)
                    except Exception:
                        pass
                kept.clear()
                gc.collect()
                rec[arm]["dram_after_free_b"] = dram()
            s_ok = rec["shipped"].get("after_b")
            h_ok = rec["held"].get("after_b")
            if s_ok and h_ok:
                rec["retained_b"] = h_ok - s_ok
                rec["retained_per_block_b"] = (h_ok - s_ok) // rb
            out["retain"].append(rec)
            dump()
            print("retain %4d aa  shipped %6.3f GB  held %6.3f GB  retained %6.3f GB"
                  % (n, rec["shipped"].get("after_b", 0) / 1e9,
                     rec["held"].get("after_b", 0) / 1e9,
                     rec.get("retained_b", 0) / 1e9), flush=True)
        ttnn.deallocate = shipped_dealloc

        # ---- arm 2: the checkpointed trunk tape is the block boundaries, allocated for real
        for n in sizes:
            gc.collect()
            base = dram()
            held, rec = [], {"tokens": n, "blocks": nb, "base_b": base}
            try:
                for _ in range(nb):
                    held.append(ttnn.from_torch(torch.zeros(1, n, n, c_z), dtype=ttnn.bfloat16,
                                                layout=ttnn.TILE_LAYOUT, device=dev))
                    held.append(ttnn.from_torch(torch.zeros(1, n, 384), dtype=ttnn.bfloat16,
                                                layout=ttnn.TILE_LAYOUT, device=dev))
                rec["verdict"] = "PASS"
            except Exception as exc:
                text = f"{exc}\n{traceback.format_exc()}"
                m = re.search(r"Not enough space to allocate (\d+) B", text)
                rec.update({"verdict": "OOM" if SL.is_alloc_refusal(exc) else "FAIL",
                            "placed_pairs": len(held) // 2,
                            "oom_kind": SL.classify_device_oom(text),
                            "refused_b": int(m.group(1)) if m else None,
                            "error": str(exc)[:400]})
            rec["after_b"] = dram()
            rec["boundary_tape_b"] = rec["after_b"] - base
            rec["largest_contiguous_free_b"] = freeblocks()
            rec["z_bytes_each"] = n * n * c_z * 2
            rec["s_bytes_each"] = n * 384 * 2
            for t_ in held:
                try:
                    ttnn.deallocate(t_)
                except Exception:
                    pass
            held.clear()
            gc.collect()
            rec["dram_after_free_b"] = dram()
            out["boundary"].append(rec)
            dump()
            print("boundary %4d aa  %d pairs  %6.3f GB  %s"
                  % (n, nb, rec["boundary_tape_b"] / 1e9, rec["verdict"]), flush=True)

        # ---- arm 3: does the tape give its graph back
        ag.install()
        try:
            n = sizes[0]
            w = ag.Tensor(ttnn.from_torch(torch.randn(c_z, c_z) / 16.0, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=dev),
                          requires_grad=True)
            for it in range(args.leak_iters):
                gc.collect()
                before = dram()
                x = ag.Tensor(ttnn.from_torch(torch.randn(1, n, n, c_z) * 0.1,
                                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                              device=dev), requires_grad=True)
                h = x
                for _ in range(4):
                    h = ag.linear(h, w)
                    h = ag.layer_norm(h)
                peak = dram()
                h.backward()
                after_bw = dram()
                x.grad = w.grad = None
                del x, h
                gc.collect()
                out["leak"].append({"iter": it, "before_b": before, "fwd_peak_b": peak,
                                    "after_backward_b": after_bw, "after_drop_b": dram()})
                dump()
            try:
                ttnn.deallocate(w.value)
            except Exception:
                pass
        finally:
            ag.uninstall()
        first, last = out["leak"][0], out["leak"][-1]
        out["leak_verdict"] = {
            "growth_b": last["after_drop_b"] - first["after_drop_b"],
            "iters": len(out["leak"]),
            "reads": ("no leak: DRAM returns to the same figure after every graph is dropped"
                      if last["after_drop_b"] <= first["after_drop_b"]
                      else "DRAM grows across iterations -- investigate before trusting any "
                           "tape figure"),
        }
        print(json.dumps(out["leak_verdict"]), flush=True)

    out["aiclk"] = clk.summary()
    out["clock_line"] = clk.line(0)
    dump()
    print(out["clock_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
