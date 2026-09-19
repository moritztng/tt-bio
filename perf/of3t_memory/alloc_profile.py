#!/usr/bin/env python3
"""What OF3's training-size forward costs in BYTES and in ALLOCATION COUNT, on the card.

`of3-1024aa-oom-allocation-count-not-size` is this campaign's standing warning: an OOM names
the allocation that failed to place, and that allocation is usually the last straw of several
co-live tensors, not a single tensor that is too big. `ptx-crop` solved Protenix's 384-token
crop as a SIZE problem. Whether OF3's wall is the same problem is a measurement, so this
script measures both numbers at once and reports them side by side.

The instrument is `ttnn._ttnn.reports.get_buffers(device)`, which is the allocator's own live
list. It is read at the byte high-water mark rather than at the end, because the end of a
forward is the one moment the answer is smallest.

How the peak is found. The proxy below rebinds the name `ttnn` inside every tt-bio module,
exactly the mechanism `tt_bio.taped_ttnn` already uses, and reads DRAM and L1 allocated bytes
after every verb the shipped modules call. Bytes are one allocator field and cost nothing;
the full buffer census walks the live list, so it is taken only when bytes set a new high.
The peak census is therefore the census AT the byte peak, not near it.

The watcher composes with the tape: enter `tape()` first and the module's `ttnn` is already
the tape's proxy, so the watcher wraps that and sees the taped verbs. `tt_bio.autograd` is
watched too -- the tape excludes it so its backward closures reach the real verbs, but an
observer has no such reason to look away, and the backward is where a reverse-mode peak lives.

    alloc_profile.py --sizes 256,384 --blocks 4 --arms off,tape --out out/x.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CKPT = Path(os.environ.get("OF3_CKPT", str(Path.home() / ".boltz" / "of3-p2-155k.pt")))


# --- the watcher --------------------------------------------------------------------------

class _Watch:
    """Whatever `ttnn` currently is, with a probe fired after every call.

    Same shape as `taped_ttnn._Ttnn`: nested namespaces wrap recursively, lookups cache into
    the instance dict so a hot call site pays one attribute load, and anything that is not a
    module or a plain callable is handed back untouched.
    """

    def __init__(self, real, probe, prefix: str = ""):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_probe", probe)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        import types
        real = object.__getattribute__(self, "_real")
        probe = object.__getattribute__(self, "_probe")
        attr = getattr(real, name)
        qual = object.__getattribute__(self, "_prefix") + name
        if isinstance(attr, (types.ModuleType, _Watch)) or type(attr).__name__ == "_Ttnn":
            out = _Watch(attr, probe, qual + ".")
        elif callable(attr) and not isinstance(attr, type):
            def call(*a, _f=attr, _q=qual, **k):
                r = _f(*a, **k)
                probe(_q)
                return r
            out = call
        else:
            return attr
        object.__setattr__(self, name, out)
        return out


class Peak:
    """DRAM and L1 high-water, with the live buffer census taken at the DRAM high-water."""

    def __init__(self, dev, ttnn, reports):
        self.dev, self.ttnn, self.reports = dev, ttnn, reports
        mvd = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        mvl = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        self.dram_banks = int(mvd.num_banks)
        self.l1_banks = int(mvl.num_banks)
        self.dram_total = int(mvd.total_bytes_per_bank) * self.dram_banks
        self.l1_total = int(mvl.total_bytes_per_bank) * self.l1_banks
        self.every = 1
        self.reset()

    def reset(self):
        self.dram_hw = 0
        self.l1_hw = 0
        self.census = None
        self.at_verb = None
        self.calls = 0

    def dram_now(self):
        d = self.ttnn.get_memory_view(self.dev, self.ttnn.BufferType.DRAM)
        return int(d.total_bytes_allocated_per_bank) * self.dram_banks

    def bytes_now(self):
        t = self.ttnn
        l = t.get_memory_view(self.dev, t.BufferType.L1)
        return (self.dram_now(), int(l.total_bytes_allocated_per_bank) * self.l1_banks)

    def free_now(self):
        """Total free and largest contiguous free, both across banks, DRAM."""
        t = self.ttnn
        d = t.get_memory_view(self.dev, t.BufferType.DRAM)
        alloc = int(d.total_bytes_allocated_per_bank) * self.dram_banks
        lcf = d.largest_contiguous_bytes_free_per_bank
        lcf = min(lcf) if isinstance(lcf, (list, tuple)) else int(lcf)
        return {"free_b": self.dram_total - alloc,
                "largest_contiguous_free_per_bank_b": lcf,
                "largest_contiguous_free_across_banks_b": lcf * self.dram_banks}

    def take_census(self):
        """The live buffer list, bucketed by buffer type. Sizes are per-bank as the
        allocator reports them, and also across banks, which is what a shape implies."""
        bufs = self.reports.get_buffers(self.dev)
        out = {}
        for bt, banks in (("DRAM", self.dram_banks), ("L1", self.l1_banks)):
            sizes = [int(b.max_size_per_bank) * banks for b in bufs
                     if str(b.buffer_type).upper().endswith(bt)]
            sizes.sort()
            out[bt] = {
                "count": len(sizes),
                "sum_b": sum(sizes),
                "largest_b": sizes[-1] if sizes else 0,
                "median_b": int(statistics.median(sizes)) if sizes else 0,
                "p90_b": sizes[int(0.9 * (len(sizes) - 1))] if sizes else 0,
                "smallest_b": sizes[0] if sizes else 0,
                "top10_b": sizes[-10:][::-1],
            }
        out["total_buffers"] = len(bufs)
        return out

    def probe(self, verb):
        """One allocator read per call, DRAM only.

        `tenstorrent.dram_peak` documents what this costs: `get_memory_view` drains the
        pipeline, and a 117 aa fold went 12.0 s to 44.7 s under dense tags. The BYTES are
        unaffected -- a drain changes when ops finish, not what is resident -- so the census
        is sound and the wall clock from this script is not a perf number. `every` subsamples
        the probe for a long run; the L1 read is folded into the census rather than paid per
        call, so it is sampled at the DRAM peak and nowhere else.
        """
        self.calls += 1
        if self.every > 1 and self.calls % self.every:
            return
        d = self.dram_now()
        if d > self.dram_hw:
            self.dram_hw = d
            _, l = self.bytes_now()
            self.l1_hw = max(self.l1_hw, l)
            self.census = self.take_census()
            self.at_verb = verb


def _swap_watch(peak, install: bool, saved: list):
    """Rebind `ttnn` in every tt-bio module to the watcher, or put it back.

    Every module that holds the name, whatever it currently holds -- real ttnn outside a tape,
    the tape's proxy inside one. `tt_bio.autograd` included: the tape excludes it so its
    closures call real verbs, but watching them is what makes the backward peak visible.

    `tt_bio.taped_ttnn` is the ONE module that must be left alone, and the reason is not
    politeness. `_Ttnn.__getattr__` decides whether an attribute is a nested namespace with
    `isinstance(attr, type(ttnn))`, reading its own module global. Rebind that global to a
    watcher and the test asks whether `ttnn.experimental` is a `_Watch`, which it is not, so
    the namespace falls through to the plain-attribute return and comes back RAW. Every
    `ttnn.experimental.*` verb is then untaped, and the symptom is not an error from the tape
    but a pybind TypeError several frames later: `minimal_matmul` handed an `autograd.Tensor`.
    Two proxies over one module global, and the second one silently disarms the first.
    """
    import types
    if install:
        del saved[:]
        for name, mod in list(sys.modules.items()):
            if not name.startswith("tt_bio") or mod is None or name == "tt_bio.taped_ttnn":
                continue
            cur = getattr(mod, "ttnn", None)
            if cur is None or isinstance(cur, _Watch):
                continue
            if isinstance(cur, types.ModuleType) or type(cur).__name__ == "_Ttnn":
                saved.append((mod, cur))
                mod.ttnn = _Watch(cur, peak.probe)
    else:
        for mod, cur in saved:
            mod.ttnn = cur
        del saved[:]



def _keeper(kept, shipped_dealloc, ttnn):
    """A `ttnn.deallocate` that HOLDS the tensor instead of freeing it, DRAM only.

    `ptx-crop`'s mechanism, reused because it is the only way to get an upper bound that is a
    card answer rather than an arithmetic one: a reverse-mode tape retains a SUBSET of what the
    forward produced, so the forward with every DRAM free suppressed bounds it from above, in
    bytes and in count at once.

    DRAM only on purpose. An L1 intermediate is transient inside one op, and holding it past
    the op clashes with the next program's statically allocated circular buffers, which makes
    the shipped trimul retry at a narrower chunk width. That measures a different kernel, not
    a bigger tape.
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


# --- the run ------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="256,384")
    ap.add_argument("--blocks", type=int, default=4, help="pairformer blocks; 0 = all 48")
    ap.add_argument("--arms", default="off,tape")
    ap.add_argument("--backward", action="store_true", help="also run and watch the backward")
    ap.add_argument("--probe-every", type=int, default=1,
                    help="read the allocator every Nth verb call (1 = every call)")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()

    import torch
    import ttnn
    from ttnn._ttnn import reports
    import tt_bio as _TB
    from tt_bio.tenstorrent import get_device, Pairformer, accurate_softmax_site
    from tt_bio import openfold3_weights as OW
    from tt_bio import size_limits as SL
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    torch.set_grad_enabled(False)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    peak = Peak(dev, ttnn, reports)
    peak.every = max(1, args.probe_every)
    g = dev.compute_with_storage_grid_size()

    out = {
        "doc": __doc__,
        "env": {
            "host": socket.gethostname(), "arch": str(dev.arch()), "grid": [g.x, g.y],
            "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
            "tt_bio_file": _TB.__file__, "ttnn_file": ttnn.__file__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg": os.getloadavg(),
            "dram_total_b": peak.dram_total, "dram_banks": peak.dram_banks,
            "l1_total_b": peak.l1_total, "l1_banks": peak.l1_banks,
            "probe_every": peak.every,
            "ckpt": str(args.ckpt),
        },
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    # --- weights: the OF3 pairformer_stack at the checkpoint's own dims ---------------------
    t0 = time.perf_counter()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    pat = re.compile(r"^pairformer_stack\.blocks\.(\d+)\.")
    nb_ckpt = 1 + max(int(pat.match(k).group(1)) for k in sd if pat.match(k))
    nb = args.blocks or nb_ckpt
    c_z = int(sd["layer_norm_z.weight"].shape[0]) if "layer_norm_z.weight" in sd \
        else int(sd["layernorm_z.weight"].shape[0])
    c_s = int(sd["layer_norm_s.weight"].shape[0]) if "layer_norm_s.weight" in sd \
        else int(sd["layernorm_s.weight"].shape[0])
    comb = {}
    for i in range(nb):
        blk = OW._sub(sd, f"pairformer_stack.blocks.{i}")
        for k, v in OW.remap_pairformer_block(blk).items():
            comb[f"layers.{i}.{k}"] = v
    out["env"].update({"ckpt_pairformer_blocks": nb_ckpt, "blocks_built": nb,
                       "c_z": c_z, "c_s": c_s,
                       "ckpt_load_s": round(time.perf_counter() - t0, 1)})
    del ck, sd
    gc.collect()
    dump()

    base_dram, base_l1 = peak.bytes_now()
    pf = Pairformer(nb, 32, 4, 24, 16, True, comb, ckc,
                    scale_pair_bias=False, fp32_softmax=True, transpose_bias=True,
                    accurate_softmax=accurate_softmax_site("openfold3.trunk"))
    ttnn.synchronize_device(dev)
    w_dram, w_l1 = peak.bytes_now()
    out["env"]["weights_dram_b"] = w_dram - base_dram
    out["env"]["weights_census"] = peak.take_census()
    del comb
    gc.collect()
    dump()
    print("weights %.3f GB, %d live buffers"
          % (out["env"]["weights_dram_b"] / 1e9,
             out["env"]["weights_census"]["total_buffers"]), flush=True)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    saved: list = []

    for n in sizes:
        for arm in arms:
            gc.collect()
            peak.reset()
            rec = {"tokens": n, "arm": arm, "blocks": nb,
                   "resting_dram_b": peak.bytes_now()[0],
                   "z_bytes": n * n * c_z * 2, "s_bytes": n * c_s * 2}
            t0 = time.perf_counter()
            st = zt = so = zo = None
            tape_cm = None
            try:
                st_t = torch.randn(1, n, c_s) * 0.5 if arm != "boundary" else None
                zt_t = torch.randn(1, n, n, c_z) * 0.5 if arm != "boundary" else None
                if arm in ("off", "keep"):
                    kept: list = []
                    shipped_dealloc = ttnn.deallocate
                    if arm == "keep":
                        ttnn.deallocate = _keeper(kept, shipped_dealloc, ttnn)
                    st = ttnn.from_torch(st_t, dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=dev)
                    zt = ttnn.from_torch(zt_t, dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=dev)
                    rec["inputs_dram_b"] = peak.bytes_now()[0]
                    _swap_watch(peak, True, saved)
                    try:
                        so, zo = pf(st, zt, None, None, None)
                        ttnn.synchronize_device(dev)
                    finally:
                        _swap_watch(peak, False, saved)
                        ttnn.deallocate = shipped_dealloc
                    rec["kept_tensors"] = len(kept)
                    rec["end_dram_b"] = peak.dram_now()
                    rec["end_census"] = peak.take_census()
                    for t_ in kept:
                        try:
                            shipped_dealloc(t_)
                        except Exception:
                            pass
                    kept.clear()
                elif arm == "boundary":
                    # The checkpointed trunk tape IS the block-boundary (s, z) pairs and
                    # nothing inside a block. Allocated for real at the full checkpoint depth,
                    # so the figure is the allocator's answer and not an arithmetic one.
                    held = []
                    nbb = nb_ckpt
                    for _ in range(nbb):
                        held.append(ttnn.from_torch(torch.zeros(1, n, n, c_z),
                                                    dtype=ttnn.bfloat16,
                                                    layout=ttnn.TILE_LAYOUT, device=dev))
                        held.append(ttnn.from_torch(torch.zeros(1, n, c_s),
                                                    dtype=ttnn.bfloat16,
                                                    layout=ttnn.TILE_LAYOUT, device=dev))
                        peak.probe("boundary_pair")
                    rec["boundary_pairs"] = nbb
                    rec["boundary_tape_b"] = peak.dram_now() - rec["resting_dram_b"]
                    rec["end_census"] = peak.take_census()
                    for t_ in held:
                        try:
                            ttnn.deallocate(t_)
                        except Exception:
                            pass
                    held.clear()
                else:
                    from tt_bio import autograd as ag
                    from tt_bio import taped_ttnn as TT
                    tape_cm = TT.tape()
                    tape_cm.__enter__()
                    st = ag.Tensor(ttnn.from_torch(st_t, dtype=ttnn.bfloat16,
                                                   layout=ttnn.TILE_LAYOUT, device=dev),
                                   requires_grad=True)
                    zt = ag.Tensor(ttnn.from_torch(zt_t, dtype=ttnn.bfloat16,
                                                   layout=ttnn.TILE_LAYOUT, device=dev),
                                   requires_grad=True)
                    rec["inputs_dram_b"] = peak.bytes_now()[0]
                    _swap_watch(peak, True, saved)
                    so, zo = pf(st, zt, None, None, None)
                    ttnn.synchronize_device(dev)
                    rec["fwd_dram_b"] = peak.bytes_now()[0]
                    rec["fwd_peak_dram_b"] = peak.dram_hw
                    rec["fwd_peak_census"] = peak.census
                    rec["fwd_peak_at_verb"] = peak.at_verb
                    rec["fwd_verb_calls"] = peak.calls
                    _swap_watch(peak, False, saved)
                    tape_cm.__exit__(None, None, None)
                    tape_cm = None
                    if args.backward:
                        _swap_watch(peak, True, saved)
                        loss = zo.sum_all() if hasattr(zo, "sum_all") else None
                        (loss if loss is not None else zo).backward()
                        ttnn.synchronize_device(dev)
                        _swap_watch(peak, False, saved)
                        rec["after_backward_dram_b"] = peak.bytes_now()[0]
                rec["verdict"] = "PASS"
            except Exception as exc:
                text = f"{exc}\n{traceback.format_exc()}"
                m = re.search(r"Not enough space to allocate (\d+) B", text)
                rec["verdict"] = "OOM" if SL.is_alloc_refusal(exc) else "FAIL"
                rec["refused_b"] = int(m.group(1)) if m else None
                rec["oom_kind"] = SL.classify_device_oom(text)
                rec["error"] = str(exc)[:600]
                rec["traceback"] = traceback.format_exc()[-4000:]
                rec["at_failure"] = peak.free_now()
                try:
                    rec["at_failure_census"] = peak.take_census()
                except Exception:
                    pass
            finally:
                _swap_watch(peak, False, saved)
                if tape_cm is not None:
                    try:
                        tape_cm.__exit__(None, None, None)
                    except Exception:
                        pass
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            rec["peak_dram_b"] = peak.dram_hw
            rec["peak_l1_b"] = peak.l1_hw
            rec["peak_census"] = peak.census
            rec["peak_at_verb"] = peak.at_verb
            rec["verb_calls"] = peak.calls
            rec["free_at_peak"] = peak.free_now()
            for t_ in (st, zt, so, zo):
                try:
                    ttnn.deallocate(t_.value if hasattr(t_, "value") else t_)
                except Exception:
                    pass
            del st, zt, so, zo
            gc.collect()
            rec["dram_after_free_b"] = peak.bytes_now()[0]
            out["runs"].append(rec)
            dump()
            c = rec.get("peak_census") or {}
            d = c.get("DRAM", {})
            print("%4d aa %-5s %-5s peak %6.3f GB  %4d DRAM bufs  largest %6.3f GB  "
                  "median %8.3f MB  at %s  %.0fs"
                  % (n, arm, rec["verdict"], rec["peak_dram_b"] / 1e9, d.get("count", 0),
                     d.get("largest_b", 0) / 1e9, d.get("median_b", 0) / 1e6,
                     rec.get("peak_at_verb"), rec["wall_s"]), flush=True)

    dump()
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
