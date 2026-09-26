#!/usr/bin/env python3
"""bcx-bigtarget: the DRAM curve of a BindCraft 2 gradient step against n, and why 384 refuses.

Forked from `perf/bcx_large/ladder.py` (bcx-large) so the step is the same one it measured:
sequence logits -> AF2 embedding (host, torch autograd) -> 4 extra-MSA + 48 Evoformer blocks on
the card, taped and checkpointed per block -> cotangent seeds on both outputs -> backward through
every block -> logit gradient on the host. Every block runs; nothing is shortened to reach a size.

Three things this adds, all of them about the instrument rather than the model:

1. RESIDENT and PEAK are reported as separate lines. The block-boundary samples ladder.py takes
   read the allocator between a block's pieces of work, so they measure the line a step carries
   from block to block. They cannot see a transient that is born and dies inside one block. Every
   number ladder.py calls a "peak" is a resident reading, and comparing one of those against an
   allocator's figure at the moment it refuses compares two different instruments.
2. FRAGMENTATION is read off the allocator's own block table, not inferred from a total. At the
   highest-resident block, and again at a refusal, this records every free block in a bank: how
   many, how big the largest is, and what the total is. "Free total exceeds the request" and
   "largest free block is smaller than the request" are both answerable from that, and they are
   different failures.
3. NODE sampling is WINDOWED to named blocks. bcx-large's `--node-peak` samples every tape node
   in the whole step and produced nothing at n=352 in 17 minutes. `--node-blocks evo1,evo0` turns
   it on only while those blocks are in the backward, which is where the resident line is highest
   and therefore where the in-block transient is most likely to be the thing that refuses.

One size per process, so an allocation refusal at one size cannot colour the next.

  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-bigtarget \
    python3 perf/bcx_bigtarget/curve.py --n 320 --node-blocks evo1,evo0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import re
import resource
import sys
import time
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack import stack as S  # noqa: E402

OUT = ROOT / "perf" / "bcx_bigtarget"
GB = 1e9

#: tt-metal's refusal names the request, the bank and what the bank has. Every one of those is a
#: number this row needs, and reading them out of the string is the only way to get them: the
#: exception carries no structured payload.
OOM_RE = re.compile(
    r"allocate (\d+) B (\w+) buffer across (\d+) banks, where each bank needs to store (\d+) B, "
    r"but bank size is (\d+) B \(allocated: (\d+) B, free: (\d+) B, largest free block: (\d+) B\)")


def rss_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB


def parse_oom(msg):
    m = OOM_RE.search(msg or "")
    if not m:
        return None
    req, kind, banks, per_bank, bank_size, alloc, free, largest = m.groups()
    req, banks, per_bank = int(req), int(banks), int(per_bank)
    alloc, free, largest, bank_size = int(alloc), int(free), int(largest), int(bank_size)
    return {
        "buffer_type": kind, "request_b": req, "banks": banks, "request_per_bank_b": per_bank,
        "bank_size_b": bank_size, "allocated_per_bank_b": alloc, "free_per_bank_b": free,
        "largest_free_block_per_bank_b": largest,
        "free_total_b": free * banks, "allocated_total_b": alloc * banks,
        "occupancy": alloc / bank_size,
        # The two questions the row turns on, answered as booleans rather than left to a reader.
        "free_total_exceeds_request": free * banks > req,
        "largest_free_block_fits_request": largest >= per_bank,
        "free_total_over_request": (free * banks) / req if req else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="stack")
    ap.add_argument("--reps", type=int, default=1, help="rep 0 is cold (JIT compile)")
    ap.add_argument("--node-blocks", default="",
                    help="comma-separated block names (evo1,evo0,extra0). While one of these is "
                         "the current block in the BACKWARD, sample DRAM at every tape node and "
                         "after every node's backward. Each sample drains the pipeline, so a run "
                         "with this on is not a timing")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    n = args.n
    node_blocks = {b for b in args.node_blocks.split(",") if b}
    rec = {"n": n, "k_extra": args.extra, "k_evo": args.evo, "ckpt": True, "arm": args.arm,
           "seed": args.seed, "stamp": A.stamp(os.environ.get("TT_VISIBLE_DEVICES", "?")),
           "pci": S.sysfs_node()[1], "completed": False, "reps": [],
           "node_blocks": sorted(node_blocks)}
    rec["stamp"].pop("subsystem_device", None)     # afgrad reads the naive node, wrong on qb1
    tag = args.tag or ("_nodes" if node_blocks else "")
    out = OUT / f"curve_n{n}{tag}.json"
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        rec["host_rss_peak_gb"] = rss_peak_gb()
        out.write_text(json.dumps(rec, indent=1, default=str))

    t_open = time.time()
    lv = S.Levers()
    dm, ref = A.load_models(args.params)
    dev = A.Dev(dm)
    lv.arm(args.arm)
    ttnn, ag = dev.ttnn, dev.ag
    rec["open_and_load_s"] = time.time() - t_open
    rec["host_rss_after_load_gb"] = rss_peak_gb()

    def mview():
        return ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)

    def view():
        mv = mview()
        nb = int(mv.num_banks)
        return (int(mv.total_bytes_allocated_per_bank) * nb, int(mv.total_bytes_free_per_bank) * nb,
                int(mv.total_bytes_per_bank) * nb, int(mv.largest_contiguous_bytes_free_per_bank))

    def frag():
        """How broken up the free space is, from the allocator's own two figures.

        `MemoryView.block_table()` is the richer instrument and it is unusable in this build:
        the pybind return type `list[unordered_map<string,string>]` has no registered caster, so
        calling it raises TypeError before any data crosses. The two scalars below survive, and
        they answer the question the row asks. A request can be smaller than the free total and
        still larger than the largest contiguous block, and that is a fragmentation refusal, not
        an exhausted card.
        """
        mv = mview()
        nb = int(mv.num_banks)
        free = int(mv.total_bytes_free_per_bank)
        largest = int(mv.largest_contiguous_bytes_free_per_bank)
        return {"banks": nb, "free_per_bank_b": free, "largest_free_per_bank_b": largest,
                "free_total_b": free * nb,
                "largest_over_free": largest / free if free else None}

    base, _, total, _ = view()
    rec["dram_total_gb"], rec["dram_weights_gb"] = total / GB, base / GB
    st = {"phase": "fwd", "resident_peak": 0, "node_peak": 0, "trace": [], "node_samples": 0}

    def sample(where, node=False):
        """A block-boundary read is RESIDENT. A tape-node read is INSTANTANEOUS."""
        used, free, _, lcf = view()
        if node:
            st["node_samples"] += 1
            if used > st["node_peak"]:
                # "a named cause at a line" needs the line. The tape wrapper sees make_fn, not
                # an op name, so the call stack is the only place the op is written down.
                # Captured only when a new high is set, which is a few dozen times per block.
                frames = [f"{f.filename.split(chr(47))[-1]}:{f.lineno} {f.name}"
                          for f in traceback.extract_stack()
                          if "/tt_bio/" in f.filename or "curve.py" in f.filename]
                st.update(node_peak=used, node_peak_at=(st["phase"], st.get("cur")),
                          node_peak_free_gb=free / GB,
                          node_peak_largest_free_per_bank_gb=lcf / GB,
                          node_peak_stack=frames[-14:])
            return
        st["trace"].append((st["phase"], where, round(used / GB, 4), round(free / GB, 4)))
        if used > st["resident_peak"]:
            st.update(resident_peak=used, resident_peak_at=(st["phase"], where, st.get("cur")),
                      resident_peak_free_gb=free / GB,
                      resident_peak_largest_free_per_bank_gb=lcf / GB,
                      resident_peak_frag=frag())

    ex, ev = dev.extra, dev.evo

    def set_cur(name):
        st["cur"] = name
        st["node_on"] = st["phase"] == "bwd" and name in node_blocks

    def extra(i, *a, **k):                         # afgrad owns the signature; pass it through
        set_cur(f"extra{i}")
        r = ex(i, *a, **k)
        sample(f"extra{i}")
        return r

    def evo(i, *a, **k):
        set_cur(f"evo{i}")
        r = ev(i, *a, **k)
        sample(f"evo{i}")
        return r

    dev.extra, dev.evo = extra, evo

    if node_blocks:                                # both bindings: taped_ttnn imports its own
        from tt_bio import taped_ttnn as T
        orig = ag._tape

        def tape(out_value, parents, make_fn, reads=None):
            def make(*a, **k):
                fn = make_fn(*a, **k)

                def bw(g):
                    r = fn(g)
                    if st.get("node_on"):
                        sample("node", node=True)
                    return r
                return bw
            t = orig(out_value, parents, make, reads)
            if st.get("node_on"):
                sample("node", node=True)
            return t
        ag._tape = T._tape = tape

    def peaks():
        d = {"resident_peak_gb": st["resident_peak"] / GB,
             "resident_peak_at": st.get("resident_peak_at"),
             "resident_peak_free_gb": st.get("resident_peak_free_gb"),
             "resident_peak_largest_free_per_bank_gb":
                 st.get("resident_peak_largest_free_per_bank_gb"),
             "resident_peak_frag": st.get("resident_peak_frag")}
        if node_blocks:
            d.update(node_peak_gb=st["node_peak"] / GB, node_peak_at=st.get("node_peak_at"),
                     node_peak_free_gb=st.get("node_peak_free_gb"),
                     node_peak_largest_free_per_bank_gb=st.get("node_peak_largest_free_per_bank_gb"),
                     node_samples=st["node_samples"],
                     node_peak_stack=st.get("node_peak_stack"),
                     in_block_transient_gb=(st["node_peak"] - st["resident_peak"]) / GB)
        return d

    torch.manual_seed(args.seed)                  # `whole`'s draws, in its order
    ridx = torch.arange(n)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5
    clock = S.Clock()
    try:
        for rep in range(args.reps):
            st.update(phase="fwd", resident_peak=0, node_peak=0, trace=[], node_samples=0,
                      node_on=False)
            gc.collect()
            lgt = logits.clone().float().requires_grad_(True)
            m0, z0 = A.embed(ref["bf16"], lgt, ridx)
            ml, zl = dev.leaf(m0), dev.leaf(z0)
            dev.sync()
            t0 = time.time()
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, args.extra, args.evo, ckpt=True)
            dev.sync()
            t1 = time.time()
            sample("end of forward")
            after_fwd = view()[0] / GB
            seeds = [dev.seed(wm, mo), dev.seed(wz, zo)]
            st["phase"] = "bwd"
            t2 = time.time()
            ag.backward([mo, zo], seeds)
            dev.sync()
            t3 = time.time()
            st["node_on"] = False
            sample("end of backward")
            gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
            m_host, z_host = dev.down(mo.value, m0.shape), dev.down(zo.value, z0.shape)
            out_finite = {"m": bool(torch.isfinite(m_host).all()),
                          "z": bool(torch.isfinite(z_host).all())}
            torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
            ag.release_pins()
            g = lgt.grad.double()
            r = {"rep": rep, "fwd_s": t1 - t0, "bwd_s": t3 - t2, "step_s": t3 - t0,
                 "aiclk": clock.window([(t0, t3)]), "load1_end": os.getloadavg()[0],
                 "dram_after_fwd_gb": after_fwd, **peaks(),
                 "grad": {"finite": bool(torch.isfinite(g).all()), "norm": float(g.norm()),
                          "absmax": float(g.abs().max()),
                          "nonzero_frac": float((g != 0).double().mean()),
                          "gz0_norm": float(gz0.double().norm())},
                 "out_finite": out_finite}
            del m_host, z_host
            del mo, zo, ml, zl, seeds, m0, z0, lgt, gm0, gz0
            gc.collect()
            r["dram_after_release_gb"] = view()[0] / GB
            r["trace"] = list(st["trace"])
            rec["reps"].append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "trace"}, default=str),
                  flush=True)
            save()
        rec["completed"] = True
    except Exception as e:                         # an allocation refusal is the answer
        rec["error"] = str(e)[:2000]
        rec["error_type"] = type(e).__name__
        rec["oom"] = parse_oom(str(e))
        rec["frames"] = [f"{f.filename.split(chr(47))[-1]}:{f.lineno} {f.name}"
                         for f in traceback.extract_tb(e.__traceback__)]
        try:                                       # the state the refusal left behind
            rec["frag_at_refusal"] = frag()
            u, f_, _, l = view()
            rec["view_at_refusal"] = {"allocated_total_b": u, "free_total_b": f_,
                                      "largest_free_per_bank_b": l}
        except Exception as e2:
            rec["frag_at_refusal"] = {"error": f"{type(e2).__name__}: {e2}"}
        rec["failed_rep"] = {"rep": len(rec["reps"]), "phase": st["phase"],
                             "block": st.get("cur"), **peaks(),
                             "trace": list(st["trace"]), "load1": os.getloadavg()[0],
                             "aiclk": clock.window([(t_open, time.time())])}
    finally:
        clock.stop()
        save()
    print(json.dumps({k: rec.get(k) for k in ("n", "completed", "error_type", "oom")},
                     default=str)[:1500], flush=True)
    return 0 if rec["completed"] else 3


if __name__ == "__main__":
    sys.exit(main())
