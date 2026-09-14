#!/usr/bin/env python3
"""Peak memory and gate engagement for the three trunk byte levers, 512 -> 1536 aa.

The levers are bit-exact on both architectures across 87 folds, so this row is not a parity row
and not a perf row. It is the one question the two rows that measured them both left open: does a
lever that allocates a concatenated qkv+gate+bias weight per triangle attention, and holds the
trimul gate live across the channel loop, still fit at a large target?

Pass/fail, so contention cannot corrupt it. No ratio is computed here and none may be quoted from
it -- the probes below are cheap but they are not free, and the arms do not share a wall clock.

Per (size, arm):

  verdict        PASS (a CIF with the right residue count) or OOM (an allocator refusal, classified
                 by the ENGINE's own pattern) or FAIL.
  peak           device DRAM and L1 high-water, sampled at the `dram_peak` tag sites the model
                 already carries -- which sit inside trimul, tri_att, the transition and every
                 pairformer block, i.e. exactly where these levers act. `ttnn.get_memory_view`
                 measured at <0.05 ms on this wheel, so every tag is sampled and nothing is
                 decimated.
  fragmentation  the smallest `largest_contiguous_bytes_free_per_bank` seen, and the allocator's
                 own free-block count at the DRAM high-water. A literal live-allocation count is
                 NOT available: `MemoryView.block_table` does not convert through this wheel's
                 pybind layer (it returns list[unordered_map<string,string>] and raises). The
                 free-block census is the observable that `of3-1024aa-oom-allocation-count-not-size`
                 is actually about -- many co-live allocations show up as a collapsing largest free
                 block against a still-large total free.
  engagement     `QKVG_STATS`, `QKVGB_STATS`, `TRIMUL_GOUT_STATS` and every named reject reason,
                 reset per fold. A lever that silently falls back reads as "it worked"; this row
                 asserts on the counter, per size, in both arms.
  digest         sha256[:16] of the CIF. The two arms at one size must be byte-identical, and at
                 512 aa both must be `a91aa44441f0d9c5`.

Arms are applied in-process. That is legal here for a reason the flag-lever harnesses had to
argue and this one does not: the fused path is a DIFFERENT op on a DIFFERENT weight, so it cannot
be handed the other arm's compiled program out of the cache.

    ladder.py --out <json> --cifdir <dir> --sizes 512,640,1024,1536
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

EXPECTED_512_DIGEST = "2bc758a1fb24ef30"   # moved off a91aa44441f0d9c5 when
# TT_BIO_DEVICE_CONDITIONING shipped on by default (perf/b2z2_cond/out/acc_qb2c1.json).


def _free_block_census(path: Path) -> dict:
    """Free-block count and size classes from the allocator's own detailed report."""
    try:
        txt = path.read_text()
    except OSError:
        return {}
    dram = txt.split(",L1", 1)[0]
    blocks = re.findall(r"Size class \d+: \(\d+ - \d+\) blocks: (.*)", dram)
    n = sum(len([b for b in line.split() if b.strip()]) for line in blocks)
    return {"dram_free_blocks": n}


class Probe:
    """Replaces `tenstorrent.dram_peak`, keeping its contract and adding L1 and fragmentation."""

    def __init__(self, ttnn, dev, reports: Path):
        self.ttnn, self.dev, self.reports = ttnn, dev, reports
        self.reset()

    def reset(self):
        self.samples = 0
        self.peak = {"dram": 0, "l1": 0}
        self.peak_tag = {"dram": None, "l1": None}
        self.minfree = {"dram": None, "l1": None}
        self.census = {}
        self.probe_s = 0.0

    def __call__(self, tag=None):
        if tag is None:
            return self.peak["dram"]
        t0 = time.perf_counter()
        try:
            for key, bt in (("dram", self.ttnn.BufferType.DRAM), ("l1", self.ttnn.BufferType.L1)):
                mv = self.ttnn.get_memory_view(self.dev, bt)
                used = (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks
                lcf = mv.largest_contiguous_bytes_free_per_bank
                lcf = min(lcf) if isinstance(lcf, (list, tuple)) else int(lcf)
                if self.minfree[key] is None or lcf < self.minfree[key]:
                    self.minfree[key] = lcf
                if used > self.peak[key]:
                    self.peak[key], self.peak_tag[key] = used, tag
                    if key == "dram":
                        # 0.1 ms, and only on a rise, so it fires a few dozen times per fold.
                        self.ttnn.dump_device_memory_state(self.dev, "ladder_")
                        self.census = _free_block_census(
                            self.reports / "ladder_detailed_memory_usage.csv")
            self.samples += 1
        except Exception:
            pass                                   # a diagnostic must never break a fold
        self.probe_s += time.perf_counter() - t0
        return self.peak["dram"]


# A lever set is the only lever-specific thing in this rig: which flags the arms flip, which
# counters a fold resets, and what the engagement census reads. Everything below it -- the memory
# probe, the OOM classification, the digest comparison -- is size machinery that does not care
# which flag moved, so a second campaign picks a set with `--levers` instead of forking the file.
def _b2z2_apply(arm, TQ, TT):
    on = arm == "on"
    TQ._QKVG_ENABLED = on
    TQ._QKVGB_ENABLED = on
    TT.set_trimul_fused_gout(on)


def _b2z2_reset(TQ, TT):
    TQ.QKVG_STATS[:] = [0, 0]
    TQ.QKVGB_STATS[:] = [0, 0]
    TQ.QKVG_REJECTS.clear()
    TQ.QKVGB_REJECTS.clear()
    TT.TRIMUL_GOUT_STATS[:] = [0, 0]
    TT.TRIMUL_GOUT_REJECTS.clear()


def _b2z2_census(TQ, TT):
    return {
        "qkvg": {"served": TQ.QKVG_STATS[0], "declined": TQ.QKVG_STATS[1],
                 "rejects": {f"{r}@{'x'.join(map(str, s))}": n
                             for (r, s), n in TQ.QKVG_REJECTS.items()}},
        "qkvgb": {"served": TQ.QKVGB_STATS[0], "declined": TQ.QKVGB_STATS[1],
                  "rejects": {f"{r}@{'x'.join(map(str, s))}": n
                              for (r, s), n in TQ.QKVGB_REJECTS.items()}},
        "trimul_gout": {"served": TT.TRIMUL_GOUT_STATS[0], "declined": TT.TRIMUL_GOUT_STATS[1],
                        "rejects": dict(TT.TRIMUL_GOUT_REJECTS)},
    }


def _binaryng_l1_apply(arm, TQ, TT):
    on = arm == "on"
    TT._TRIMUL_MASK_L1 = on
    TT._RESIDUAL_L1 = on


def _binaryng_l1_reset(TQ, TT):
    TT.TRIMUL_MASK_L1_STATS[:] = [0, 0]
    TT.RESIDUAL_L1_STATS[:] = [0, 0]


def _binaryng_l1_census(TQ, TT):
    # Both counters are [taken L1, left in DRAM]. There is no reject dictionary: the only reason
    # either site declines is that the tensor does not fit, and the size that produced the
    # decline is the row's own `size`.
    return {
        "trimul_mask_l1": {"served": TT.TRIMUL_MASK_L1_STATS[0],
                           "declined": TT.TRIMUL_MASK_L1_STATS[1], "rejects": {}},
        "residual_l1": {"served": TT.RESIDUAL_L1_STATS[0],
                        "declined": TT.RESIDUAL_L1_STATS[1], "rejects": {}},
    }


def _transition_h_apply(arm, TQ, TT):
    """`ship` leaves the derivation alone; `hNN` forces every 4-D Transition to NN rows.

    The screen hook is read from the environment on every call by design (it has to reach the
    non-monotonic heights the derivation will not produce), so the arm is set there and not on a
    module global. Popped rather than set to 0 on the reference arm: `if _h` would take 0 as
    unset anyway, and leaving a stale value visible is how a ladder measures one arm twice.
    """
    if arm == "ship":
        os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
    else:
        os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(int(arm.lstrip("h")))


def _transition_h_reset(TQ, TT):
    TT.TRANSITION_H_CHUNK_STATS[:] = [0, 0]
    TT.TRANSITION_H_CHUNK_REJECTS.clear()
    TT.TRANSITION_H_CHUNK_SHAPES.clear()


def _transition_h_census(TQ, TT):
    return {
        "transition_h_chunk": {
            "served": TT.TRANSITION_H_CHUNK_STATS[0],
            "declined": TT.TRANSITION_H_CHUNK_STATS[1],
            "rejects": {f"{r}@{shape}": n
                        for (r, shape), n in TT.TRANSITION_H_CHUNK_REJECTS.items()},
            "heights": {f"{shape}x{hid}@h{h}": n
                        for (shape, hid, h), n in TT.TRANSITION_H_CHUNK_SHAPES.items()},
        },
    }


LEVER_SETS = {
    "transition_h": {"pins": ("TT_BIO_TRANSITION_H_CHUNK",),
                     "apply": _transition_h_apply, "reset": _transition_h_reset,
                     "census": _transition_h_census},
    "b2z2": {"pins": ("TT_BIO_TRIATT_FUSED_QKVG", "TT_BIO_TRIATT_FUSED_QKVGB",
                      "TT_BIO_TRIMUL_FUSED_GOUT"),
             "apply": _b2z2_apply, "reset": _b2z2_reset, "census": _b2z2_census},
    "binaryng_l1": {"pins": ("TT_BIO_TRIMUL_MASK_L1", "TT_BIO_RESIDUAL_L1"),
                    "apply": _binaryng_l1_apply, "reset": _binaryng_l1_reset,
                    "census": _binaryng_l1_census},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="512,640,1024,1536")
    ap.add_argument("--arms", default="off,on")
    ap.add_argument("--levers", default="b2z2", choices=sorted(LEVER_SETS))
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()

    levers = LEVER_SETS[args.levers]
    assert not (set(os.environ) & set(levers["pins"])), \
        "no lever may be pinned in the environment; the arms are set in-process"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_qkv as TQ
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio.size_limits import ALLOC_REFUSAL
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    ttnn.device.EnableMemoryReports()
    reports = REPO / "generated" / "reports"
    probe = Probe(ttnn, dev, reports)
    # Every `dram_peak` call site inside the engine resolves the name as a module global at call
    # time, so rebinding it here reaches the trimul, tri_att, transition and pairformer tags
    # without a line of engine code changing.
    TT.dram_peak = probe
    for name, mod in list(sys.modules.items()):
        if name.startswith("tt_bio.") and getattr(mod, "dram_peak", None) is not None:
            mod.dram_peak = probe

    mvd = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    mvl = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    out = {
        "doc": __doc__,
        "env": {
            "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
            "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
            "tt_bio_file": _TB.__file__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loadavg": os.getloadavg(),
            "seq_len_more_chunking": TT.SEQ_LEN_MORE_CHUNKING,
            "fast_mode": TT._FAST_MODE,
            "dram_total_b": int(mvd.total_bytes_per_bank) * int(mvd.num_banks),
            "l1_total_b": int(mvl.total_bytes_per_bank) * int(mvl.num_banks),
            "levers": args.levers,
            "protocol": {"sampling_steps": args.steps, "recycling_steps": args.recycles,
                         "seed": AB.SEED, "diffusion_trace": False},
        },
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-ladder-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for s in sizes:
        AB._seed_msa(AB.FIX / f"cdk2x2_{s}.yaml", (AB.FIX / f"cdk2x2_{s}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-size-ladder", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    out["dram_after_load_b"] = probe("model-loaded")
    dump()

    def fold(size, arm):
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        levers["apply"](arm, TQ, TT)
        levers["reset"](TQ, TT)
        probe.reset()
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        rec = {"size": int(size), "arm": arm,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "loadavg": os.getloadavg()}
        t = time.perf_counter()
        try:
            metrics, _b, _f = state.predict_one(target, cfg)
            rec["wall_s"] = round(time.perf_counter() - t, 2)
            cifs = sorted(struct_dir.glob("*.cif"))
            if not cifs:
                rec["verdict"] = "FAIL"
                rec["why"] = "no CIF written"
            else:
                keep = args.cifdir / f"{size}_{arm}"
                keep.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cifs[0], keep / cifs[0].name)
                body = (keep / cifs[0].name).read_bytes()
                res = len({ln[22:27] for ln in body.decode().splitlines()
                           if ln.startswith("ATOM")}) or None
                rec["verdict"] = "PASS"
                rec["digest"] = hashlib.sha256(body).hexdigest()[:16]
                rec["residues_in_cif"] = res
                rec["plddt"] = round(float(metrics.get(
                    "plddt", metrics.get("confidence_score", 0))), 6)
        except Exception as e:                       # an allocator refusal is a RESULT here
            rec["wall_s"] = round(time.perf_counter() - t, 2)
            msg = str(e)
            m = ALLOC_REFUSAL.search(msg)
            rec["verdict"] = "OOM" if m else "FAIL"
            rec["error"] = msg[:1500]
            rec["traceback"] = traceback.format_exc()[-2000:]
            if m:
                rec["refusal"] = {k: int(v) for k, v in m.groupdict().items() if v is not None}
        rec["peak_dram_b"] = probe.peak["dram"]
        rec["peak_l1_b"] = probe.peak["l1"]
        rec["peak_dram_gib"] = round(probe.peak["dram"] / 2 ** 30, 3)
        rec["peak_l1_mib"] = round(probe.peak["l1"] / 2 ** 20, 3)
        rec["peak_tag"] = probe.peak_tag
        rec["min_largest_free_per_bank_b"] = probe.minfree
        rec["free_block_census_at_dram_peak"] = probe.census
        rec["probe_samples"] = probe.samples
        rec["probe_overhead_s"] = round(probe.probe_s, 2)
        rec["census"] = levers["census"](TQ, TT)
        return rec

    for size in sizes:
        for arm in [a.strip() for a in args.arms.split(",")]:
            r = fold(size, arm)
            out["runs"].append(r)
            c = r["census"]
            gates = " ".join(f"{k}={v['served']}/{v['declined']}" for k, v in c.items())
            print(f"  {size:>5} {arm:3s} {r['verdict']:5s} {r.get('wall_s'):8.2f}s "
                  f"dram={r['peak_dram_gib']:6.3f}GiB l1={r['peak_l1_mib']:8.1f}MiB "
                  f"{gates} sha={r.get('digest')}", flush=True)
            for k, v in c.items():
                if v["rejects"]:
                    print(f"        {k} rejects: {v['rejects']}", flush=True)
            dump()

    # Verdicts, computed here rather than by eye.
    by = {(r["size"], r["arm"]): r for r in out["runs"]}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ref_arm, test_arms = arms[0], arms[1:]
    checks = {}
    for size in sorted({r["size"] for r in out["runs"]}):
        ref = by.get((size, ref_arm))
        e = {"reference_arm": ref_arm}
        for arm in test_arms:
            on = by.get((size, arm))
            if not (ref and on):
                continue
            a = {"both_complete": ref["verdict"] == "PASS" and on["verdict"] == "PASS",
                 "bit_exact": ref.get("digest") is not None
                 and ref.get("digest") == on.get("digest")}
            if ref["peak_dram_b"]:
                a["peak_dram_delta_pct"] = round(
                    100.0 * (on["peak_dram_b"] - ref["peak_dram_b"]) / ref["peak_dram_b"], 4)
            if ref["peak_l1_b"]:
                a["peak_l1_delta_pct"] = round(
                    100.0 * (on["peak_l1_b"] - ref["peak_l1_b"]) / ref["peak_l1_b"], 4)
            c = on["census"]
            a["gates_engaged_on_arm"] = {k: c[k]["served"] > 0 and c[k]["declined"] == 0
                                         for k in c}
            a["verdict"] = on["verdict"]
            e[arm] = a
        if ref:
            e["arm_ref_is_silent"] = all(v["served"] == 0 for v in ref["census"].values())
            if size == 512:
                e["ref_digest_is_published_512"] = ref.get("digest") == EXPECTED_512_DIGEST
        checks[str(size)] = e
    out["checks"] = checks
    dump()
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
