#!/usr/bin/env python3
"""Interleaved fold A/B for M18: OpenFold3's triangle attention at HiFi4 / fp32_dest_acc.

`triatt_sdpa_hifi_site` is read at CONSTRUCTION, so an env-var A/B would need one process per arm
and would lose the interleave that keeps compile and warmup bias off a single arm. Instead this
walks the built model for `TriangleAttention` instances, records the attribute PATH of each, and
flips `sdpa_hifi` on a chosen subset per leg -- so arms alternate inside one process on one device
open, and the same run can score all four sites together or one site at a time.

Firing is counted, never read: `TRIATT_FUSED_HIFI_STATS` (served / declined / too_short) is
snapshotted per leg, and `too_short` is the length floor read directly rather than argued about.

  python3 perf/allm_gates/m18_hifi_ab.py --model openfold3 --arms off,on,off,on,off,on \
      --out perf/allm_gates/ab_m18_openfold3_512.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "gpu_vs_tt", ROOT / "perf"):
    sys.path.insert(0, str(p))


def digest(struct_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(struct_dir.glob("**/*")):
        if f.is_file() and f.suffix in (".cif", ".pdb"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def walk_triatt(root, TriangleAttention, max_depth=14):
    """Every TriangleAttention reachable from `root`, with the attribute path that reached it."""
    seen, out = set(), []

    def rec(o, path, d):
        if d > max_depth or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, TriangleAttention):
            out.append((path, o))
            return
        if isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, f"{path}[{i}]", d + 1)
            return
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, f"{path}.{k}", d + 1)
            return
        dd = getattr(o, "__dict__", None)
        if isinstance(dd, dict):
            for k, v in dd.items():
                if not k.startswith("__"):
                    rec(v, f"{path}.{k}", d + 1)

    rec(root, "model", 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--sites", default="", help="substring of the walked path; empty = all sites")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--attr", default="fused_hifi", choices=("fused_hifi", "sdpa_hifi"),
                    help="which TriangleAttention attribute the arm flips. `fused_hifi` is "
                         "the fp32_softmax branch (OpenFold3, M18); `sdpa_hifi` is the "
                         "else-branch precision config (Protenix-v2 / OpenDDE, M26), which "
                         "changes the fidelity of a fused path those models ALREADY take "
                         "rather than switching route")
    ap.add_argument("--seed", type=int, default=None,
                    help="override tt_baseline.SEED; a second seed turns a single "
                         "deterministic effect into a distribution comparable to the floor")
    ap.add_argument("--savecifs", type=Path, default=None,
                    help="write <size>_<arm>_<leg>/ per leg for perf/other512/cif_rmsd.py")
    a = ap.parse_args()

    # Another worker deleted biotite's bundled `components.bcif` out of the SHARED venv at
    # 2026-09-21 01:28Z (the directory mtime says so; every other file in it is dated Sep 1), which
    # makes any OpenFold3 fold die in `build_openfold3_features` with "Internal CCD not found".
    # `set_ccd_path` repairs it for THIS PROCESS ONLY -- the shared venv is not mine to write to,
    # and running `biotite.setup_ccd` there would re-download a different snapshot for everyone.
    # The donor is the stock biotite 1.6.0 file, byte-identical in size across four venvs on this
    # box and the same version the shared venv reports, so this restores what was removed rather
    # than substituting something else. Both arms see the same CCD either way, so the A/B is
    # unaffected by the choice; it is recorded so the provenance is not lost.
    import biotite.structure.info as _bsi
    _ccd_note = None
    _stock = (Path(_bsi.__file__).parent / "components.bcif")
    if not _stock.is_file():
        for _d in ("/home/ttuser/scratch/i14venv", "/home/ttuser/ptxft-venv",
                   "/home/ttuser/scratch/rel090/cleanvenv090"):
            _c = Path(_d) / "lib/python3.12/site-packages/biotite/structure/info/components.bcif"
            if _c.is_file():
                _bsi.set_ccd_path(_c)
                _ccd_note = str(_c)
                print(f"  [ccd] shared venv copy missing; using {_c}", flush=True)
                break
        else:
            raise RuntimeError("biotite CCD missing and no donor found")

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    if a.seed is not None:
        # The effect must be a DISTRIBUTION, not a point, because the floor it is compared
        # against is one: allm-safety's floor is six seed pairs, so a clearance resting on a
        # single pair is not held to the same shape as the failure it overturns.
        B.SEED = a.seed
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arms": a.arms, "sites": a.sites,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "ccd_donor": _ccd_note,
           "seed": a.seed if a.seed is not None else B.SEED, "legs": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_allmgates_{a.model}_{a.size}",
                                         tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    found = walk_triatt(state, T.TriangleAttention)
    if not found:
        found = walk_triatt(meta, T.TriangleAttention)
    targets = [(p, o) for p, o in found if a.sites in p]
    res["triatt_found"] = len(found)
    res["triatt_targeted"] = len(targets)
    res["paths"] = sorted({p for p, _ in targets})[:40]

    # Group by the construction site each instance was reached through, so a per-site A/B names a
    # --sites substring instead of guessing one, and so "four sites" is a count rather than an
    # assumption carried over from the ledger.
    def _grp(path):
        for tok, pat in (("trunk", "trunk.pairformer"), ("msa", "msa_module"),
                         ("template", "template"), ("confidence", "confidence")):
            if pat in path:
                return tok
        return "other"
    import collections
    res["site_groups"] = dict(collections.Counter(_grp(q) for q, _ in found))
    res["site_groups_targeted"] = dict(collections.Counter(_grp(q) for q, _ in targets))
    res["other_paths"] = sorted({q for q, _ in found if _grp(q) == "other"})[:12]
    # Everything downstream is meaningless if the walk found nothing to flip, and a silent zero
    # would read as "the lever is worth nothing" instead of "the harness missed the modules".
    assert targets, (f"walked the model and found no TriangleAttention matching {a.sites!r} "
                     f"({len(found)} total) -- the A/B would be vacuous")
    res["shipped_sdpa_hifi"] = sorted({bool(o.sdpa_hifi) for _, o in found})
    res["shipped_fused_hifi"] = sorted({repr(o.fused_hifi) for _, o in found})
    res["arm_attr"] = a.attr

    # `fused_hifi`, NOT `sdpa_hifi`. `_attend_heads` has two branches and they read DIFFERENT
    # attributes: the `_FP32_SOFTMAX or self.fp32_softmax` branch gates the fused route on
    # `_fused_hifi_on(self.fused_hifi)` (tenstorrent.py:7625), and only the else-branch reads
    # `self.sdpa_hifi` (:7656). OpenFold3 passes fp32_softmax=True at all four sites, so it is
    # always in the first branch and never reads `sdpa_hifi` at all -- flipping it moved the
    # counters not at all, 0 served / 0 declined / 0 too_short on both arms of a six-leg run.
    # `fused_hifi` is `bool | None`: None follows the process-wide TT_BIO_TRIATT_FUSED_HIFI (False
    # by default), a bool pins the instance and ignores it.
    ATTR = a.attr

    # An arm is "off" (nothing on), "on" (every targeted instance on), or a SITE GROUP token, which
    # turns on only the instances reached through that construction site. Per site is the whole
    # point: the sites differ in sequence length and in how many blocks they run, so a lever that
    # pays on the trunk can still be the one moving the structure somewhere small.
    # An arm may carry a "+1k" suffix, which additionally sets `tri_att_one_k_chunk` on the same
    # instances. That is the fidelity-for-speed trade the campaign asked for rather than a second
    # lever: `_tri_att_sdpa_hifi`'s own docstring says one k chunk spans the whole key length so the
    # online softmax makes no running-max rescale and reduces each row IN THE SAME ORDER as the
    # torch reference and `_fp32_softmax_attention`. If the structural divergence is an artefact of
    # the chunked reduction order rather than of the fused kernel itself, this is what collapses it.
    def set_arm(name):
        one_k = name.endswith("+1k")
        base = name[:-3] if one_k else name
        for q, o in targets:
            on = base == "on" or (base != "off" and _grp(q) == base)
            setattr(o, ATTR, on)
            o.tri_att_one_k_chunk = bool(on and one_k)
        T.TRIATT_FUSED_HIFI_STATS.update(served=0, declined=0, too_short=0)
        T.TRIATT_FUSED_HIFI_PICKS.clear()
        T.SDPA_HIFI_CALLS[0] = 0

    set_arm("on")
    print(f"=== {a.model} {a.size}: {len(targets)}/{len(found)} TriangleAttention targeted "
          f"(shipped sdpa_hifi={res['shipped_sdpa_hifi']} fused_hifi={res['shipped_fused_hifi']}) ===", flush=True)
    for p in res["paths"]:
        print(f"    {p}", flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0),
               "digest": digest(struct_dir),
               "hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
               "hifi_calls": T.SDPA_HIFI_CALLS[0],
               "picks": {str(k): v for k, v in list(T.TRIATT_FUSED_HIFI_PICKS.items())[:12]}}
        if a.savecifs:
            import shutil
            d = a.savecifs / f"{a.size}_{arm}_{i}"
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(struct_dir.glob("**/*.cif")):
                shutil.copy2(f, d / f.name)
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  leg {i} {arm:3s}: {fold_s:8.3f}s  hifi={leg['hifi_stats']} "
              f"calls={leg['hifi_calls']} {leg['digest'][:12]}  {leg['clock_line']}", flush=True)

    by = {}
    for leg in res["legs"]:
        by.setdefault(leg["arm"], []).append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    # A single-leg arm has NO floor, and reporting 0.0 is worse than reporting nothing: it reads
    # as "perfectly repeatable" when it means "measured once". I read exactly that off the 256 aa
    # cell and built a shipping requirement on a 0.061 s difference that sits inside the noise of
    # every multi-leg cell this harness has produced.
    res["aa_floor"] = {k: ((max(v) - min(v)) if len(v) > 1 else None) for k, v in by.items()}
    res["legs_per_arm"] = {k: len(v) for k, v in by.items()}
    res["floor_note"] = "null aa_floor means one leg and no floor; that arm's ratio is directional only"
    res["digests"] = {k: sorted({leg["digest"] for leg in res["legs"] if leg["arm"] == k})
                      for k in by}
    if "off" in res["medians"] and "on" in res["medians"]:
        res["ratio"] = round(res["medians"]["off"] / res["medians"]["on"], 5)
        res["delta_s"] = round(res["medians"]["off"] - res["medians"]["on"], 3)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("medians", "aa_floor", "digests", "ratio", "delta_s")
                      if k in res}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
