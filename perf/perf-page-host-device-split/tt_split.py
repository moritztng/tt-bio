#!/usr/bin/env python3
"""Host/device split of one published 512 aa perf-page cell, on Tenstorrent.

Measurement only. Nothing here touches ``site/data/perf-512aa.json`` or what the page
publishes; it produces the other reading of the same fold so we know what it is.

Two arms alternate inside ONE process on the same loaded model and the same fixture:

  ``plain``  -- the published configuration, no brackets installed
  ``instr``  -- ``tt_baseline.Instrument`` installed around the named phases

Alternating, not blocked: a block of one arm followed by a block of the other cannot
separate the instrument from drift, which is how two earlier NO-GOs turned out to be
compile-cost artifacts. The cold fold is discarded before either arm starts.

The instrument is only usable if it does not move the wall it measures, so the plain
arm is not a control that gets reported once -- it is half the measurement. The A/A
spread across the plain folds is the floor everything else is read against.

    benchlock.sh perf-page-host-device-split -- \\
      env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \\
          TT_BIO_LEASE_HOLDER=worker:perf-page-host-device-split \\
      python3 -u perf/perf-page-host-device-split/tt_split.py \\
          --model protenix-v2 --rounds 3 --out .../tt_protenix-v2_qb2c0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def _host_cpu() -> dict:
    """The host CPU is part of the number: featurization is largely single-threaded and
    tracks clock, not core count, so an absolute host share without its CPU is
    meaningless (a 9700X read 8.33 s where a 26.88-vCPU Xeon read 12.734 s)."""
    info = {"host": os.uname().nodename, "nproc": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        info["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    return info


def _cifs(struct_dir: Path) -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in sorted(struct_dir.glob("*.cif"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3, help="folds per arm")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch  # noqa: F401
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    # The published cells ran the shipped tree, so assert this process is folding the
    # checkout it was launched from and not an installed package of the same name.
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    # Each model at its own settings, resolved the same way the three harnesses behind
    # the published cells resolve them (fold_ab512.py:138, fold_ab.py:96, xmodel_ab.py:73).
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    # build_fold's cfg carries no Boltz-2 hyperparameters, so load_model raises
    # KeyError('conf_kwargs'). perf/other512/fold_ab_multi.py already holds the injector
    # that the published cell's harness used (of3_4xpd/xmodel_ab.py:86); import it rather
    # than restate the hyperparameters, and only after the two step counts above, which
    # it reads. Its module body puts its own root at sys.path[0], so snapshot the path.
    if a.model == "boltz2":
        snap = list(sys.path)
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        from fold_ab_multi import patch_boltz2_cfg
        sys.path[:] = snap
        patch_boltz2_cfg()

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    msa_dir = ROOT / f".msa_split_{a.model}_{a.size}"

    import importlib.metadata as im
    res = {
        "model": a.model, "size": a.size, "rounds": a.rounds,
        "fixture": str(tgt.relative_to(ROOT)), "msa_file": str(a3m.relative_to(ROOT)),
        "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
        "ttnn": im.version("ttnn"), "torch": torch.__version__,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "phases": {k: [list(r) for r in v] for k, v in B.PHASES.items() if k == a.model},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_host_cpu(), "folds": [],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.out.write_text(json.dumps(res, indent=1))

    one_fold, meta, state = B.build_fold(a.model, msa_dir, tgt, a3m)
    res["meta"] = {k: meta[k] for k in ("hardware", "load_s", "n_msa", "card_type",
                                        "aiclk_mhz") if k in meta}
    g = T.get_device().compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])
    inst = B.Instrument(a.model, state)
    save()

    def fold(arm: str) -> dict:
        if arm == "instr":
            inst.on()
        try:
            fold_s, m = one_fold()
        finally:
            inst.off()
        rec = {"arm": arm, "fold_s": round(fold_s, 3),
               "plddt": m.get("plddt", m.get("complex_plddt")),
               "n_tokens": m.get("n_tokens"), "cifs": _cifs(struct_dir),
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        if arm == "instr":
            rec["phase"] = inst.row(fold_s)
        res["folds"].append(rec)
        save()
        print(f"  {arm:5s} {fold_s:8.3f}s plddt={rec['plddt']} "
              f"{rec.get('phase', {})}", flush=True)
        return rec

    print(f"=== {a.model} {a.size} aa: cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    assert (a.model.startswith("esmfold2") or cold_m.get("msa") or meta.get("n_msa")), \
        "fold ran without an MSA -- cache seeding failed"
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.2f}s", flush=True)
    save()

    for _ in range(a.rounds):
        for arm in ("plain", "instr"):
            fold(arm)

    plain = [f["fold_s"] for f in res["folds"] if f["arm"] == "plain"]
    instr = [f["fold_s"] for f in res["folds"] if f["arm"] == "instr"]
    rows = [f["phase"] for f in res["folds"] if f["arm"] == "instr"]
    digests = {json.dumps(f["cifs"], sort_keys=True) for f in res["folds"]}

    def med(xs):
        return round(statistics.median(xs), 3)

    res["summary"] = {
        # A/A floor first: the spread of the published arm against itself is what any
        # instrument-vs-plain delta has to be read against.
        "aa_floor_s": round(max(plain) - min(plain), 3),
        "aa_floor_pct": round(100 * (max(plain) - min(plain)) / statistics.median(plain), 3),
        "plain_median_s": med(plain), "instr_median_s": med(instr),
        "plain_s": plain, "instr_s": instr,
        "instr_delta_s": round(statistics.median(instr) - statistics.median(plain), 3),
        "instr_delta_pct": round(100 * (statistics.median(instr) - statistics.median(plain))
                                 / statistics.median(plain), 3),
        "host_s": med([r["host"] for r in rows]),
        "device_s": med([r["device"] for r in rows]),
        "transfer_s": med([r["transfer"] for r in rows]),
        "residual_s": med([r["residual"] for r in rows]),
        "residual_pct": round(100 * statistics.median([r["residual"] for r in rows])
                              / statistics.median(instr), 3),
        # One digest across both arms is the proof the split did not change the output.
        "same_structure_both_arms": len(digests) == 1,
        "cif_digests": sorted(digests),
        # A patch that never fired leaves no per_fn row at all, so the check is
        # against the table, not against n == 0. Any name here means the bracket
        # measured nothing and the split under-counts that bucket.
        "never_fired": sorted({t for _k, t, _b in B.PHASES[a.model]}
                              - {n for r in rows for n in r["per_fn"]}),
    }
    save()
    s = res["summary"]
    print(f"\n=== {a.model}: A/A floor {s['aa_floor_s']} s ({s['aa_floor_pct']} %) ===")
    print(f"  plain {s['plain_median_s']} s   instr {s['instr_median_s']} s   "
          f"delta {s['instr_delta_s']} s ({s['instr_delta_pct']} %)")
    print(f"  host {s['host_s']} s  device {s['device_s']} s  "
          f"transfer {s['transfer_s']} s  residual {s['residual_s']} s "
          f"({s['residual_pct']} %)")
    print(f"  same structure both arms: {s['same_structure_both_arms']}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
