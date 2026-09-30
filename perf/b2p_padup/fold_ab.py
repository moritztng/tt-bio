#!/usr/bin/env python3
"""Interleaved fold A/B of the fused-HiFi pad-up on a model that shares `tenstorrent.py` with BC2.

The arm is `tenstorrent._TRIATT_HIFI_PAD_UP_TILES`, which `_tri_att_hifi_pad_up` reads at call time,
so `off` (0) and `on` (the shipped value) alternate in one process on one device open with every
site default left as shipped. Per leg it records, counted rather than inferred:

  * every call into `_tri_att_sdpa_hifi` by query length and outcome (served natively, served on
    the padded axis, declined), and every call into `_fp32_softmax_attention` by length, so a leg
    says which route each triangle attention took;
  * `TRIATT_FUSED_HIFI_STATS` including `padded`;
  * fold time with AICLK sampled DURING the fold (`clocksample.during`), the structure digest,
    pLDDT, and optionally the CIFs for `perf/other512/cif_rmsd.py`.

  fold_ab.py --model openfold3 --size 544 --arms off,on,off,on,off,on --out f.json --savecifs d/
"""
from __future__ import annotations

import argparse, collections, json, os, shutil, socket, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "gpu_vs_tt", ROOT / "perf"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(ROOT / "perf" / "allm_gates"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--yaml", type=Path, default=None, help="target; default the tiled-CDK2 fixture")
    ap.add_argument("--a3m", type=Path, default=None)
    ap.add_argument("--savecifs", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), _TB.__file__
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from m18_hifi_ab import digest
    from tt_bio.main import ensure_p300_mesh_descriptor, _resolve_recycling_steps, _resolve_sampling_steps
    ensure_p300_mesh_descriptor()
    if a.seed is not None:
        B.SEED = a.seed
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    SHIPPED = T._TRIATT_HIFI_PAD_UP_TILES
    assert SHIPPED > 0, "pad-up is off in this process; the A/B would compare off against off"

    calls = collections.Counter()
    hifi, fp32 = T._tri_att_sdpa_hifi, T._fp32_softmax_attention

    def hifi_counted(q, *args, **kw):
        S = int(q.shape[2])
        before = T.TRIATT_FUSED_HIFI_STATS["padded"]
        o = hifi(q, *args, **kw)
        how = ("declined" if o is None else
               "padded" if T.TRIATT_FUSED_HIFI_STATS["padded"] > before else "native")
        calls[f"hifi:{S}:{how}"] += 1
        return o

    def fp32_counted(q, *args, **kw):
        calls[f"composed:{int(q.shape[2])}"] += 1
        return fp32(q, *args, **kw)

    T._tri_att_sdpa_hifi, T._fp32_softmax_attention = hifi_counted, fp32_counted

    tgt = a.yaml or a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.a3m or a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "target": str(tgt), "a3m": str(a3m), "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arms": a.arms, "pad_up_shipped": SHIPPED,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "seed": B.SEED, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg_start": os.getloadavg(), "legs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    one_fold, meta, _state = B.build_fold(a.model, ROOT / f".msa_b2p_padup_{a.model}_{tgt.stem}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])

    def set_arm(arm):
        T._TRIATT_HIFI_PAD_UP_TILES = {"off": 0, "on": SHIPPED}[arm]
        for k in T.TRIATT_FUSED_HIFI_STATS:
            T.TRIATT_FUSED_HIFI_STATS[k] = 0
        calls.clear()

    set_arm("on")
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    res["cold_calls"] = dict(calls)
    print(f"cold {cold_s:.2f}s calls={dict(calls)}", flush=True)
    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        leg = {"i": i, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "aiclk": clk.summary(), "clock_line": clk.line(0), "digest": digest(struct_dir),
               "stats": dict(T.TRIATT_FUSED_HIFI_STATS), "calls": dict(sorted(calls.items())),
               "loadavg": os.getloadavg()}
        if a.savecifs:
            d = a.savecifs / f"{a.size}_{arm}_{i}"
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted(struct_dir.glob("**/*.cif")):
                shutil.copy2(f, d / f.name)
        res["legs"].append(leg)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"leg {i} {arm:3s} {fold_s:8.3f}s plddt={leg['plddt']} {leg['digest'][:12]} "
              f"{leg['clock_line']} stats={leg['stats']} calls={leg['calls']}", flush=True)
    by = collections.defaultdict(list)
    for leg in res["legs"]:
        by[leg["arm"]].append(leg["fold_s"])
    res["medians"] = {k: statistics.median(v) for k, v in by.items()}
    res["spread"] = {k: (max(v) - min(v)) if len(v) > 1 else None for k, v in by.items()}
    res["digests"] = {k: sorted({l["digest"] for l in res["legs"] if l["arm"] == k}) for k in by}
    if {"off", "on"} <= set(by):
        res["off_over_on"] = round(res["medians"]["off"] / res["medians"]["on"], 4)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res.get(k) for k in ("medians", "spread", "digests", "off_over_on")}), flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
