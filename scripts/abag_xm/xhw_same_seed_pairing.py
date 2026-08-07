#!/usr/bin/env python3
"""Cross-hardware same-seed pairing test (WH galaxy vs BH qb1) — the px/esm root-cause tool.

Question: at identical seeds/config/code, are WH-vs-BH fold differences a systematic
hardware bias or chaotic amplification of reduction-order numerics?

Method (runs on qb1; consumes the N=64 pilot arms, same seed block both hardwares):
  1. STREAM IDENTITY: per-index Spearman of per-sample DockQ (labels.json, fixed scorer)
     and ptm (results.json all_runs) between arms. Same seed -> same host-seeded noise
     stream on both arches shows as strong per-index correlation; a device-side RNG
     difference would decorrelate.
  2. SIGNED BIAS: per-target oracle(gx)-oracle(qb) and dqmean deltas with Wilcoxon
     signed-rank. A systematic hardware quality bias shows as a significant signed shift;
     chaotic amplification is zero-mean with fat tails.

Context for the gate amendment (2026-08-04): the pre-registered N=64 gate null (random
partitions of the pooled arms) measures i.i.d. sampling noise, but the hardware split is
aligned with the same-seed pairing: per-index cross-arch perturbations cancel under random
partition and accumulate under the hardware split, so exceed_q95 is inflated even under
zero bias. The bias-focused statistics below are the decision-relevant ones. Measured
2026-08-04 (post-scorer-fix labels): all four models stream-identical (DockQ spr
0.78-0.92, ptm spr 0.84-0.96), zero significant signed oracle bias (wilcoxon p >= 0.19),
and the on-galaxy mps 1->5 control shifted oracle stats by ~0.014 — the same magnitude as
the WH-vs-BH offset, i.e. the perturbation class reproduces on a single arch.
"""
import json
import glob
import os
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr, wilcoxon

BASE = Path.home() / "abag_xm" / "deepn"
MODELS = ["boltz2", "opendde", "protenix", "esmfold2"]


def getdq(s):
    d = s.get("dockq")
    return d.get("dockq") if isinstance(d, dict) else d


def main():
    for m in MODELS:
        bq, bg = BASE / m, BASE / "galaxy" / m
        if not (bq.is_dir() and bg.is_dir()):
            print(f"{m}: missing arm dir")
            continue
        tq = {os.path.basename(d)[:-4] for d in glob.glob(str(bq / "*_n64"))}
        tg = {os.path.basename(d)[:-4] for d in glob.glob(str(bg / "*_n64"))}
        aq_dq, ag_dq, aq_ptm, ag_ptm = [], [], [], []
        orac_d, dqmean_d = [], []
        for t in sorted(tq & tg):
            fq, fg = bq / f"{t}_n64" / "labels.json", bg / f"{t}_n64" / "labels.json"
            if not (fq.exists() and fg.exists()):
                continue
            lq, lg = json.load(open(fq)), json.load(open(fg))
            sq, sg = lq.get("samples", lq), lg.get("samples", lg)
            dq, dg = [getdq(s) for s in sq], [getdq(s) for s in sg]
            n = min(len(dq), len(dg))
            if n < 16:
                continue
            valid = [(a, b) for a, b in zip(dq[:n], dg[:n])
                     if a is not None and b is not None]
            if len(valid) < 16:
                continue
            va, vb = zip(*valid)
            orac_d.append(max(vb) - max(va))
            dqmean_d.append(st.mean(vb) - st.mean(va))
            aq_dq += list(va)
            ag_dq += list(vb)
            try:
                rq = json.load(open(glob.glob(str(bq / f"{t}_n64" / "*results*" / "results.json"))[0]))
                rg = json.load(open(glob.glob(str(bg / f"{t}_n64" / "*results*" / "results.json"))[0]))
                ar_q = (rq[0] if isinstance(rq, list) else rq).get("all_runs", [])
                ar_g = (rg[0] if isinstance(rg, list) else rg).get("all_runs", [])
                pq = [s.get("ptm") for s in ar_q][:n]
                pg = [s.get("ptm") for s in ar_g][:n]
                if len(pq) == len(pg) == n and all(x is not None for x in pq + pg):
                    aq_ptm += list(pq)
                    ag_ptm += list(pg)
            except Exception:
                pass
        if not orac_d:
            print(f"{m}: no paired targets (qb1-arm labels may still be draining)")
            continue
        out = (f"{m}: paired={len(orac_d)}  per-index DockQ "
               f"spr={spearmanr(aq_dq, ag_dq).statistic:.3f} (n={len(aq_dq)})")
        if aq_ptm:
            out += f"  ptm spr={spearmanr(aq_ptm, ag_ptm).statistic:.3f}"
        print(out)
        neg = sum(1 for d in orac_d if d < -0.02)
        pos = sum(1 for d in orac_d if d > 0.02)
        try:
            w = wilcoxon(orac_d).pvalue
        except Exception:
            w = float("nan")
        print(f"   oracle(gx-qb): med={st.median(orac_d):+.4f} mean={st.mean(orac_d):+.4f} "
              f"wilcoxon_p={w:.3f} sign(+/-)={pos}/{neg}")
        print(f"   dqmean(gx-qb): med={st.median(dqmean_d):+.4f} mean={st.mean(dqmean_d):+.4f}")


if __name__ == "__main__":
    main()
