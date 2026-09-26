#!/usr/bin/env python3
"""The host loss set at upstream's 48 diffusion replicates, measured rather than extrapolated.

`of3t-gpugap` projects it at 24.8-62.1 s (`perf/of3t_throughput/throughput.py:193-199`) from
`0.455 * 48 + 3.0` and `1.226 * 48 + 3.2`, on the premise -- written one line above -- that
"host af3_loss is linear in diffusion samples". That premise comes from `fullstep.py`, which
loops the WHOLE seven-term set once per diffusion root. Neither `recipes.py` nor
`openfold3.py` does: the objective is called once per dataset sample, the distogram head
reads the trunk's pair representation, and the confidence heads read a rollout that is
detached and produced once.

What genuinely repeats per replicate is the diffusion-coupled set -- `mse`, `smooth_lddt`,
`bond` -- and its own `pairwise_distance`. This times that shape directly: the once-terms
once, the per-replicate terms S times, at the stage's real weights.

No card. Run it on a quiet host and record MemAvailable: the same arithmetic reads 0.485 s
at 17 GiB and 1.44-1.69 s at 1 GiB (`balloon.py`), so this number is only comparable beside
the memory it was taken at.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf/of3t_p10host"))

from tt_bio.train import losses as L                      # noqa: E402
from tt_bio.train import objectives as O                  # noqa: E402
from tt_bio.train.losses import of3_loss_weights          # noqa: E402
from lossprofile import build                             # noqa: E402

PER_REPLICATE = ("mse", "smooth_lddt", "bond")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--replicates", type=int, default=48)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "perf/of3t_p10host/out/at48.json")
    a = ap.parse_args()

    n, rng = a.tokens, np.random.default_rng(0)
    weights = of3_loss_weights(a.stage)
    labels, outputs, _ = build(n, rng)
    b2, _ = O._with_entity_flags(labels)
    fires = [t for t, w in weights.items()
             if w != 0.0 and all(k in outputs or k in b2 for k in O._NEEDS[t])]
    once = [t for t in fires if t not in PER_REPLICATE]
    per = [t for t in fires if t in PER_REPLICATE]

    rows = []
    for rep in range(a.reps):
        t0 = time.perf_counter()
        for t in once:
            O._TERMS[t](b2, outputs)
        once_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(a.replicates):
            # A replicate is a fresh denoise: new coordinates, and `pred_dist` recomputed
            # from them, which `openfold3.py` does with `ag.pairwise_distance` on the tape.
            pred = outputs["pred_xyz"] + rng.standard_normal(outputs["pred_xyz"].shape) * 0.1
            o = {**outputs, "pred_xyz": pred, "pred_dist": L._pdist(pred)}
            for t in per:
                O._TERMS[t](b2, o)
        per_s = time.perf_counter() - t0
        rows.append({"rep": rep, "once_s": once_s, "replicated_s": per_s,
                     "total_s": once_s + per_s})
        print(f"rep {rep}: once ({','.join(once)}) {once_s:.3f}s + "
              f"{a.replicates} x ({','.join(per) or 'none'}) {per_s:.3f}s "
              f"= {once_s + per_s:.3f}s", flush=True)

    steady = rows[-1]
    mem = {l.split(":")[0]: int(l.split()[1]) / 1048576
           for l in open("/proc/meminfo") if l.startswith("MemAvailable")}
    out = {"tokens": n, "stage": a.stage, "replicates": a.replicates,
           "once_terms": once, "per_replicate_terms": per, "reps": rows, "steady": steady,
           "mem_available_gib": round(mem["MemAvailable"], 2),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "gpugap_projection_s": [24.8, 62.1],
           "gpugap_source": "perf/of3t_throughput/throughput.py:193-199"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nat {a.replicates} replicates: {steady['total_s']:.3f}s measured, against "
          f"24.8-62.1 s projected. MemAvailable {out['mem_available_gib']} GiB, "
          f"loadavg {out['loadavg'][0]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
