#!/usr/bin/env python3
"""AF2-IG with the structure module on host (off), on card (on) and on card from a trace (tr), one
process, same weights. The device opens with the "af2" trace region so the tr arm can capture.

Per input: two warm folds per arm (the trace captures on its second call), then `--reps` timed
folds cycling the arms, and the confidence metrics and atom coordinates of each arm. Writes
ab.json: fold time medians, metric deltas, each arm's coordinate deviation from off (all atoms,
CA, Angstrom; same model frame, so no superposition) and whether tr reproduces on exactly.

    python perf/spd/af2ig_sm_ab.py --params .../params_model_1_ptm.npz --out out/smab \\
        perf/kitcmp/fixtures/af2ig_pdl1_l80_s{0,1,2}.yaml
"""
import argparse, json, statistics as st, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("inputs", nargs="+")
p.add_argument("--params", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--arms", default="off,on,tr")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tt_bio.af2 import load_af2_device_model  # noqa: E402
from tt_bio.af2_weights import load_af2_state_dict  # noqa: E402
from tt_bio.af2ig import fold  # noqa: E402
from tt_bio.af2ig_input import read_af2ig_input  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

get_device(trace="af2")

model = load_af2_device_model(load_af2_state_dict(a.params), template=True).eval()


ARMS = a.arms.split(",")


def run(spec, arm: str):
    model.structure_on_card = arm != "off"
    model.structure_trace = arm == "tr"
    t = time.perf_counter()
    with torch.no_grad():
        pred = fold(model, spec)
    return time.perf_counter() - t, pred


def coords(pred):
    return np.asarray(pred.coords, dtype=np.float64)


rows = []
for path in a.inputs:
    spec = read_af2ig_input(path)
    for _ in range(2):
        for arm in ARMS:
            run(spec, arm)
    times = {arm: [] for arm in ARMS}
    preds = {}
    for _ in range(a.reps):
        for arm in ARMS:
            dt, preds[arm] = run(spec, arm)
            times[arm].append(dt)
    base = preds[ARMS[0]]
    ca = np.asarray([i for i, name in enumerate(base.atom_array.atom_name) if name == "CA"])
    row = {"input": Path(path).name, "tokens": base.tokens, "arms": ARMS}
    for arm in ARMS:
        d = np.linalg.norm(coords(preds[arm]) - coords(base), axis=-1)
        row[arm] = {"s": st.median(times[arm]), "runs": times[arm],
                    "atoms_max_A": float(d.max()), "atoms_rms_A": float(np.sqrt((d ** 2).mean())),
                    "ca_rms_A": float(np.sqrt((d[ca] ** 2).mean())) if len(ca) else None,
                    "finite": bool(np.isfinite(coords(preds[arm])).all()), "metrics": preds[arm].metrics}
    if "on" in preds and "tr" in preds:
        row["tr_equals_on"] = bool(np.array_equal(coords(preds["tr"]), coords(preds["on"])))
    rows.append(row)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "metrics"})
                      for k, v in row.items()}), flush=True)
(a.out / "ab.json").write_text(json.dumps(rows, indent=1))
