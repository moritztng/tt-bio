"""One Nesso-1 screen, phase-timed: what `tt_bio.nesso1.screen` does, with a clock on each step.

    PYTHONPATH=<tree> python perf/nesso1/e2e_ab.py --yaml <file> --records 4 --out <dir> --report r.json

The YAML is copied `--records` times under distinct ids so record 0 pays the kernel compile
and records 1.. are warm. Per record: featurise (`dataset[idx]`) and forward (`model.predict`)
seconds, and the reported scalars, so two trees can be compared for identical output.
The driver imports only public tt_bio names, so the same file times any tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--records", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()

    t_start = time.perf_counter()
    import tt_bio
    from tt_bio.nesso1 import REPORTED_SCALARS, Nesso1
    from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "in").mkdir(parents=True)
    src = Path(a.yaml).read_text()
    for i in range(a.records):
        (out / "in" / f"rec{i}.yaml").write_text(src)

    rep = {"host": socket.gethostname(), "tree": str(Path(tt_bio.__file__).parents[1]),
           "visible": os.environ.get("TT_VISIBLE_DEVICES"), "load_start": os.getloadavg(),
           "import_s": round(time.perf_counter() - t_start, 4), "records": []}
    t0 = time.perf_counter()
    dataset, manifest, failed = prepare(out / "in", out / "run", num_workers=0)
    rep["prepare_s"] = round(time.perf_counter() - t0, 4)
    assert not failed, failed
    t0 = time.perf_counter()
    # The same settings screen() ships: bf16 trunk, fp32 affinity heads, 5 recycles.
    model = Nesso1.from_pretrained(use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)
    model.predict_args["recycling_steps"] = 5
    rep["model_load_s"] = round(time.perf_counter() - t0, 4)

    for idx, record in enumerate(manifest.records):
        torch.manual_seed(a.seed)
        t0 = time.perf_counter()
        feats = collate(dataset[idx])
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(feats)
        t2 = time.perf_counter()
        vals = {k: float(pred[k].reshape(-1)[0]) for k in REPORTED_SCALARS if k in pred}
        rep["records"].append({"id": record.id, "featurize_s": round(t1 - t0, 4),
                               "forward_s": round(t2 - t1, 4), "start_epoch": time.time() - (t2 - t0),
                               "n_tokens": int(feats["token_pad_mask"].shape[-1]), **vals})
        print(f"[e2e] {record.id} featurize {t1 - t0:.3f}s forward {t2 - t1:.3f}s "
              f"affinity {vals.get('affinity_pred_value')}", flush=True)
    rep["total_s"] = round(time.perf_counter() - t_start, 4)
    rep["load_end"] = os.getloadavg()
    rep["scalars_sha"] = hashlib.sha256(json.dumps(
        [{k: v for k, v in r.items() if k in REPORTED_SCALARS} for r in rep["records"]],
        sort_keys=True).encode()).hexdigest()[:16]
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("import_s", "prepare_s", "model_load_s", "total_s", "scalars_sha")}))


if __name__ == "__main__":
    main()
