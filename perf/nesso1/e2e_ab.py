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
    ap.add_argument("--split", action="store_true",
                    help="also time the forward's parts (trunk passes, ESM fold-in, crop, affinity)")
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
    split: list = []
    if a.split:
        _time_parts(model, split)

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
        if a.split:
            rep["records"][-1]["split"] = _summarise(split)
            split.clear()
        print(f"[e2e] {record.id} featurize {t1 - t0:.3f}s forward {t2 - t1:.3f}s "
              f"affinity {vals.get('affinity_pred_value')}", flush=True)
    rep["total_s"] = round(time.perf_counter() - t_start, 4)
    rep["load_end"] = os.getloadavg()
    rep["scalars_sha"] = hashlib.sha256(json.dumps(
        [{k: v for k, v in r.items() if k in REPORTED_SCALARS} for r in rep["records"]],
        sort_keys=True).encode()).hexdigest()[:16]
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("import_s", "prepare_s", "model_load_s", "total_s", "scalars_sha")}))


def _time_parts(model, log: list) -> None:
    """Wrap the forward's parts on this instance so each call appends (name, N, seconds).

    Device calls return host tensors, so wall time around a call includes its device work."""
    def wrap(owner, attr, name):
        fn = getattr(owner, attr)

        def timed(*args, **kw):
            t = time.perf_counter()
            r = fn(*args, **kw)
            z = next((x for x in args if isinstance(x, torch.Tensor) and x.dim() == 4), None)
            log.append((name, int(z.shape[1]) if z is not None else 0, time.perf_counter() - t))
            return r
        setattr(owner, attr, timed)

    wrap(model, "forward", "forward")
    wrap(model, "_trunk", "trunk")
    wrap(model, "pocket_crop", "crop")
    wrap(model, "_affinity", "affinity")
    wrap(model.esm_module, "forward", "esm_fold")
    for m in (model.affinity_module, model.affinity_module2):
        wrap(m, "forward", "affinity_member")


def _summarise(log: list) -> dict:
    out: dict = {}
    for name, n, s in log:
        key = f"{name}@{n}" if name in ("trunk", "esm_fold") else name
        out[key] = round(out.get(key, 0.0) + s, 4)
    out["trunk_calls"] = [(n, round(s, 4)) for name, n, s in log if name == "trunk"]
    return out


if __name__ == "__main__":
    main()
