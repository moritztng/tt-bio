"""Mol-1 per-prediction A/B for the trimul levers: one process per arm, a cold then warm folds, AICLK sampled during each.

Usage: TT_VISIBLE_DEVICES=<card> TT_BIO_LEVERS=<set> python perf/mol1_trimul/fold_ab.py CKPT SEQS_JSON SIZE OUT_JSON [N_WARM]
SIZE 256 / 512 / 1024 picks upstream example proteins (protein chains only, single sequence, no MSA):
7wux chain A (233 aa), 8cnh-assembly1 (475 aa), 8h0q-assembly1 chains 1, 3, 5 (384 + 361 + 247 = 992 tokens).
Upstream defaults: 20 recycles, 200 steps, 5 samples, seed 101. Coords are saved so arms compare at the same seed.
The lever set comes from TT_BIO_LEVERS (tenstorrent.model_levers), recorded beside the timings.
"""
import json
import os
import sys
import time

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from perf.clocksample import during  # noqa: E402
from tt_bio.protenix import Protenix  # noqa: E402
from tt_bio.protenix_data import build_complex_features  # noqa: E402
import tt_bio.trimul_tail as TTL  # noqa: E402

PICK = {"256": ("7wux", [0]), "512": ("8cnh-assembly1", [0]), "1024": ("8h0q-assembly1", [0, 2, 4])}


def main(ckpt, seqs, size, out, n_warm=2):
    name, idx = PICK[size]
    chains = [json.load(open(seqs))[name][i][2] for i in idx]
    res = {"ckpt": ckpt, "size": size, "input": name, "chain_lens": [len(c) for c in chains],
           "levers_env": os.environ.get("TT_BIO_LEVERS"), "folds": []}
    model = Protenix.load_from_checkpoint(ckpt)
    res["levers"] = sorted(getattr(model, "_levers", ()) or ())
    feats = build_complex_features([(c, None) for c in chains])
    for i in range(1 + int(n_warm)):
        TTL.T1_STATS[:] = [0, 0]
        TTL.T1_WHY.clear()
        with during() as clk, torch.no_grad():
            t = time.time()
            coords, conf = model.fold(feats, n_step=200, n_sample=5, seed=101, n_cycles=20, return_confidence=True)
            wall = time.time() - t
        f = {"tag": "cold" if i == 0 else "warm", "wall_s": round(wall, 2), "clock": clk.summary().get(0),
             "t1_served_declined": list(TTL.T1_STATS), "t1_why": dict(TTL.T1_WHY),
             "t2": list(getattr(TTL, "T2_STATS", [])), "t2cm": list(getattr(TTL, "T2CM_STATS", []))}
        res["folds"].append(f)
        print(json.dumps(f, default=str), flush=True)
    c = conf[0] if isinstance(conf, list) else conf
    res["finite"] = bool(torch.isfinite(torch.as_tensor(coords)).all())
    for k in ("plddt", "ptm", "iptm", "ranking_score"):
        v = c.get(k) if isinstance(c, dict) else None
        if v is not None:
            res[k] = round(float(torch.as_tensor(v).float().mean()), 4)
    torch.save({"coords": torch.as_tensor(coords)}, out.replace(".json", ".pt"))
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
