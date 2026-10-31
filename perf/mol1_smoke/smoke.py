"""Mol-1 through the Protenix engine: load the checkpoint, fold one protein cold then warm.

Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/smoke.py CKPT OUT_JSON [SEQ]
Single sequence (no MSA), upstream defaults: 20 recycles, 200 steps, 1 sample. The warm
fold's wall time carries the AICLK sampled during it.
"""
import json
import sys
import time

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from perf.clocksample import during  # noqa: E402
from tt_bio.protenix import Protenix  # noqa: E402
from tt_bio.protenix_data import build_protein_features  # noqa: E402

# Trp-cage x5-ish 100 aa ubiquitin-like test chain (ubiquitin 76 aa + linker), no MSA.
SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGSGSGSNLYIQWLKDGGPSSGRPPPS"


def main(ckpt, out, seq=SEQ):
    res = {"ckpt": ckpt, "n_res": len(seq)}
    t = time.time()
    model = Protenix.load_from_checkpoint(ckpt)
    res["load_s"] = round(time.time() - t, 2)
    feats = build_protein_features(seq)
    for tag in ("cold", "warm"):
        with during() as clk, torch.no_grad():
            t = time.time()
            coords, conf = model.fold(feats, n_step=200, n_sample=1, seed=101, n_cycles=20,
                                      return_confidence=True)
            res[f"{tag}_s"] = round(time.time() - t, 2)
        res[f"{tag}_clock"] = clk.summary().get(0)
        print(tag, res[f"{tag}_s"], clk.line(), flush=True)
    c = conf[0] if isinstance(conf, list) else conf
    res["finite"] = bool(torch.isfinite(torch.as_tensor(coords)).all())
    for k in ("plddt", "ptm", "iptm", "ranking_score"):
        v = c.get(k) if isinstance(c, dict) else None
        if v is not None:
            v = torch.as_tensor(v).float()
            res[k] = round(float(v.mean()), 4)
    ca = torch.as_tensor(coords).reshape(-1, 3)
    res["coord_rms_spread_A"] = round(float((ca - ca.mean(0)).norm(dim=-1).mean()), 3)
    torch.save({"coords": torch.as_tensor(coords), "conf": c}, out.replace(".json", ".pt"))
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
