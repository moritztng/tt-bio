#!/usr/bin/env python3
"""of3t-ieatom: upstream 0.4.3's float64 s_input on a batch, scored against probe_sinput's dump.

    ref_sinput.py --batch B.pt --checkpoint of3-p2-155k.pt --probe SINPUT.pt --out F.json

Upstream's `input_embedder`, built and loaded by of3t-fullstep64's `ref_step.load` in float64,
called as `run_trunk` calls it. rel over real tokens for the device leg (training) and the host
leg (inference), on the whole of s_input and on its atom-encoder part (the first 384 channels).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_fullstep64"))
import ref_step  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("--batch", "--checkpoint", "--probe", "--out"):
        ap.add_argument(x, required=True, type=Path)
    a = ap.parse_args()
    _cfg, model, _drop, ck = ref_step.load(torch.float64, a.checkpoint, 20260919)
    batch = ref_step.bm.move(torch.load(a.batch, weights_only=False), "cpu", torch.float64)
    with torch.no_grad():
        s_input, _s, _z = model.input_embedder(batch=batch, inplace_safe=False,
                                               use_high_precision_attention=True)
    ref = s_input[0]
    p = torch.load(a.probe, weights_only=False)
    real = p["token_mask"] > 0
    rel = lambda x, sl: float((x[real][:, sl].double() - ref[real][:, sl]).norm()  # noqa: E731
                              / ref[real][:, sl].norm())
    rec = {"batch": {"file": str(a.batch),
                     "sha256": hashlib.sha256(a.batch.read_bytes()).hexdigest()},
           "checkpoint": ck, "n_real": int(real.sum())}
    for leg in ("device", "host"):
        rec[leg] = {"s_input_rel": rel(p[leg], slice(None)), "ai_rel": rel(p[leg], slice(0, 384))}
    rec["device_vs_host_ai_rel"] = float((p["device"][real][:, :384] - p["host"][real][:, :384]).norm()
                                         / p["host"][real][:, :384].norm())
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("device", "host", "device_vs_host_ai_rel")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
