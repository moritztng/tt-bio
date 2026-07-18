"""Pass-3 verification: --fast bit-exactness vs default, CSAR-HiQ_36 accuracy,
and multi-card fanout parity. Run on card 0:

  TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 textproto> \
    PYTHONPATH=. python -u tests/verify_affinity_p3.py
"""
import csv
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from tt_bio import affinity as aff

CSAR = Path("/tmp/PLAPT/data/CSAR-HiQ_36.csv")


def pcc(a, b):
    a = torch.as_tensor(a).flatten().float()
    b = torch.as_tensor(b).flatten().float()
    if a.numel() < 2:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    torch.set_grad_enabled(False)
    rows = []
    with open(CSAR, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["seq"], r["smiles_can"], float(r["neg_log10_affinity_M"])))
    print(f"CSAR-HiQ_36: {len(rows)} pairs", flush=True)

    t0 = time.perf_counter()
    model = aff.Affinity.from_pretrained()
    print(f"load_total {time.perf_counter()-t0:.2f}s", flush=True)

    fast_pkd = []
    labels = []
    t0 = time.perf_counter()
    for i, (seq, sm, lbl) in enumerate(rows):
        f = model.predict(seq, sm, fast=True)["neg_log10_affinity_M"]
        fast_pkd.append(f)
        labels.append(lbl)
        if i % 6 == 0:
            print(f"  fast [{i}/{len(rows)}] pKd={f:.3f} t={time.perf_counter()-t0:.1f}s", flush=True)
    dt = time.perf_counter() - t0
    print(f"fast: {len(rows)} pairs in {dt:.2f}s ({dt/len(rows):.2f}s/pair)", flush=True)

    # default path on a 12-pair subset for bit-exactness vs fast.
    sub_idx = list(range(0, len(rows), 3))[:12]
    default_pkd = []
    t0 = time.perf_counter()
    for i in sub_idx:
        seq, sm, _ = rows[i]
        default_pkd.append(model.predict(seq, sm, fast=False)["neg_log10_affinity_M"])
    print(f"default: {len(sub_idx)} subset pairs in {time.perf_counter()-t0:.2f}s", flush=True)

    fast_t = torch.tensor(fast_pkd)
    labels_t = torch.tensor(labels)
    default_t = torch.tensor(default_pkd)
    fast_sub = torch.tensor([fast_pkd[i] for i in sub_idx])
    maxdiff = (default_t - fast_sub).abs().max().item()
    print(f"fast-vs-default (n={len(sub_idx)}) max|dPKd| = {maxdiff:.6g}  PCC = {pcc(default_t, fast_sub):.6f}", flush=True)

    r = pcc(fast_t, labels_t)
    rmse = (torch.sqrt(((fast_t - labels_t) ** 2).mean())).item()
    mae = ((fast_t - labels_t).abs().mean()).item()
    print(f"CSAR-HiQ_36 (fast, n={len(rows)}): Pearson r={r:.4f} RMSE={rmse:.3f} MAE={mae:.3f}", flush=True)

    # Release the parent's device before fanout so a shard on device 0 does not
    # self-deadlock on the CHIP_IN_USE_0 lock the parent holds.
    from tt_bio.tenstorrent import cleanup
    cleanup()
    print("parent device released before fanout", flush=True)

    # multi-card fanout parity: 8 pairs via predict_multicard on [0,1] (fast),
    # compared to the in-process single-card fast results above.
    sub = rows[:8]
    sub_pairs = [(s, m) for s, m, _ in sub]
    devices = [0, 1]
    t0 = time.perf_counter()
    mc = aff.predict(sub_pairs, devices=devices, fast=True)
    print(f"multicard {len(sub_pairs)} pairs on {devices} in {time.perf_counter()-t0:.2f}s", flush=True)
    mc_pkd = [r["neg_log10_affinity_M"] for r in mc]
    single_pkd = fast_pkd[: len(sub_pairs)]
    mc_t = torch.tensor(mc_pkd)
    single_t = torch.tensor(single_pkd)
    md = (mc_t - single_t).abs().max().item()
    print(f"multicard-vs-single (n={len(sub_pairs)}) max|dPKd| = {md:.6g}  PCC = {pcc(mc_t, single_t):.6f}", flush=True)


if __name__ == "__main__":
    main()
