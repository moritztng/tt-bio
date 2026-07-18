"""Pass-3 multi-card fanout parity check (minimal).

Loads the pipeline in-process on card 0, scores 8 CSAR pairs (single-card
reference), then fans the same 8 pairs out across devices [1,2] (which the
parent never opened, so no CHIP_IN_USE self-deadlock) and compares. Each pair
is independent, so sharded pKd must equal single-card pKd.

  TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 textproto> \
    PYTHONPATH=. python -u tests/verify_affinity_p3_multicard.py
"""
import csv, os, sys, time
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
            rows.append((r["seq"], r["smiles_can"]))
    sub = rows[:8]
    print(f"multicard parity check: {len(sub)} pairs", flush=True)

    t0 = time.perf_counter()
    model = aff.Affinity.from_pretrained()
    print(f"load_total {time.perf_counter()-t0:.2f}s", flush=True)

    single = [model.predict(s, m, fast=True)["neg_log10_affinity_M"] for s, m in sub]
    print(f"single-card (device 0, fast): {single}", flush=True)

    devices = [1, 2]
    t0 = time.perf_counter()
    mc = aff.predict(sub, devices=devices, fast=True)
    print(f"multicard on {devices} in {time.perf_counter()-t0:.2f}s", flush=True)
    mc_pkd = [r["neg_log10_affinity_M"] for r in mc]
    print(f"multicard ({devices}, fast): {mc_pkd}", flush=True)

    st = torch.tensor(single)
    mt = torch.tensor(mc_pkd)
    md = (st - mt).abs().max().item()
    print(f"multicard-vs-single max|dPKd| = {md:.6g}  PCC = {pcc(st, mt):.6f}", flush=True)
    assert md == 0.0, f"multicard fanout NOT bit-exact: max|dPKd|={md}"
    print("PASS multicard fanout is bit-exact vs single-card", flush=True)


if __name__ == "__main__":
    main()
