"""Partition the RFD3 step's exposed device time by the callsite that drains it.

At the pinned rfd3_R4 fixture the step is 82 % `tt.to_torch` (perf/p42/r4_b2_ledger.json), and
`to_torch` is a blocking read: its wall is the device time of everything enqueued since the previous
drain that had not finished. So timing the 16 drains BY CALLSITE partitions the whole device half
into named regions, exactly, with no added syncs and no profiler.

This is the instrument p29's per-op sync-bracketed profile should have been: that one fences every
op and charges every drain to the host, which is why its device half alone (211.7 ms) exceeded the
real 241 ms step.

Region = the model line that issued the read. The device work attributed to it is everything the
host enqueued after the previous read returned.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

ACC: dict[str, float] = defaultdict(float)
CNT: dict[str, int] = defaultdict(int)
SNAP: list[tuple[float, dict, dict]] = []


def instrument() -> None:
    orig = ttnn.to_torch

    def w(*a, **k):
        fr = traceback.extract_stack(limit=4)
        gp, dc = fr[-3], fr[-2]
        label = f"{Path(gp.filename).name}:{gp.lineno}->{Path(dc.filename).name}:{dc.lineno}"
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            ACC[label] += time.perf_counter() - t0
            CNT[label] += 1

    ttnn.to_torch = w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="perf/dsfix/targets/R4_9q6y_A.pdb")
    ap.add_argument("--contig", default="A1-585,100")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=10)
    ap.add_argument("--designs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    instrument()
    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    dev_ti = build_token_initializer(
        torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    dev_dm = build_diffusion_module(
        torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    ACC.clear()
    CNT.clear()

    cls = type(dev_dm)
    dm_call = cls.__call__

    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            SNAP.append((time.perf_counter() - t0, dict(ACC), dict(CNT)))

    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    with torch.no_grad():
        sampler.sample(dev_dm, a.designs, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=[torch.Generator().manual_seed(a.seed + i)
                                  for i in range(a.designs)])

    keys = set().union(*[s[1].keys() for s in SNAP])
    med_ms = {k: statistics.median([(SNAP[i][1].get(k, 0.0) - SNAP[i - 1][1].get(k, 0.0)) * 1e3
                                    for i in range(2, len(SNAP))]) for k in keys}
    med_cnt = {k: statistics.median([SNAP[i][2].get(k, 0) - SNAP[i - 1][2].get(k, 0)
                                     for i in range(2, len(SNAP))]) for k in keys}
    step = statistics.median([s[0] * 1e3 for s in SNAP[2:]])
    total = sum(med_ms.values())
    print(f"\n[drain] L={L} atoms  I={int(f['atom_to_token_map'].max()) + 1} tokens  "
          f"designs={a.designs}  median warm step={step:.1f} ms  drains={total:.1f} ms "
          f"({100 * total / step:.1f} %)", flush=True)
    print(f"{'drain callsite':28s} {'ms/step':>9s} {'% step':>7s} {'calls':>6s}")
    for k, ms in sorted(med_ms.items(), key=lambda kv: -kv[1]):
        if ms < 0.05:
            continue
        print(f"{k:28s} {ms:9.2f} {100 * ms / step:6.1f}% {med_cnt.get(k, 0):6.1f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"atoms": L, "designs": a.designs,
                                     "median_warm_step_ms": step, "drain_ms_total": total,
                                     "drain_ms_per_step": med_ms, "calls_per_step": med_cnt},
                                    indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
