#!/usr/bin/env python3
"""Which stack's `ttnn.softmax` is right? Scored against float64, never against the other stack.

The op trace puts the first disagreement between 0.69.0 and 0.70.1 at the MSA module's
pair-weighted-averaging softmax (`tt_bio/tenstorrent.py:9591`), a [8, 320, 320] bf16 call made with
`dim=-1`, `numeric_stable=True` and the trunk's own compute kernel config: Blackhole/Wormhole
HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`. Everything
downstream of that call diverges, so this is the call to score.

Reference is `torch.softmax` in float64 of the SAME bf16-rounded bytes the device sees, so the
input's own quantisation is not charged to the op and only the op's arithmetic is measured. Row
sums are reported next to it because a softmax's rows must sum to 1 whatever reference you hold:
that reading needs no comparison at all.

`numeric_stable` is probed on and off at three logit scales. The fold's pre-softmax logits reach
absmax ~5600 by the time the trunk is warm (`perf/ttx/results/trace_*.jsonl`, call 809), which is
past bf16's exp range, so an ignored stability flag would be catastrophic there and nearly
invisible at the small scale the first call runs at.

    <venv>/bin/python softmax_probe.py --out probe_<version>.json
    softmax_probe.py --compare a.json b.json
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from pathlib import Path

import numpy as np

SEED = 0
SHAPE = (8, 320, 320)
INPUTS = Path("/home/ttuser/scratch/ttx/softmax_probe_inputs.npz")


def make_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(SHAPE, dtype=np.float32)
    return {
        # the first call's regime: the trace records absmax 0.46875 on its OUTPUT, so the logits
        # going in are order 1
        "small": x,
        # a warm trunk's regime
        "mid": (x * 40.0).astype(np.float32),
        # what the trace actually sees at call 809, absmax ~5600
        "large": (x * 1800.0).astype(np.float32),
        # every row shifted by a large constant. A correct softmax is invariant to this; a softmax
        # that does not subtract the row max is not. This is the numeric_stable contract itself.
        "shifted": (x + 600.0).astype(np.float32),
    }


def run(out: Path) -> int:
    import torch
    import ttnn
    torch.set_grad_enabled(False)
    if not INPUTS.exists():
        INPUTS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(INPUTS, **make_inputs())
    src = dict(np.load(INPUTS))
    dev = ttnn.open_device(device_id=0)
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"ttnn": md.version("ttnn"), "torch": torch.__version__,
           "arch": str(dev.arch()), "shape": list(SHAPE), "cases": {}}
    try:
        for name, arr in sorted(src.items()):
            t = torch.from_numpy(arr)
            # the bytes the device will actually hold, so the reference sees the same input
            tb = t.to(torch.bfloat16)
            ref = torch.softmax(tb.to(torch.float64), dim=-1)
            dev_in = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            for tag, kw in (("ns_true_ckc", dict(numeric_stable=True, compute_kernel_config=ckc)),
                            ("ns_false_ckc", dict(numeric_stable=False, compute_kernel_config=ckc)),
                            ("ns_true_default", dict(numeric_stable=True)),
                            ("default", {})):
                try:
                    o = ttnn.softmax(dev_in, dim=-1, **kw)
                    got = ttnn.to_torch(o).to(torch.float64)
                    ttnn.deallocate(o)
                    err = (got - ref).abs()
                    rs = got.sum(-1)
                    res["cases"].setdefault(name, {})[tag] = {
                        "max_abs_err": float(err.max()),
                        "mean_abs_err": float(err.mean()),
                        "rel_rms": float((err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())),
                        "rowsum_min": float(rs.min()), "rowsum_max": float(rs.max()),
                        "rowsum_mean": float(rs.mean()),
                        "n_nan": int(got.isnan().sum()), "n_inf": int(got.isinf().sum()),
                    }
                except Exception as e:
                    res["cases"].setdefault(name, {})[tag] = {"raised": f"{type(e).__name__}: {e}"}
            ttnn.deallocate(dev_in)
    finally:
        ttnn.close_device(dev)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"ttnn {res['ttnn']} -> {out}")
    for case, tags in sorted(res["cases"].items()):
        for tag, v in tags.items():
            if "raised" in v:
                print(f"  {case:8s} {tag:16s} {v['raised']}")
            else:
                print(f"  {case:8s} {tag:16s} max_err={v['max_abs_err']:.3e} "
                      f"rel_rms={v['rel_rms']:.3e} rowsum={v['rowsum_min']:.5f}..{v['rowsum_max']:.5f} "
                      f"nan={v['n_nan']} inf={v['n_inf']}")
    return 0


def compare(a: Path, b: Path) -> int:
    A, B = json.loads(a.read_text()), json.loads(b.read_text())
    print(f"A ttnn {A['ttnn']}   B ttnn {B['ttnn']}   {A['arch']}  shape {A['shape']}")
    print(f"{'case':9s}{'config':17s}{'A rel_rms':>12s}{'B rel_rms':>12s}{'A rowsum':>11s}"
          f"{'B rowsum':>11s}  who is closer to float64")
    for case in sorted(A["cases"]):
        for tag in A["cases"][case]:
            va, vb = A["cases"][case][tag], B["cases"][case].get(tag, {})
            if "raised" in va or "raised" in vb:
                print(f"{case:9s}{tag:17s} A={va.get('raised','ok')} B={vb.get('raised','ok')}")
                continue
            ra, rb = va["rel_rms"], vb["rel_rms"]
            who = "same" if ra == rb else ("A" if ra < rb else "B")
            print(f"{case:9s}{tag:17s}{ra:12.3e}{rb:12.3e}{va['rowsum_mean']:11.5f}"
                  f"{vb['rowsum_mean']:11.5f}  {who}"
                  + (f"   B/A = {rb / ra:.1f}x" if ra > 0 else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if not a.out:
        ap.error("--out or --compare")
    return run(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
