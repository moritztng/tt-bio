#!/usr/bin/env python3
"""The VJP question for a zero-fill change, answered by IDENTITY rather than by tolerance.

A46 clause 1 grades a backward on its vector-Jacobian product, per-tensor worst case, located
by parameter path. For this row the strongest available answer is stronger than a tolerance:
the change swaps one all-zeros bf16 tensor for another all-zeros bf16 tensor, so every
parameter gradient over the real 2,482-node tape should be BYTE-IDENTICAL. A digest A/B says
that or refutes it, and it covers every site `grad_zeros` reaches in the model that actually
runs -- the head-split backward, `_pad_slice`, the slice backward and the sum backward -- which
`gradcheck.py`'s synthetic cases do not all touch.

A digest comparison is only worth reading beside its A/A control, so this takes `--tag`: run
the same arm twice, confirm the two agree, and only then read the cross-arm comparison. Without
that control an agreement could be the digest being blind and a disagreement could be ordinary
run-to-run drift.

    graddigest.py --zeros host   --tag a1
    graddigest.py --zeros host   --tag a2      # A/A control
    graddigest.py --zeros device --tag b1
    python3 perf/of3t_zerosfill/compare_grads.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402
from perf.of3t_perf import step as S                                     # noqa: E402


def digest(t):
    """sha256 of the gradient's raw bf16 bytes, plus the numbers a mismatch is read with."""
    import torch
    import ttnn
    x = ttnn.to_torch(t)
    b = x.contiguous().view(torch.uint8).numpy().tobytes()
    f = x.to(torch.float64)
    return {"sha256": hashlib.sha256(b).hexdigest()[:32],
            "shape": list(x.shape), "dtype": str(x.dtype),
            "l2": float(f.norm()), "absmax": float(f.abs().max()),
            "nonfinite": int((~f.isfinite()).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--zeros", required=True, choices=("device", "host"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_path = a.out or (REPO / "perf/of3t_zerosfill/out" / f"grads_{a.tag}.json")

    out = {"doc": __doc__.splitlines()[0], "argv": sys.argv[1:], "tag": a.tag,
           "zeros": a.zeros, "env": {
               "host": socket.gethostname(),
               "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
               "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "loadavg": os.getloadavg()},
           "config": {"crop": a.tokens, "cycles": a.cycles}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: out_path.write_text(json.dumps(out, indent=1, default=str))  # noqa: E731
    dump()

    with during() as clk:
        try:
            import torch
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(trunk, out)

            # exact_training OFF, per the sprint's grading convention for this row: with it ON
            # the verb under test is invisible behind a host float64 softmax that is 95.2 % of
            # the leg. It is a denominator here, never a lever.
            exact_ctx = ag.exact_training(False)
            exact_ctx.__enter__()
            ag.DEVICE_ZEROS = (a.zeros == "device")
            out["env"]["device_zeros"] = bool(ag.DEVICE_ZEROS)
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            gc.collect()

            with ag.tape():
                _s, z = trunk(*args_, **kwargs_)
                ttnn.synchronize_device(dev)
            if not isinstance(z, ag.Tensor):
                raise SystemExit("the trunk output is not taped; nothing to differentiate")

            zr = z.value
            seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
            rs = TT.recompute_scope()
            rs.__enter__()
            try:
                ag.backward([z], [seed])
                ttnn.synchronize_device(dev)
            finally:
                rs.__exit__(None, None, None)

            grads, missing = {}, []
            for name, t in params.items():
                g = getattr(t, "grad", None)
                if g is None:
                    missing.append(name)
                    continue
                grads[name] = digest(g)
            out["grads"] = grads
            out["params_declared"] = len(params)
            out["params_with_grad"] = len(grads)
            out["params_without_grad"] = len(missing)
            # One digest over the whole set, in a fixed order, so a single line decides it.
            h = hashlib.sha256()
            for name in sorted(grads):
                h.update(name.encode())
                h.update(grads[name]["sha256"].encode())
            out["whole_set_sha256"] = h.hexdigest()[:32]
            ag.release_pins()
            exact_ctx.__exit__(None, None, None)
        except Exception:                                                # noqa: BLE001
            out["error"] = traceback.format_exc()[-6000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()
    print(f"\nTAG {a.tag}  zeros={a.zeros}  {out.get('params_with_grad')} gradients  "
          f"whole-set {out.get('whole_set_sha256')}")
    print(out["env"].get("aiclk_line"))
    print(f"-> {out_path}")
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
