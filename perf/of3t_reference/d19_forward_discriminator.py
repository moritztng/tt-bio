#!/usr/bin/env python3
"""D19: attribute the 6.73e-03 forward gap between the reference and our qb2 float64 replay.

Same step, num_recycles 0, the same batch by hash and the same draws replayed, and the two sides
still disagree: loss 1.6311432393241485 on qb2 against 1.6422035029890711 here. A gradient
comparison cannot be tighter than the forward it is taken at, so that gap is a floor under
instrument A and it has to be attributed, not noted.

The suspect is upstream's own unconditional downcast. `projects/of3_all_atom/model.py`, the end of
`run_trunk`, is `return s_input.float(), s.float(), z.float()`, and their modules wrap work in
`torch.amp.autocast(device_type=..., dtype=torch.float32)`. Both are precision FLOORS in their
shipped bf16/fp32 paths and both become DOWNCASTS inside a float64 model, so whether a float64
replay neutralises them changes which function is being evaluated -- and the two downcasts do not
behave the same way on the two devices:

  autocast        CPU: torch disables the block itself.   CUDA: it downcasts a float64 graph.
  Tensor.float()  CPU: downcasts.                         CUDA: downcasts.

So a CPU replay that patches neither is NOT computing the same function as a CUDA run that
patches both, and `.float()` is the half that bites on both devices.

Four arms, forward only. Each restores the same pinned RNG state and replays the same recorded
draws, so nothing but the patching differs.

  A  autocast off, Tensor.float() identity on float64   the published reference
  B  autocast off, Tensor.float() left alone            what a CPU float64 replay computes
  C  autocast on,  Tensor.float() identity on float64   the other half, alone
  D  neither patched                                    their shipped path, in float64
"""
import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bundle_min", HERE / "bundle_min.py")
bm = importlib.util.module_from_spec(spec)
sys.modules["bundle_min"] = bm
spec.loader.exec_module(bm)


class patched:
    """no_autocast, but each half switchable, so the arms isolate one downcast at a time."""

    def __init__(self, kill_autocast: bool, keep_double: bool):
        self.kill_autocast, self.keep_double = kill_autocast, keep_double

    def __enter__(self):
        self._a, self._f = torch.amp.autocast, torch.Tensor.float
        orig_float = self._f
        if self.kill_autocast:
            torch.amp.autocast = lambda *a, **k: bm._null()
        if self.keep_double:
            torch.Tensor.float = lambda t, *a, **k: (
                t if t.dtype is torch.float64 else orig_float(t, *a, **k))
        return self

    def __exit__(self, *exc):
        torch.amp.autocast, torch.Tensor.float = self._a, self._f
        return False


ARMS = [
    ("A", True, True, "the published reference"),
    ("B", True, False, "what a CPU float64 replay computes"),
    ("C", False, True, "autocast alone"),
    ("D", False, False, "their shipped path, in float64"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--batch-sha256")
    ap.add_argument("--replay-draws", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--reference-loss", type=float, default=1.6422035029890711)
    ap.add_argument("--ours-loss", type=float, default=1.6311432393241485,
                    help="of3t-gradients' qb2 CPU float64 loss on the same boundary")
    args = ap.parse_args()

    bm.pin_deterministic_kernels(True)
    dtype, device = torch.float64, "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    got = bm.sha256_file(args.batch)
    if args.batch_sha256 and got != args.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != expected {args.batch_sha256}")

    raw = torch.load(args.batch, weights_only=False)
    cfg, model, loss_fn, _ = bm.build(dtype, args.seed, device, args.num_recycles)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    batch = bm.move(raw, device, dtype)
    pinned = bm.rng_state(model)
    replay = torch.load(args.replay_draws, map_location="cpu", weights_only=False)

    results = {}
    for name, kill_ac, keep_d, what in ARMS:
        bm.set_rng_state(pinned, model)
        rec = bm.DrawRecorder(copy.deepcopy(replay))
        t0 = time.time()
        try:
            with patched(kill_ac, keep_d):
                private = copy.deepcopy(batch)
                with rec:
                    b, out = model(private)
                    loss, breakdown = loss_fn(b, out, _return_breakdown=True)
            r = {
                "loss": float(loss),
                "terms": {k: float(v) for k, v in breakdown.items()},
                "n_draw_mismatch": len(getattr(rec, "mismatch", []) or []),
                "seconds": round(time.time() - t0, 3),
            }
            r["vs_reference"] = r["loss"] - args.reference_loss
            r["vs_ours"] = r["loss"] - args.ours_loss
        except Exception as exc:  # an arm that cannot run is itself the answer
            r = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 3)}
        r.update(autocast_killed=kill_ac, float_kept_double=keep_d, what=what)
        results[name] = r
        print(f"{name}  {json.dumps(r)}", flush=True)

    out = {
        "question": "D19: which downcast accounts for the forward gap?",
        "reference_loss": args.reference_loss,
        "ours_loss_qb2_cpu_f64": args.ours_loss,
        "gap": args.reference_loss - args.ours_loss,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": "float64",
        "num_recycles": args.num_recycles,
        "batch_sha256": got,
        "draws_sha256": bm.sha256_file(args.replay_draws),
        "arms": results,
    }
    (args.out / "d19_forward_arms.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v.get("loss", v.get("error")) for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
