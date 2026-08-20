"""Score the ttnn AF2 components against the torch reference's own activations.

Card-bound, upstream-free and JAX-free. It runs `tt_bio.af2_reference` once on the committed
fixture inputs, captures what each component was handed and what it returned, then hands the
same input to the ttnn component and asks whether the device is any further from the reference
than a second precision realisation of the reference is.

**The bar is measured per run, not asserted.** For a component `f` and a captured bfloat16
input `x`:

    accept  iff  rms(f_ttnn(x) - f_bf16(x))  <=  rms(f_fp32(x) - f_bf16(x))

Both torch arms are the same module object on the same input -- the reference keeps its
parameters in float32 and casts per call, so the only difference between the arms is the
arithmetic. The envelope is therefore the model's own precision freedom at that site and nothing
else, and a transcription error cannot hide inside it: a wrong module is wrong in both dtypes,
the two torch arms agree, and the envelope collapses.

`--block` takes the first AND the last Evoformer block on purpose. A wrong residual or a swapped
left/right shows at block 0; an accumulating precision fault only shows at block 47.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \
        env/bin/python3 scripts/af2_port/device_gate.py --component pair-block --block 0,47
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tap_gate import ARTIFACTS, DEFAULT_PARAMS, load_inputs  # noqa: E402

# The pair track's members in the order a block runs them. `pair-block` scores the whole track
# chained on device; `pair-ops` scores each member on its own captured input.
PAIR_TRACK = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition")


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).pow(2).mean().sqrt())


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.double().flatten(), b.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))


class Capture:
    """The inputs and outputs of the pair-track members of the requested blocks.

    Keyed by the last call, so with recycles on it is the last pass -- the one whose inputs carry
    the most accumulated magnitude.
    """

    def __init__(self, model, blocks: list[int]) -> None:
        self.data: dict[tuple[int, str], dict] = {}
        self.handles = []
        for index in blocks:
            block = model.evoformer[index]
            for name in (*PAIR_TRACK, "block"):
                target = block if name == "block" else getattr(block, name)
                self.handles.append(target.register_forward_hook(self._hook(index, name)))

    def _hook(self, index: int, name: str):
        def fn(_module, args, output):
            self.data[(index, name)] = {"args": args, "out": output}
        return fn

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def torch_arm(module, args, dtype: torch.dtype) -> torch.Tensor:
    """One precision realisation of a captured call: the same module, the same numbers."""
    cast = tuple(a.to(dtype) if torch.is_tensor(a) and a.is_floating_point() else a
                 for a in args)
    with torch.no_grad():
        return module(*cast).float()


def score(name: str, got: torch.Tensor, want: torch.Tensor,
          envelope: torch.Tensor) -> dict:
    row = {"component": name, "shape": tuple(want.shape),
           "rms_device": rms(got, want), "rms_envelope": rms(envelope, want),
           "pcc_device": pcc(got, want), "std_ref": float(want.double().std())}
    row["ratio"] = row["rms_device"] / max(row["rms_envelope"], 1e-30)
    row["verdict"] = "PASS" if row["rms_device"] <= row["rms_envelope"] else "FAIL"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="pair-block", choices=["pair-block", "pair-ops"])
    ap.add_argument("--block", default="0,47")
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--recycles", type=int, default=0,
                    help="extra recycles before the capture; 0 is one pass, which already "
                         "carries the 48-block accumulation the gate measures")
    args = ap.parse_args()

    import ttnn

    from tt_bio.af2 import AF2PairBlock, compute_kernel_config, get_device
    from tt_bio.af2_reference import load_af2_model, run_recycles
    from tt_bio.af2_weights import load_af2_state_dict

    suffix = "" if args.stage == "complex" else "_monomer"
    feats, prev = load_inputs(ARTIFACTS / f"ref_inputs{suffix}.npz")
    state = load_af2_state_dict(args.params)
    model = load_af2_model(state, template=args.stage == "complex",
                           trunk_dtype=torch.bfloat16)
    blocks = [int(b) for b in args.block.split(",")]
    capture = Capture(model, blocks)
    run_recycles(model, feats, prev, num_recycles=args.recycles)
    capture.remove()

    device = get_device()
    ckc = compute_kernel_config()
    members = PAIR_TRACK if args.component == "pair-ops" else ("block",)
    rows = []
    for index in blocks:
        prefix = f"evoformer.{index}."
        scope = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        tt_block = AF2PairBlock(scope, ckc)
        torch_block = model.evoformer[index]
        for name in members:
            if name == "block":
                # The block's own hook sees (msa, pair, msa_mask, pair_mask) and returns
                # (msa, pair). The pair TRACK's input is the pair after the outer product mean,
                # which is exactly what `tri_mul_out` was handed.
                entry = capture.data[(index, "tri_mul_out")]
                arm_args = entry["args"]
                want = capture.data[(index, "block")]["out"][1].float()
                module, tt_call = torch_block._pair_track, tt_block
            else:
                arm_args = capture.data[(index, name)]["args"]
                want = capture.data[(index, name)]["out"].float()
                module, tt_call = getattr(torch_block, name), getattr(tt_block, name)
            if len(arm_args) > 1:
                # The fixture's pair mask is all ones, so it is dropped rather than uploaded.
                # `tt_bio/af2.py` says why that is a property of this port, not a shortcut.
                assert bool((arm_args[1] == 1).all()), "a masked AF2 fold is not wired up"
            z = ttnn.from_torch(arm_args[0].unsqueeze(0).to(torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            got = torch.Tensor(ttnn.to_torch(tt_call(z))).float().squeeze(0)
            rows.append(score(f"block{index}/{name}", got, want,
                              torch_arm(module, arm_args, torch.float32)))

    failed = [r for r in rows if r["verdict"] != "PASS"]
    report = {"mode": "af2ig_device", "component": args.component, "stage": args.stage,
              "blocks": blocks, "recycles": args.recycles,
              "verdict": "PASS" if rows and not failed else "FAIL",
              "scored": len(rows), "failed": len(failed), "rows": rows}
    print(json.dumps(report, indent=1, default=float))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
