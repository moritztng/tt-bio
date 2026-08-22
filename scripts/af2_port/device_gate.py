"""Score the ttnn AF2 components against the torch reference's own activations.

Card-bound, upstream-free and JAX-free. It runs `tt_bio.af2_reference` once on the committed
fixture inputs, captures what each component was handed and what it returned, then hands the
same input to the ttnn component and asks whether the device is any further from the reference
than a second precision realisation of the reference is.

**The envelope is measured per run, not asserted.** For a component `f` and a captured bfloat16
input `x`:

    rms_envelope = rms(f_fp32(x) - f_bf16(x))        rms_device = rms(f_ttnn(x) - f_bf16(x))

Both torch arms are the same module object on the same input -- the reference keeps its
parameters in float32 and casts per call, so the only difference between the arms is the
arithmetic. The envelope is therefore the model's own precision freedom at that site and nothing
else, and a transcription error cannot hide inside it: a wrong module is wrong in both dtypes,
the two torch arms agree, and the envelope collapses.

`--bar envelope` demands `rms_device <= rms_envelope` at every site. No correct ttnn
transcription reaches that: ttnn is a THIRD arithmetic realisation, not a variant of the two
torch ones the envelope is built from, and 9 of 10 correctly-ported pair ops miss it (measured,
pass 9). `--bar chained` is the bar that replaced it, at the boundary it was always meaningful
at:

    chained component   ratio = rms_device / rms_envelope <= 1.5  AND  pcc >= 0.9999
    single op           ratio <= 3.0, printed either way and adjudicated above it
    --mutate <name>     the SAME numbers, inverted: ratio >= 5.0 or the instrument is blind

Correct code measured 0.998x and 1.071x chained with pcc 0.9999948; the mildest real
transcription error measured 9.0x and pcc 0.9993; the worst correct single op measured 2.784x.
`--mutate` is the part that matters. It breaks one thing on the DEVICE arm only and requires the
gate to see it, so a-c establish something at each new site instead of being asserted there.

`--block` takes the first AND the last Evoformer block on purpose. A wrong residual or a swapped
left/right shows at block 0; an accumulating precision fault only shows at block 47.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \
        env/bin/python3 scripts/af2_port/device_gate.py --component evo-block --bar chained
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

# The two tracks' members in the order a block runs them.
PAIR_TRACK = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition")
MSA_TRACK = ("msa_row_attn", "msa_col_attn", "msa_transition", "opm")

# `*-ops` scores each member on its own captured input; the chained components run the real
# device code end to end and are the ones the bar is written for.
COMPONENTS = {
    "pair-ops": PAIR_TRACK,
    "pair-block": ("pair-track",),
    "msa-ops": MSA_TRACK,
    "msa-block": ("msa-track",),
    "evo-block": ("evo-block",),
}
CHAINED = ("pair-track", "msa-track", "evo-block")

BAR_CHAINED_RATIO = 1.5
BAR_CHAINED_PCC = 0.9999
BAR_OP_RATIO = 3.0
BAR_MUTATION_RATIO = 5.0

# One deliberate mutation per component, each breaking a different KIND of thing: a bias that
# never reaches the softmax, an attention over the wrong axis, two swapped projections, a
# dropped residual. Applied to the device arm only; the torch arms stay correct, so a live
# instrument reads a large ratio and a blind one does not.
MUTATIONS = ("msa_row_attn", "msa_col_attn", "opm", "msa-track")

# Which captured arguments are masks the port asserts are all ones. Everything not listed has
# its mask at index 1.
MASK_ARGS = {"evo-block": (2, 3), "pair_transition": (), "msa_transition": ()}


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).pow(2).mean().sqrt())


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.double().flatten(), b.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))


class Capture:
    """The inputs and outputs of every track member of the requested blocks.

    Keyed by the last call, so with recycles on it is the last pass -- the one whose inputs carry
    the most accumulated magnitude.
    """

    def __init__(self, model, blocks: list[int]) -> None:
        self.data: dict[tuple[int, str], dict] = {}
        self.handles = []
        for index in blocks:
            block = model.evoformer[index]
            for name in (*PAIR_TRACK, *MSA_TRACK, "block"):
                target = block if name == "block" else getattr(block, name)
                self.handles.append(target.register_forward_hook(self._hook(index, name)))

    def _hook(self, index: int, name: str):
        def fn(_module, args, output):
            self.data[(index, name)] = {"args": args, "out": output}
        return fn

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def torch_arm(module, args, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """One precision realisation of a captured call: the same module, the same numbers."""
    cast = tuple(a.to(dtype) if torch.is_tensor(a) and a.is_floating_point() else a
                 for a in args)
    with torch.no_grad():
        out = module(*cast)
    return tuple(t.float() for t in (out if isinstance(out, tuple) else (out,)))


def score(name: str, got: torch.Tensor, want: torch.Tensor, envelope: torch.Tensor,
          kind: str, bar: str, mutated: bool) -> dict:
    row = {"component": name, "kind": kind, "shape": tuple(want.shape),
           "rms_device": rms(got, want), "rms_envelope": rms(envelope, want),
           "pcc_device": pcc(got, want), "std_ref": float(want.double().std())}
    row["ratio"] = row["rms_device"] / max(row["rms_envelope"], 1e-30)
    if mutated:
        row["bar"] = f"ratio >= {BAR_MUTATION_RATIO}"
        ok = row["ratio"] >= BAR_MUTATION_RATIO
    elif bar == "chained":
        limit = BAR_CHAINED_RATIO if kind == "chained" else BAR_OP_RATIO
        ok = row["ratio"] <= limit
        row["bar"] = f"ratio <= {limit}"
        if kind == "chained":
            ok = ok and row["pcc_device"] >= BAR_CHAINED_PCC
            row["bar"] += f" and pcc >= {BAR_CHAINED_PCC}"
    else:
        row["bar"] = "rms_device <= rms_envelope"
        ok = row["rms_device"] <= row["rms_envelope"]
    row["verdict"] = "PASS" if ok else "FAIL"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="pair-block", choices=sorted(COMPONENTS))
    ap.add_argument("--block", default="0,47")
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--bar", default="envelope", choices=["envelope", "chained"])
    ap.add_argument("--mutate", default=None, choices=MUTATIONS,
                    help="break one thing on the device arm; the bar inverts to ratio >= 5.0 "
                         "and only the mutated member is scored")
    ap.add_argument("--opm-eps", type=float, default=0.0,
                    help="add an epsilon to the outer product mean's divisor: n_msa = depth + "
                         "eps. 0 is the default and what the port ships, and lets the op read "
                         "the depth off the tensor; --opm-eps 1e-3 is AF2's own epsilon, which "
                         "is below bfloat16 resolution and measures 1.7x worse")
    ap.add_argument("--recycles", type=int, default=0,
                    help="extra recycles before the capture; 0 is one pass, which already "
                         "carries the 48-block accumulation the gate measures")
    args = ap.parse_args()

    import ttnn

    from tt_bio.af2 import (AF2Attention, AF2EvoformerBlock, AF2PairBlock,
                            compute_kernel_config, get_device)
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

    members = COMPONENTS[args.component]
    if args.mutate:
        assert args.mutate in members, f"--mutate {args.mutate} is not in --component {args.component}"
        members = (args.mutate,)

    device = get_device()
    ckc = compute_kernel_config()

    def up(t: torch.Tensor) -> "ttnn.Tensor":
        return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                               device=device, dtype=ttnn.bfloat16)

    def down(t: "ttnn.Tensor", shape) -> torch.Tensor:
        x = torch.Tensor(ttnn.to_torch(t)).float()
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        assert tuple(x.shape) == tuple(shape), f"device gave {tuple(x.shape)}, want {tuple(shape)}"
        return x

    def scoped(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    rows = []
    for index in blocks:
        scope = scoped(f"evoformer.{index}.")
        if args.mutate == "msa_row_attn":
            # The pair bias never reaches the softmax at all.
            scope["msa_row_attn.linear.weight"] = torch.zeros_like(
                scope["msa_row_attn.linear.weight"])
        if args.mutate == "opm":
            for field in ("weight", "bias"):
                a, b = f"opm.proj_a.{field}", f"opm.proj_b.{field}"
                scope[a], scope[b] = scope[b], scope[a]
        pair_only = args.component.startswith("pair-")
        tt_block = (AF2PairBlock if pair_only else AF2EvoformerBlock)(scope, ckc)
        if args.mutate == "msa_col_attn":
            # Attend over the residue axis: the shipped constructor flag, no debug hook.
            tt_block.msa_col_attn = AF2Attention(
                {k[len("msa_col_attn."):]: v for k, v in scope.items()
                 if k.startswith("msa_col_attn.")}, ckc, column=False)
        if args.mutate == "msa-track":
            # A zero update is exactly a dropped residual.
            tt_block.msa_transition = ttnn.zeros_like
        torch_block = model.evoformer[index]
        cap = capture.data

        for name in members:
            if name == "pair-track":
                # The pair track's input is the pair after the outer product mean, which is
                # exactly what `tri_mul_out` was handed.
                margs = cap[(index, "tri_mul_out")]["args"]
                labels, module = ("pair",), torch_block._pair_track
                want = (cap[(index, "block")]["out"][1],)
                got_tt = (tt_block(up(margs[0])),)
            elif name == "msa-track":
                margs = cap[(index, "msa_row_attn")]["args"]      # (msa, msa_mask, pair)
                # The chained MSA track's output is what the outer product mean was handed.
                want = (cap[(index, "opm")]["args"][0],)
                labels = ("msa",)

                def module(msa, msa_mask, pair, _b=torch_block):
                    msa = msa + _b.msa_row_attn(msa, msa_mask, pair)
                    msa = msa + _b.msa_col_attn(msa, msa_mask)
                    return msa + _b.msa_transition(msa)

                got_tt = (tt_block._msa_track(up(margs[0]), up(margs[2])),)
            elif name == "evo-block":
                margs = cap[(index, "block")]["args"]             # (msa, pair, msa_mask, mask)
                labels, module = ("msa", "pair"), torch_block
                want = tuple(cap[(index, "block")]["out"])
                got_tt = tt_block(up(margs[0]), up(margs[1]))
            else:
                margs = cap[(index, name)]["args"]
                labels, module = (name,), getattr(torch_block, name)
                want = (cap[(index, name)]["out"],)
                tt_call = getattr(tt_block, name)
                if name == "msa_row_attn":
                    got_tt = (tt_call(up(margs[0]), up(margs[2])),)
                elif name == "opm":
                    n_msa = (None if not args.opm_eps
                             else float(margs[0].shape[0]) + args.opm_eps)
                    got_tt = (tt_call(up(margs[0]), None, n_msa),)
                else:
                    got_tt = (tt_call(up(margs[0])),)

            for i in MASK_ARGS.get(name, (1,)):
                # `tt_bio/af2.py` says why an all-ones mask is a property of this port, not a
                # shortcut, and what a genuinely masked AF2 fold would need first.
                assert bool((margs[i] == 1).all()), "a masked AF2 fold is not wired up"
            want = tuple(t.float() for t in want)
            envelope = torch_arm(module, margs, torch.float32)
            kind = "chained" if name in CHAINED else "op"
            for label, g, w, e in zip(labels, got_tt, want, envelope):
                rows.append(score(f"block{index}/{name}/{label}" if len(labels) > 1
                                  else f"block{index}/{name}",
                                  down(g, w.shape), w, e, kind, args.bar, bool(args.mutate)))

    failed = [r for r in rows if r["verdict"] != "PASS"]
    report = {"mode": "af2ig_device", "component": args.component, "stage": args.stage,
              "bar": args.bar, "mutate": args.mutate, "opm_eps": args.opm_eps,
              "blocks": blocks, "recycles": args.recycles,
              "verdict": "PASS" if rows and not failed else "FAIL",
              "scored": len(rows), "failed": len(failed), "rows": rows}
    print(json.dumps(report, indent=1, default=float))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
