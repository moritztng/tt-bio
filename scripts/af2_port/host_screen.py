"""W6 and W7's two decisions, both host-only: what the template embedding costs per design, and
whether the extra-MSA stack's MSA track can be deleted.

Neither needs a card. Both decide the port's scope, so they are measured once here and recorded
rather than re-derived by whoever writes the ttnn code. One command, and its output is the
decision.

**W6 -- the template embedding.** It is called once per recycling pass, so the rule fixed in
advance (`state/pxdesign-af2ig-port.md` §W6) is 1.0 s per design over four passes: under that it
stays on host, at or above it gets ported. The measurement here also asks a question the rule did
not: whether the four calls have to happen at all. With a single template -- which is what
PXDesign runs, the design's own coordinates -- the template cross-attention softmaxes over one
key, so its weights are exactly 1.0, the query never reaches the output, and the whole module is
constant in its `pair` argument. Everything else it reads (`feats`, the two masks) is constant
across a design's recycles too. So the port computes it ONCE and reuses it, which is bit-exact
rather than an approximation, and the number the rule is applied to is one call, not four.

The cost is O(L^3) in the pair stack, so the verdict is size-conditioned and the number is
printed with the sequence length it was measured at.

**W7 -- the extra-MSA stack's MSA track.** `extra_msa_mask` is all zeros in this featurisation
(PXDesign runs AF2-IG single-sequence, so there is no extra MSA). `OuterProductMean` multiplies
both projections by that mask, so `outer` and `norm` are exactly zero and its output is
`proj_o(0) / (eps + 0)` -- the bias over the epsilon, a per-channel constant independent of its
`msa` argument. The outer product mean is the stack's only msa-to-pair path and the stack's msa
output is discarded (`af2_reference.AF2Model.forward`), so the whole MSA track is dead code here.
The screen proves that instead of asserting it: four real blocks against four blocks that never
touch the MSA track and inject the constant, and the post-stack pair must be **bit-exact**.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tap_gate import ARTIFACTS, DEFAULT_PARAMS, load_inputs  # noqa: E402

TEMPLATE_BUDGET_S = 1.0   # per design. The rule, fixed in advance, applied to whichever
PASSES_PER_DESIGN = 4     # call count the port actually pays.


class _Stop(Exception):
    """Abort the forward pass once everything this script needs has been captured."""


def _capture(model, store: dict):
    """Hooks enough of one pass to serve both measurements, then stops it.

    The template embedding and the four extra-MSA blocks all run before the 48 Evoformer blocks
    and the structure module, so aborting at the end of `extra_msa[-1]` costs seconds instead of
    the minutes a full CPU pass takes.
    """
    handles = []
    if model.template is not None:
        handles.append(model.template.register_forward_hook(
            lambda _m, args, out: store.__setitem__("template_args", args)))
    handles.append(model.extra_msa[0].register_forward_pre_hook(
        lambda _m, args: store.__setitem__("extra_args", args)))

    def last(_m, _args, out):
        store["extra_out"] = out
        raise _Stop

    handles.append(model.extra_msa[-1].register_forward_hook(last))
    return handles


def _clock(fn, args, repeat: int) -> list[float]:
    """Wall seconds per call, first call discarded: it pays torch's one-time allocator and
    kernel-selection cost, which a design pays once and not once per call.

    Both the min and the median are reported. On a shared host the two can differ by 4x -- this
    box carried loadavg 25 of 32 cores from an unrelated release gate while W6 was measured, and
    the same call read 0.44 s and 1.93 s in two runs an hour apart. Contention only ever adds
    time, so the min over enough repeats is the estimator of what a design actually pays on a
    fold host, and the median is what it pays while sharing the box. The decision is reported
    under both.
    """
    samples = []
    with torch.no_grad():
        for _ in range(repeat + 1):
            start = time.perf_counter()
            fn(*args)
            samples.append(time.perf_counter() - start)
    return samples[1:]


def time_template(model, feats, store, repeat: int) -> dict:
    args = store["template_args"]
    pair = args[0]
    dtype = model.trunk_dtype

    # Is the module constant in `pair`? Two arbitrary substitutes, bit-exact or the claim is
    # void. Also the fp32 arm, so the answer cannot be a bfloat16 rounding artifact.
    with torch.no_grad():
        base = model.template(*args)
        rest = args[1:]
        independent = all(torch.equal(base, model.template(sub, *rest)) for sub in (
            torch.zeros_like(pair), (torch.randn_like(pair.float()) * 3.0).to(pair.dtype)))
        base32 = model.template(pair.float(), *rest)
        independent32 = torch.equal(base32,
                                    model.template(torch.zeros_like(pair.float()), *rest))
        # And the degenerate attention IS two chained linears, exactly.
        repr_ = model.template.pair_representation(*args)
        num_res, c_z = pair.shape[0], pair.shape[-1]
        num_templ, c_t = repr_.shape[0], repr_.shape[-1]
        flat = repr_.permute(1, 2, 0, 3).reshape(num_res * num_res, num_templ, c_t)
        value = model.template.attn.linear_v(flat)
        direct = model.template.attn.linear_o(
            value.reshape(num_res * num_res, num_templ, -1)).reshape(num_res, num_res, c_z)
        direct = direct * (feats["template_mask"].sum() > 0).to(direct.dtype)

    # The pair stack is measured on its own too: it is the O(L^3) part, and the only part a
    # device port would move (`AF2PairBlock` already exists; the degenerate attention is two
    # linears and the 88-channel input build is host featurisation either way).
    embed = _clock(model.template, args, repeat)
    stack = _clock(model.template.pair_representation, args, repeat)
    torsion = _clock(model.template.torsion_rows, (feats, dtype), repeat)

    def median(xs):
        return sorted(xs)[len(xs) // 2]

    calls = 1 if independent else PASSES_PER_DESIGN
    out = {"num_res": num_res, "num_templates": num_templ,
           "threads": torch.get_num_threads(), "loadavg": os.getloadavg(),
           "repeat": repeat,
           "embedding_s": embed, "pair_stack_s": stack, "torsion_rows_s": torsion,
           "pair_independent": bool(independent), "pair_independent_fp32": bool(independent32),
           "attention_is_two_linears": bool(torch.equal(base, direct)),
           "calls_per_design": calls, "budget_s": TEMPLATE_BUDGET_S}
    for label, pick in (("min", min), ("median", median)):
        per_call = pick(embed) + pick(torsion)
        out[f"{label}_per_call_s"] = per_call
        out[f"{label}_pair_stack_s"] = pick(stack)
        out[f"{label}_per_design_s"] = per_call * calls
        out[f"{label}_per_design_s_if_uncached"] = per_call * PASSES_PER_DESIGN
        out[f"decision_by_{label}"] = ("host" if per_call * calls < TEMPLATE_BUDGET_S
                                       else "port")
    out["decision"] = ("host" if out["decision_by_min"] == out["decision_by_median"] == "host"
                       else "port" if out["decision_by_min"] == "port" else "contended")
    return out


def screen_extra_msa(model, store) -> dict:
    """The dead-track screen. Bit-exact or it is not taken."""
    extra, pair, extra_mask, mask_2d = store["extra_args"]
    want_msa, want_pair = store["extra_out"]
    assert bool((extra_mask == 0).all()), "extra_msa_mask is not all zeros; the screen is void"

    rows = []
    simplified = pair
    for index, block in enumerate(model.extra_msa):
        with torch.no_grad():
            const = block.opm(torch.zeros_like(extra), extra_mask)
            from_real = block.opm(extra, extra_mask)
        flat = const.reshape(-1, const.shape[-1])
        channel = flat[0]
        rows.append({
            "block": index,
            # (1) the outer product mean ignores its msa argument under a zero mask,
            "opm_input_independent": bool(torch.equal(const, from_real)),
            # (2) and its output is one vector repeated over every pair position,
            "opm_spatially_constant": bool(torch.equal(flat, channel.expand_as(flat))),
            # (3) which is exactly proj_o's bias over the epsilon, in the trunk dtype.
            "equals_bias_over_eps": bool(torch.equal(
                channel, (block.opm.proj_o.bias.to(const.dtype)
                          / (block.opm.eps + torch.zeros((), dtype=const.dtype))))),
            "constant_absmax": float(channel.abs().max()),
            "constant_head": [float(v) for v in channel[:4]],
        })
        with torch.no_grad():
            simplified = block._pair_track(simplified + const, mask_2d)

    exact = bool(torch.equal(simplified, want_pair))
    return {"blocks": rows,
            "post_stack_bit_exact": exact,
            "post_stack_max_abs_delta": float((simplified.float()
                                               - want_pair.float()).abs().max()),
            "pair_shape": tuple(want_pair.shape), "msa_shape": tuple(want_msa.shape),
            "verdict": "DEAD" if exact and all(r["opm_input_independent"]
                                               and r["opm_spatially_constant"]
                                               for r in rows) else "ALIVE"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="complex", choices=["complex", "monomer"])
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--repeat", type=int, default=4)
    args = ap.parse_args()

    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict

    suffix = "" if args.stage == "complex" else "_monomer"
    feats, prev = load_inputs(ARTIFACTS / f"ref_inputs{suffix}.npz")
    template = args.stage == "complex"
    model = load_af2_model(load_af2_state_dict(args.params), template=template,
                           trunk_dtype=torch.bfloat16)

    store: dict = {}
    handles = _capture(model, store)
    try:
        with torch.no_grad():
            model(feats, prev)
    except _Stop:
        pass
    finally:
        for handle in handles:
            handle.remove()

    w6 = (time_template(model, feats, store, args.repeat) if template
          else "no template stack in the monomer config")
    w7 = screen_extra_msa(model, store)
    print(json.dumps({"mode": "af2ig_host_screen", "stage": args.stage,
                      "w6_template": w6, "w7_extra_msa": w7}, indent=1, default=float))
    return 0 if w7["verdict"] == "DEAD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
