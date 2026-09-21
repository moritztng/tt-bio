#!/usr/bin/env python3
"""What fraction of OpenFold3's gradient mass this campaign can actually compare, and what
blocks the rest -- one census over the reference, per tensor, by mass.

The campaign has been quoting "about 2 % has no reading, and part of it never can". Neither
half is read off an artifact. This reads both off artifacts in ONE denominator and gives every
uncompared tensor a named blocker instead of rounding it into a section total.

DENOMINATOR. BUNDLE-MIN-043's float64 reference gradient `grads_f64_043.pt`, pinned by digest.
A tensor's mass is its own squared float64 gradient 2-norm; the model's mass is the sum over
every tensor that HAS a gradient there. That is the same denominator
`of3t-wholemodel/model_scope.py` builds, recomputed here from the file rather than copied, so
the comparable share below is an independent reproduction of `MODEL_shipped.json`'s
`coverage_total`, not a restatement of it.

BLOCKERS, in precedence order. A tensor can have several; it is filed under the one that has to
be removed FIRST, and where two stack the doc says so.

  COMPARED                 a per-tensor device-vs-float64 reading on the model's own batch,
                           batch_step003 at crop 384. The number this campaign quotes.
  READ_ON_ANOTHER_BOUNDARY a per-tensor device gradient exists and was taken at the captured
                           64-token block boundary, not on the model's batch. Not comparable in
                           one vector with the rest -- concatenating gradients of two different
                           inputs is not a gradient -- but read, and read per THEIR tensor
                           including the fused ones.
  HOST_APPLIED             the shipped forward applies the weight on the HOST after a
                           `ttnn.to_torch`, so the cotangent cannot return through the graph the
                           forward severed. No device gradient exists.
  NOT_A_LEAF_LAZY          the device tensor is materialised inside `forward`, after the walk
                           that registers trainable leaves has already run, so it is never a
                           leaf and never receives a gradient. The weight is on the card and the
                           arithmetic is performed; nothing reads its gradient because none is
                           produced.
  FUSED_NOT_SPLIT          carried inside a fused device tensor whose leaf DOES receive a
                           gradient, and no instrument in this scope applies the inverse
                           selection. The trunk's instrument does exactly that for its own
                           fusion; the diffusion transformer's has not been given one.
  NO_ARM                   nothing structural is in the way. No device arm covering that scope
                           on the model's batch has been run.

NO_ARM is the only class that means "we have not compared it yet". Every other class names
something that has to change in the port or in an instrument before a reading exists at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

# The 14 weights `OF3AtomTransformer` resolves through `_w_tt` inside `forward`
# (tt_bio/openfold3_atom_transformer.py:130, 140-141, 143-144, 179-181, 194, 202, 211-212, 215).
# Its other weights -- `layer_norm_z.weight` at :70 and the nine `AdaLN` submodules at :71-79 --
# are built in `__init__` and are leaves. That split is not asserted here: it is checkable
# against the census, and the counts land exactly (42 of a module's 79 atom-transformer tensors
# uncarried, 37 carried).
LAZY_ATOM_TRANSFORMER_LEAVES = (
    "attention_pair_bias.linear_z.weight",
    "attention_pair_bias.linear_ada_out.weight",
    "attention_pair_bias.linear_ada_out.bias",
    "attention_pair_bias.mha.linear_q.weight",
    "attention_pair_bias.mha.linear_q.bias",
    "attention_pair_bias.mha.linear_k.weight",
    "attention_pair_bias.mha.linear_v.weight",
    "attention_pair_bias.mha.linear_g.weight",
    "attention_pair_bias.mha.linear_o.weight",
    "conditioned_transition.linear_g.weight",
    "conditioned_transition.linear_g.bias",
    "conditioned_transition.swiglu.linear_a.weight",
    "conditioned_transition.swiglu.linear_b.weight",
    "conditioned_transition.linear_out.weight",
)
LAZY_RE = re.compile(r"\.atom_transformer\.blocks\.\d+\.(?P<leaf>.+)$")

# `openfold3_diffusion_transformer.py:122-140` concatenates these four into one padded
# `qkv_w` / `qkv_b` in `__init__`. The fused tensor IS a registered leaf and does receive a
# gradient; what is missing is the inverse selection that hands each of their four tensors its
# own slice, which `of3t-trunkg043/dev_grad.py:apb_inverse` already performs for the identical
# fusion in `tenstorrent.AttentionPairBias`.
DIT_FUSED_LEAVES = (
    "attention_pair_bias.mha.linear_q.weight",
    "attention_pair_bias.mha.linear_q.bias",
    "attention_pair_bias.mha.linear_k.weight",
    "attention_pair_bias.mha.linear_v.weight",
)
DIT_RE = re.compile(r"^diffusion_module\.diffusion_transformer\.blocks\.\d+\.(?P<leaf>.+)$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def section_of(name: str) -> str:
    p = name.split(".")
    if p[0] == "diffusion_module" and len(p) > 1:
        return f"{p[0]}.{p[1]}"
    return p[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True, type=Path)
    ap.add_argument("--expect-f64", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--compared", required=True, type=Path)
    ap.add_argument("--other-boundary", required=True, type=Path,
                    help="the trunk arm's gradient dump, keyed by THEIR tensor name")
    ap.add_argument("--other-boundary-key", default="grads")
    ap.add_argument("--host-applied", action="append", default=[],
                    help="exact reference tensor name the shipped forward applies on the host")
    ap.add_argument("--host-applied-prefix", action="append", default=[],
                    help="prefix of a whole submodule the shipped forward applies on the host")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    got = sha256(args.f64)
    if got != args.expect_f64:
        raise SystemExit(f"STOP: {args.f64} is {got}, the pin says {args.expect_f64}")

    g = torch.load(args.f64, map_location="cpu", weights_only=False)
    mass = {n: float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
            for n, v in g.items() if v is not None}
    n_none = sum(1 for v in g.values() if v is None)
    del g
    total = sum(mass.values())

    ck = torch.load(args.checkpoint, map_location="meta", weights_only=False, mmap=True)
    ck = ck.get("state_dict", ck)
    ck_names = set(ck.keys())
    del ck
    no_gradient = sorted(ck_names - set(mass))
    if set(mass) - ck_names:
        raise SystemExit("STOP: the reference carries a tensor the checkpoint does not")

    compared = {e["param"] for e in json.loads(args.compared.read_text())}
    if compared - set(mass):
        raise SystemExit("STOP: a compared tensor is absent from the reference")

    ob = torch.load(args.other_boundary, map_location="meta", weights_only=False, mmap=True)
    ob = ob[args.other_boundary_key]
    other = {n for n, v in ob.items() if v is not None}
    del ob

    host_applied = set(args.host_applied)
    host_applied |= {n for n in mass
                     if any(n.startswith(p) for p in args.host_applied_prefix)}
    missing_host = set(args.host_applied) - set(mass)
    if missing_host:
        raise SystemExit(f"STOP: host-applied name not in the reference: {sorted(missing_host)}")

    cls, both = {}, {}
    for n in mass:
        m = LAZY_RE.search(n)
        lazy = bool(m and m.group("leaf") in LAZY_ATOM_TRANSFORMER_LEAVES)
        d = DIT_RE.match(n)
        dit = bool(d and d.group("leaf") in DIT_FUSED_LEAVES)
        if n in compared:
            cls[n] = "COMPARED"
        elif n in other:
            cls[n] = "READ_ON_ANOTHER_BOUNDARY"
        elif n in host_applied:
            cls[n] = "HOST_APPLIED"
        elif lazy:
            cls[n] = "NOT_A_LEAF_LAZY"
            if section_of(n) in ("input_embedder",):
                both[n] = "also NO_ARM: no device arm covers input_embedder on the model's batch"
        elif dit:
            cls[n] = "FUSED_NOT_SPLIT"
        else:
            cls[n] = "NO_ARM"

    by_class = defaultdict(lambda: {"n": 0, "mass_sq": 0.0,
                                    "sections": defaultdict(lambda: {"n": 0, "mass_sq": 0.0})})
    for n, c in cls.items():
        b = by_class[c]
        b["n"] += 1
        b["mass_sq"] += mass[n]
        s = b["sections"][section_of(n)]
        s["n"] += 1
        s["mass_sq"] += mass[n]
    classes = {c: {"n_tensors": b["n"], "mass_sq": b["mass_sq"],
                   "pct_of_model_mass": 100.0 * b["mass_sq"] / total,
                   "by_section": {s: {"n_tensors": v["n"],
                                      "pct_of_model_mass": 100.0 * v["mass_sq"] / total}
                                  for s, v in sorted(b["sections"].items(),
                                                     key=lambda kv: -kv[1]["mass_sq"])}}
               for c, b in sorted(by_class.items(), key=lambda kv: -kv[1]["mass_sq"])}

    worst = lambda c: [{"param": n, "pct_of_model_mass": 100.0 * m / total}
                       for m, n in sorted(((mass[n], n) for n, k in cls.items() if k == c),
                                          reverse=True)[:10]]

    doc = {
        "instrument": "of3t-readable-mass readable_mass.py -- one census of the reference "
                      "gradient, per tensor, by mass, every uncompared tensor given a blocker",
        "inputs": {
            "float64_reference": {"path": str(args.f64), "sha256": got,
                                  "bytes": args.f64.stat().st_size},
            "checkpoint": {"path": str(args.checkpoint), "n_tensors": len(ck_names)},
            "compared_from": str(args.compared),
            "other_boundary_from": str(args.other_boundary),
            "host_applied_names": sorted(args.host_applied),
            "host_applied_prefixes": sorted(args.host_applied_prefix),
            "host_applied_resolved_to": len(host_applied),
        },
        "denominator": {
            "model_squared_gradient_norm": total,
            "n_reference_tensors_with_a_gradient": len(mass),
            "n_reference_entries_that_are_None": n_none,
            "n_checkpoint_tensors": len(ck_names),
            "n_checkpoint_tensors_with_no_gradient": len(no_gradient),
            "rule": "mass is the tensor's own squared float64 gradient 2-norm. A checkpoint "
                    "tensor the reference step gives no gradient carries no mass and is "
                    "outside the denominator -- counted, never given a share, because a share "
                    "of zero reads as 'small' where the right word is 'absent'.",
        },
        "headline": {
            "pct_comparable_today": 100.0 * sum(mass[n] for n in compared) / total,
            "n_compared": len(compared),
            "pct_not_comparable_today": 100.0 * sum(mass[n] for n in mass
                                                    if n not in compared) / total,
            "n_not_compared": len(mass) - len(compared),
            "pct_with_a_per_tensor_device_gradient_somewhere":
                100.0 * sum(mass[n] for n in mass if n in compared or n in other) / total,
        },
        "classes": classes,
        "two_blockers": both,
        "worst_by_class": {c: worst(c) for c in classes if c != "COMPARED"},
        "no_gradient_by_top_level": {
            t: sum(1 for n in no_gradient if n.split(".")[0] == t)
            for t in sorted({n.split(".")[0] for n in no_gradient})},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"headline": doc["headline"], "denominator": doc["denominator"],
                      "classes": {c: {"n": v["n_tensors"], "pct": v["pct_of_model_mass"]}
                                  for c, v in classes.items()},
                      "no_gradient_by_top_level": doc["no_gradient_by_top_level"]}, indent=2))


if __name__ == "__main__":
    main()
