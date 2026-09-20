#!/usr/bin/env python3
"""PROTOCOL SS3a: the bijection manifest, derived empirically rather than read off the source.

SS3a requires this published BEFORE any gradient is compared: every one of their parameter
tensors mapped to exactly one (our tensor, slice) pair, with the unmapped set on BOTH sides
enumerated and individually explained. The reason is R3 -- our OF3 is an independent ttnn
reimplementation and the remap FUSES, `protenix_weights.py:25-26` concatenating their
`linear_a_g.weight` and `linear_b_g.weight` into our single `g_in.weight` along dim 0. A
comparison made over a map that silently drops tensors is the same defect as a mean hiding
an outlier, and a fused comparison lets one half's agreement mask the other half's error.

METHOD. The map is not read out of the remap source and transcribed, because a transcription
is an assertion. Each of their tensors is replaced by a TRACER -- a float64 tensor of its own
shape filled with a unique integer id -- and the real remap functions are then run on the
tracer state dict. Every tensor we produce is afterwards decoded back into the ids it is made
of and the dim-0 span each id occupies. So the manifest is whatever the shipped code actually
does, including any fusion nobody documented.

float64 tracers are deliberate: ids run past 4,900 and bf16 carries 8 mantissa bits, so an id
above 256 would not survive in the checkpoint's own dtype and the decode would silently alias.

SCOPE, stated honestly. tt-bio has no whole-checkpoint remap; each device module calls
`_sub(sd, prefix)` and remaps its own slice, and those modules are `of3t-tape`'s files and
need a card to construct. What is reachable without one, and what this covers, is the three
stack remaps that are callable as pure functions -- the trunk pairformer, the MSA module and
the template pair stack -- which is exactly where SS3a says the fusion lives. Their tensors
outside those prefixes are inventoried and classified, not mapped, and they are listed as
such.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

from tt_bio import openfold3_weights as of3w

CKPT = "/home/ttuser/.boltz/of3-p2-155k.pt"

# (label, callable, their prefix). Each takes the full state dict and a prefix.
STACKS = [
    ("trunk_pairformer", of3w.remap_pairformer_stack, "pairformer_stack"),
    ("msa_module", of3w.remap_msa_module, "msa_module"),
    ("template_pair_stack", of3w.remap_template_pair_stack,
     "template_embedder.template_pair_stack"),
    # The confidence head's own pairformer. Worth covering here specifically: the
    # confidence path is the one A2 found running untaped, and finetune_3.yml is the
    # confidence-only stage, so it is the last place a silent gap should be tolerated.
    ("confidence_pairformer", of3w.remap_pairformer_stack,
     "aux_heads.pairformer_embedding.pairformer_stack"),
]


def load_keys() -> dict:
    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def tracers(sd: dict, id_of: dict, override: dict | None = None) -> dict:
    """Their state dict with every tensor replaced by a constant-id float64 tracer."""
    override = override or {}
    out = {}
    for k, v in sd.items():
        val = override.get(k, id_of[k])
        out[k] = torch.full(tuple(v.shape), float(val), dtype=torch.float64)
    return out


def walk(obj, path=""):
    """Yield (dotted path, tensor) over the nested dict/list the remaps return."""
    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def decode(t: torch.Tensor, name_of: dict) -> list[dict]:
    """Decode one of our tensors into the dim-0 spans its source ids occupy.

    A pure rename gives one span covering the whole tensor. A dim-0 fusion gives one span
    per concatenated source. Anything else -- a value that is not an id, or an id appearing
    in a non-contiguous span -- is reported rather than smoothed over.
    """
    flat = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(-1, 1)
    spans, start, cur = [], 0, None
    for i in range(flat.shape[0]):
        row = flat[i]
        vals = torch.unique(row)
        v = float(vals[0]) if vals.numel() == 1 else None
        if v != cur:
            if cur is not None:
                spans.append((start, i, cur))
            start, cur = i, v
    spans.append((start, flat.shape[0], cur))
    out = []
    for lo, hi, v in spans:
        src = name_of.get(int(v)) if v is not None and float(v).is_integer() else None
        out.append({"rows": [lo, hi], "source": src,
                    "raw_value": None if v is None else float(v),
                    "resolved": src is not None})
    return out


def build(sd, id_of, name_of, override=None):
    tr = tracers(sd, id_of, override)
    ours, errors = {}, []
    for label, fn, prefix in STACKS:
        try:
            res = fn(tr, prefix) if prefix else fn(tr)
        except Exception as e:  # a remap that cannot run is a finding, not a crash
            errors.append({"stack": label, "error": f"{type(e).__name__}: {e}"})
            continue
        for path, t in walk(res, label):
            ours[path] = decode(t, name_of)
    return ours, errors


def main() -> int:
    sd = load_keys()
    keys = list(sd)
    id_of = {k: i + 1 for i, k in enumerate(keys)}
    name_of = {i + 1: k for i, k in enumerate(keys)}

    report = {"instrument": "PROTOCOL SS3a -- bijection manifest",
              "checkpoint": CKPT, "their_tensor_count": len(keys)}

    # --- their-side inventory ---------------------------------------------------------
    tops = collections.Counter(k.split(".")[0] for k in keys)
    report["their_top_level"] = dict(tops.most_common())

    # sample_diffusion and diffusion_module carry the same count; settle whether the
    # checkpoint stores one set of weights twice, because a duplicate is not a second
    # parameter and must not be counted as an unmapped one.
    dm = {k[len("diffusion_module."):]: v for k, v in sd.items()
          if k.startswith("diffusion_module.")}
    sdf = {k[len("sample_diffusion."):]: v for k, v in sd.items()
           if k.startswith("sample_diffusion.")}
    shared = sorted(set(dm) & set(sdf))
    same = (all(torch.equal(dm[k].to(torch.float32), sdf[k].to(torch.float32))
                for k in shared) if shared else None)
    # Equal counts invited the guess that one is a copy of the other. It is not: they share
    # no suffix at all, so the equal 763 is a coincidence of structure and both prefixes
    # carry their own parameters. Recorded because "equal counts" is exactly the kind of
    # coincidence a manifest is supposed to resolve rather than wave through.
    report["duplicate_prefixes"] = {
        "diffusion_module": len(dm), "sample_diffusion": len(sdf),
        "shared_suffixes": len(shared),
        "bitwise_identical": same,
        "diffusion_module_sample_keys": sorted(dm)[:4],
        "sample_diffusion_sample_keys": sorted(sdf)[:4],
        "note": ("no suffix is shared, so sample_diffusion is NOT a second copy of "
                 "diffusion_module; the matching 763 counts are a coincidence and both "
                 "prefixes carry distinct parameters that each need mapping")}

    # --- the manifest ------------------------------------------------------------------
    ours, errors = build(sd, id_of, name_of)
    report["remap_errors"] = errors

    manifest, fused, unresolved = {}, {}, []
    for path, spans in ours.items():
        for s in spans:
            if not s["resolved"]:
                unresolved.append({"our_tensor": path, **s})
        if len(spans) > 1:
            fused[path] = spans
        for s in spans:
            if s["source"]:
                manifest.setdefault(s["source"], []).append(
                    {"our_tensor": path, "rows": s["rows"]})

    # Dotted-boundary matching, not bare startswith: "msa_module" is a prefix of the
    # STRING "msa_module_embedder", which is a different top-level module. Matching on the
    # raw prefix put its two tensors in scope and then reported them as unmapped -- a
    # phantom finding manufactured by the filter rather than by the remap.
    covered_prefixes = tuple(p + "." for _, _, p in STACKS)
    in_scope = [k for k in keys if k.startswith(covered_prefixes)]
    mapped = [k for k in in_scope if k in manifest]
    unmapped_theirs = [k for k in in_scope if k not in manifest]
    multi = {k: v for k, v in manifest.items() if len(v) > 1}

    report["coverage"] = {
        "their_tensors_total": len(keys),
        "their_tensors_in_covered_stacks": len(in_scope),
        "their_tensors_mapped": len(mapped),
        "their_tensors_unmapped_in_scope": len(unmapped_theirs),
        "unmapped_in_scope_listed": sorted(unmapped_theirs)[:60],
        "our_tensors_produced": len(ours),
        "our_tensors_fused_from_multiple": len(fused),
        "our_unresolved_spans": len(unresolved),
        "their_tensors_mapping_to_more_than_one_of_ours": len(multi),
    }
    # --- what is NOT covered, named rather than left as a gap in the arithmetic --------
    # SS3a wants the unmapped set on both sides ENUMERATED and individually explained. The
    # tensors outside the four stacks are not unmapped by the remap; they are unreachable
    # by this instrument, because tt-bio has no whole-checkpoint remap and the modules that
    # consume them build on a device. They are classified by the tt-bio module that owns
    # them so the remaining work is a list rather than a number.
    out_of_scope = [k for k in keys if not k.startswith(covered_prefixes)]
    owner = {
        "diffusion_module": "openfold3_diffusion_module.py / _diffusion_transformer.py",
        "sample_diffusion": "openfold3_sample_diffusion.py",
        "aux_heads": "openfold3_confidence.py (the heads themselves, not its pairformer)",
        "input_embedder": "openfold3_host_prep.py / _atom_transformer.py",
        "template_embedder": "openfold3_template.py (outside the pair stack)",
        "msa_module_embedder": "openfold3_msa_embedder.py",
        "layer_norm_z": "openfold3_trunk.py", "layer_norm_s": "openfold3_trunk.py",
        "linear_z": "openfold3_trunk.py", "linear_s": "openfold3_trunk.py",
        "msa_module": "openfold3_msa_embedder.py (outside the block stack)",
    }
    oos = collections.Counter(k.split(".")[0] for k in out_of_scope)
    report["out_of_scope"] = {
        "count": len(out_of_scope),
        "by_top_level": {k: {"tensors": v,
                             "consumed_by": owner.get(k, "unclassified"),
                             "reason": "device module; needs a card to construct, so its "
                                       "remap is not callable as a pure function here"}
                         for k, v in oos.most_common()},
        "explanation": "these are not tensors the remap drops. They are tensors this "
                       "CPU-only instrument cannot reach, because tt-bio remaps per device "
                       "module rather than once per checkpoint. Closing them needs either a "
                       "card or a pure-function remap entry point per module.",
    }

    report["fused_tensors"] = {k: v for k, v in list(fused.items())[:40]}
    report["fused_tensor_count"] = len(fused)
    report["unresolved_spans"] = unresolved[:40]

    # --- negative control: the decoder must be reading VALUES, not key names -----------
    # Pick a tensor known to be fused, blank one of its two sources, and require that
    # exactly that span changes and no other.
    ctrl = {"ran": False}
    if fused:
        probe_path = sorted(fused)[0]
        srcs = [s["source"] for s in fused[probe_path] if s["source"]]
        if srcs:
            victim = srcs[0]
            ours2, _ = build(sd, id_of, name_of, override={victim: 0})
            before, after = ours[probe_path], ours2[probe_path]
            changed = [p for p in ours if ours[p] != ours2.get(p)]
            span_flipped = any(s["raw_value"] == 0.0 for s in after)
            expected = sorted(p for p in ours
                              if any(s["source"] == victim for s in ours[p]))
            ctrl = {"ran": True, "probe_tensor": probe_path, "blanked_source": victim,
                    "our_tensors_changed": len(changed),
                    "our_tensors_expected_to_change": len(expected),
                    "localised": sorted(changed) == expected,
                    "blanked_span_visible": span_flipped,
                    "before": before, "after": after,
                    "pass": sorted(changed) == expected and span_flipped}
    report["negative_control"] = ctrl

    ok = (not errors and not unresolved and not unmapped_theirs
          and not multi and ctrl.get("pass", False))
    report["verdict"] = "PASS" if ok else "FAIL"

    out = Path(__file__).resolve().parent / "bijection_manifest.json"
    full = dict(report)
    full["manifest"] = manifest
    out.write_text(json.dumps(full, indent=2))

    c = report["coverage"]
    print(f"their tensors: {c['their_tensors_total']} total, "
          f"{c['their_tensors_in_covered_stacks']} in the four covered stacks, "
          f"{report['out_of_scope']['count']} out of scope")
    print(f"  mapped   : {c['their_tensors_mapped']}")
    print(f"  UNMAPPED : {c['their_tensors_unmapped_in_scope']}")
    print(f"our tensors: {c['our_tensors_produced']} produced, "
          f"{c['our_tensors_fused_from_multiple']} fused from more than one of theirs, "
          f"{c['our_unresolved_spans']} unresolved spans")
    print(f"duplicate check: sample_diffusion == diffusion_module bitwise: "
          f"{report['duplicate_prefixes']['bitwise_identical']}")
    if errors:
        print(f"remap errors: {errors}")
    if ctrl.get("ran"):
        print(f"[{'PASS' if ctrl['pass'] else 'FAIL'}] negative control: blanking "
              f"{ctrl['blanked_source']} changed {ctrl['our_tensors_changed']} of our "
              f"tensors, expected {ctrl['our_tensors_expected_to_change']}")
    print(f"\nVERDICT: {report['verdict']}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
