#!/usr/bin/env python3
"""Stamp a captured diffusion boundary with the reference tree it was taken on.

`of3t-ditcot` scored three arms against `/home/ttuser/of3t_diffusion_cap` and published
0.7055 / 0.2584 / 0.1065 while its state doc discussed 0.4.3 and 0.5.0 in prose. Only the
`provenance.cap` field in the JSON says which capture answered, and nothing anywhere says which
upstream PACKAGE built that capture. That is D149's shape exactly: the version identity of a
number asserted rather than read back.

A capture cannot call `refpath.assert_resolved()` after the fact, so this stamps it with the two
things that CAN be read back today:

1. the tree resolved in THIS process, by `refpath.assert_resolved()`, importing `openfold3` and
   reading `openfold3.__file__` -- not a constant, not a PYTHONPATH, the resolution;
2. an ARCHITECTURE FINGERPRINT taken from the capture's own `grad_f64` key set and checked
   against the parameter names the resolved tree's `DiffusionModule` actually has.

(2) is what makes the stamp evidence rather than a label. The 0.4.3 and 0.5.0 diffusion modules
differ by 23 parameters: 0.4.3 puts a `layer_norm_z` inside every `AttentionPairBias`
(`attention_pair_bias.py:107`, applied at 156) and keeps the shared
`diffusion_transformer.layer_norm_z` only for the cross-attention atom transformers
(`diffusion_transformer.py:254`, `use_cross_attention`); 0.5.0 replaces the DiT's block with
`DiffusionAttentionPairBias`, which has no `layer_norm_z` member at all. So a capture's key set
names its own architecture, and a stamp that claims 0.4.3 over a capture with no per-block norms
fails here instead of surviving into a ratio.

Reads the capture's keys only. Nothing is rewritten and no gradient is recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import refpath  # noqa: E402

PREFIX = "diffusion_module."
PERBLOCK = ".attention_pair_bias.layer_norm_z."
SHARED = "diffusion_transformer.layer_norm_z."


def fingerprint(keys) -> dict:
    """What the key set says about which upstream built it."""
    per = sorted(k for k in keys if PERBLOCK in k and "diffusion_transformer.blocks." in k)
    shared = sorted(k for k in keys if k.endswith(SHARED + "weight") or
                    k.endswith(SHARED + "bias"))
    blocks = sorted({k.split("diffusion_transformer.blocks.")[1].split(".")[0] for k in per},
                    key=int)
    return {"n_parameters": len(keys),
            "n_perblock_layer_norm_z": len(per),
            "perblock_blocks": blocks,
            "n_shared_layer_norm_z": len(shared),
            "shared_layer_norm_z": shared,
            "architecture": ("0.4.3-family: per-block AttentionPairBias.layer_norm_z"
                             if per else
                             "0.5.0-family: DiffusionAttentionPairBias, no per-block "
                             "layer_norm_z"),
            "sample_perblock": per[:2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True, help="the captured boundary directory to stamp")
    ap.add_argument("--tree", default=refpath.OF3PKG,
                    help="the upstream tree the capture is CLAIMED to have been built on; the "
                         "stamp fails if the capture's own key set disagrees with it")
    ap.add_argument("--expect", default="", choices=["", "0.4.3", "0.5.0"],
                    help="refuse unless the fingerprint reads this family")
    ap.add_argument("--out", default="",
                    help="also write the stamp here, beside the row's other artifacts")
    a = ap.parse_args()
    t0 = time.time()

    refpath.require(a.tree, f"{a.cap}/sub_boundary.pt", f"{a.cap}/diffusion_boundary.pt")
    refpath.install(a.tree)
    tree = refpath.assert_resolved(a.tree)
    print(f"REF_TREE resolved: {tree}", flush=True)

    import torch
    from openfold3.core.model.structure.diffusion_module import DiffusionModule
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    # The capture's own key set, read back from the capture. mmap keeps this off the 3 GB path:
    # only the keys are wanted, never a storage.
    try:
        S = torch.load(f"{a.cap}/sub_boundary.pt", map_location="cpu", weights_only=False,
                       mmap=True)
        how = "mmap"
    except Exception as e:                                        # pragma: no cover - fallback
        print(f"[mmap unavailable: {type(e).__name__}: {e}; full load]", flush=True)
        S = torch.load(f"{a.cap}/sub_boundary.pt", map_location="cpu", weights_only=False)
        how = "full"
    cap_keys = sorted(S["grad_f64"])
    cap_fp = fingerprint(cap_keys)
    print(f"[{time.time()-t0:.0f}s] capture read ({how}): {cap_fp['n_parameters']} tensors, "
          f"{cap_fp['architecture']}", flush=True)
    del S

    # And what the resolved tree's own module says its parameters are. Same construction
    # `perf/of3t_trajwide/trajwide.py:270` uses: the config OBJECT, presets=["train"].
    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    m = DiffusionModule(config=cfg.architecture.diffusion_module)
    tree_keys = sorted(n for n, _ in m.named_parameters())
    tree_fp = fingerprint(tree_keys)
    print(f"[{time.time()-t0:.0f}s] tree module: {tree_fp['n_parameters']} parameters, "
          f"{tree_fp['architecture']}", flush=True)

    # The checkpoint against that module, which is the missing/unexpected count the brief asks
    # for. `strict=False` is how every builder in this campaign loads; the point is to REPORT
    # what it swallowed rather than to change it.
    ck = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    own = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    inc = m.load_state_dict(own, strict=False)
    missing, unexpected = sorted(inc.missing_keys), sorted(inc.unexpected_keys)
    print(f"[{time.time()-t0:.0f}s] checkpoint into the tree's module: "
          f"missing {len(missing)}, unexpected {len(unexpected)}", flush=True)

    same = cap_keys == tree_keys
    stamp = {
        "REF_TREE_resolved": tree,
        "cap": os.path.realpath(a.cap),
        "capture_fingerprint": cap_fp,
        "tree_fingerprint": tree_fp,
        "capture_keys_equal_tree_parameters": same,
        "checkpoint": {"file": os.path.expanduser("~/of3-weights/of3-p2-155k.pt"),
                       "n_in_scope": len(own),
                       "missing_keys": len(missing), "unexpected_keys": len(unexpected),
                       "missing": missing[:20], "unexpected": unexpected[:20]},
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamped_by": "perf/of3t_ditref/capstamp.py",
        "what": "the tree read back in-process by refpath.assert_resolved(), plus the "
                "architecture fingerprint of the capture's own grad_f64 key set checked "
                "against that tree's DiffusionModule parameter names",
    }

    bad = []
    if not same:
        only_cap = sorted(set(cap_keys) - set(tree_keys))[:10]
        only_tree = sorted(set(tree_keys) - set(cap_keys))[:10]
        stamp["mismatch"] = {"only_in_capture": only_cap, "only_in_tree": only_tree}
        bad.append(f"the capture's {len(cap_keys)} keys are not the tree's {len(tree_keys)} "
                   f"parameters; only_in_capture {only_cap}, only_in_tree {only_tree}")
    if a.expect == "0.4.3" and cap_fp["n_perblock_layer_norm_z"] == 0:
        bad.append("--expect 0.4.3 but the capture carries no per-block layer_norm_z")
    if a.expect == "0.5.0" and cap_fp["n_perblock_layer_norm_z"] != 0:
        bad.append(f"--expect 0.5.0 but the capture carries "
                   f"{cap_fp['n_perblock_layer_norm_z']} per-block layer_norm_z")
    stamp["verdict"] = "OK" if not bad else "REFUSED"
    stamp["refusals"] = bad

    # A refusal must not leave its own stamp inside the capture: the directory-level stamp is
    # what later arms READ, so writing a REFUSED one there replaces a good stamp with a bad one.
    # Caught in-pass, after the 0.4.3 negative control clobbered the 0.5.0 capture's stamp.
    targets = ([] if bad else [os.path.join(a.cap, "REFTREE.json")]) + ([a.out] if a.out else [])
    for p in targets:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(stamp, fh, indent=1, sort_keys=True)
        print(f"wrote {p}", flush=True)
    if bad:
        for b in bad:
            print(f"REFUSED: {b}", flush=True)
        return 1
    print(f"[{time.time()-t0:.0f}s] STAMP OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
