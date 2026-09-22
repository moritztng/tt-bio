"""Stamp an instrument's own artifact with the revision its reference was built from.

D23/R126: for forty passes the campaign's numbers were taken against upstream 0.5.0 running a
checkpoint 0.5.0's own registry refuses to load, and no artifact recorded which revision it was.
The revision is an input to every one of these measurements, so it belongs in the JSON beside
them and not in a log that gets rotated.

The load result comes with it, and a non-empty unexpected set raises: a reference that silently
drops the checkpoint's tensors makes the measurement a statement about the reference.
"""
from __future__ import annotations

import re
from pathlib import Path


def reference_revision(openfold3_module, model, state_dict, enforce: bool = True) -> dict:
    tree = Path(openfold3_module.__file__).resolve().parent.parent
    version = "unknown"
    for meta in sorted(tree.glob("*.dist-info/METADATA")) + [tree / "PKG-INFO"]:
        if meta.is_file():
            m = re.search(r"^Version:\s*(\S+)", meta.read_text(errors="replace"), re.M)
            if m:
                version = m.group(1)
                break
    bb = (tree / "openfold3/core/model/latent/base_blocks.py").read_text()
    apb = (tree / "openfold3/core/model/layers/attention_pair_bias.py").read_text()
    inc = model.load_state_dict(state_dict, strict=False)
    rev = {
        "tree": str(tree), "version": version,
        "transpose_bias_true_in_base_blocks": bb.count("transpose_bias=True"),
        "has_DiffusionAttentionPairBias": "class DiffusionAttentionPairBias" in apb,
        "load": {"missing": len(inc.missing_keys), "unexpected": len(inc.unexpected_keys),
                 "missing_keys": sorted(inc.missing_keys)[:8]},
    }
    if inc.unexpected_keys and enforce:
        raise SystemExit(
            f"KEY GATE FAILED: {len(inc.unexpected_keys)} checkpoint tensors have nowhere to go "
            f"in this reference and are dropped silently (D23/R126). First four: "
            f"{sorted(inc.unexpected_keys)[:4]}")
    return rev
