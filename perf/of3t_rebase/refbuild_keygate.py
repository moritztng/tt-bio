#!/usr/bin/env python3
"""The check that `strict=False` swallowed for forty passes, in the two parts it needs.

D23/R126: the campaign's reference bundle was upstream openfold3 0.5.0 loading `of3-p2-155k.pt`,
a checkpoint 0.5.0's own `entry_points/parameters.py` marks `version_compatibility=
">=0.4,<0.4.4dev0"` and lists in `LEGACY_CHECKPOINTS`. 0.5.0's `DiffusionAttentionPairBias` has
no `layer_norm_z`, so the 48 `layer_norm_z.weight` tensors the checkpoint carries for the
diffusion transformer landed in `unexpected_keys` and were dropped in silence. The published
manifest even recorded `n_unexpected: 48`. Recording a number is not gating on it.

**Gate 1, keys.** Upstream already ships this rule: `InferenceRunner.
_load_state_dict_with_version_validation` (0.5.0 `entry_points/experiment_runner.py:750`) takes
the two key sets and loads `strict=False` only when `missing == {"model.version_tensor"}` with
nothing unexpected, raising `ValueError` otherwise. At missing=3 / unexpected=48 the p2-on-0.5.0
combination takes the raise branch, so upstream's own supported entry point refuses to load it.
The bundle went around that by calling `load_state_dict(strict=False)` directly. This applies
upstream's predicate rather than inventing one, and on a tree that ships the method it calls the
real thing (`--call-upstream`) so the rule is not merely transcribed.

**Gate 2, revision window.** The key gate is necessary and not sufficient, and it is blind in a
specific direction: `transpose_bias` carries no parameter, so a model computing the wrong
ending-node function loads at missing=0 / unexpected=0. So the installed revision is also
checked against the checkpoint's own declared `version_compatibility`, read out of
`entry_points/parameters.py` in the tree being built rather than hardcoded here. Of the two,
this one catches both halves of D23.

Neither gate is about OpenFold3. Any model whose reference is built by loading a checkpoint into
somebody else's module tree has both holes.

    --allow-unexpected  record a failing key set instead of exiting on it; this is how the
                        0.5.0 negative control gets written rather than crashed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_version(tree: Path) -> str:
    """The revision of the tree on sys.path, not of whatever is pip-installed.

    `importlib.metadata.version("openfold3")` answers for the installed distribution, which is
    exactly the confusion this row exists to undo, so it is not consulted.
    """
    for meta in sorted(tree.glob("*.dist-info/METADATA")) + [tree / "PKG-INFO",
                                                             tree.parent / "PKG-INFO"]:
        if meta.is_file():
            m = re.search(r"^Version:\s*(\S+)", meta.read_text(errors="replace"), re.M)
            if m:
                return m.group(1)
    raise SystemExit(f"no PKG-INFO or *.dist-info/METADATA under {tree}: the tree carries no "
                     f"version, and a reference build must know its own revision")


def tree_properties(tree: Path) -> dict:
    """The two code changes D23 separates the revisions by, read in the tree actually used."""
    bb = (tree / "openfold3/core/model/latent/base_blocks.py").read_text()
    apb = (tree / "openfold3/core/model/layers/attention_pair_bias.py").read_text()
    p = {
        "transpose_bias_true_in_base_blocks": len(re.findall(r"transpose_bias\s*=\s*True", bb)),
        "has_DiffusionAttentionPairBias_class": bool(
            re.search(r"^class\s+DiffusionAttentionPairBias\b", apb, re.M)),
        "layer_norm_z_mentions_in_attention_pair_bias": len(re.findall(r"layer_norm_z", apb)),
    }
    p["behaves_as"] = ("0.4.3 on both D23 changes"
                       if p["transpose_bias_true_in_base_blocks"] == 0
                       and not p["has_DiffusionAttentionPairBias_class"]
                       else "0.5.0 on at least one D23 change")
    return p


def version_window(ckpt_name: str, version: str) -> dict:
    """Gate 2: does this tree's revision sit inside what the checkpoint declares?"""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from openfold3.entry_points import parameters as P

    reg = {k: v for k, v in vars(P).items() if isinstance(v, dict) and v
           and all(hasattr(e, "file_name") for e in v.values())}
    entries = {}
    for d in reg.values():
        entries.update(d)
    name = next((k for k, e in entries.items() if e.file_name == ckpt_name), None)
    if name is None:
        return {"checked": False, "why": f"{ckpt_name} is not in this tree's registry",
                "registry_names": sorted(entries)}
    spec = getattr(entries[name], "version_compatibility", None)
    legacy = name in getattr(P, "LEGACY_CHECKPOINTS", [])
    inside = None if spec is None else Version(version) in SpecifierSet(spec, prereleases=True)
    return {
        "checked": True,
        "registry_name": name,
        "declared_version_compatibility": spec,
        "installed_version": version,
        "inside_window": inside,
        "is_default_checkpoint": name == getattr(P, "DEFAULT_CHECKPOINT_NAME", None),
        "listed_legacy": legacy,
        "source": "openfold3/entry_points/parameters.py in the tree being built",
    }


def by_leaf(keys):
    c = collections.Counter(k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] if "." in k else k
                            for k in keys)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, type=Path,
                    help="directory containing the openfold3/ package to build the reference from")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="key names do not depend on dtype; float32 keeps the gate cheap")
    ap.add_argument("--call-upstream", action="store_true",
                    help="also invoke upstream's own _load_state_dict_with_version_validation "
                         "where the tree ships it, so the rule is executed and not transcribed")
    ap.add_argument("--allow-unexpected", action="store_true")
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    import torch
    import openfold3
    if not openfold3.__file__.startswith(str(tree)):
        raise SystemExit(f"openfold3 resolved to {openfold3.__file__}, not inside {tree}")

    version = tree_version(tree)
    props = tree_properties(tree)
    window = version_window(a.checkpoint.name, version)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_reference"))
    import bundle_min as BM

    dtype = getattr(torch, a.dtype)
    _, model, _, _ = BM.build(dtype, a.seed, "cpu", num_recycles=a.num_recycles)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

    # Upstream's predicate, on the same two sets it computes (experiment_runner.py:753-756).
    model_keys, ckpt_keys = set(model.state_dict()), set(sd)
    missing, unexpected = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
    upstream_branch = ("warn_and_load_nonstrict"
                       if missing == ["version_tensor"] and not unexpected
                       else "load_strict" if not missing and not unexpected
                       else "raise ValueError")

    upstream_called = None
    if a.call_upstream:
        from openfold3.entry_points import experiment_runner as ER
        # Both revisions ship the rule, under different names and by different mechanisms.
        # 0.5.0 computes the two key sets itself (experiment_runner.py:750); 0.4.3 gets the
        # same answer out of a strict=True load and only forgives the exact
        # `Missing key(s) in state_dict: "model.version_tensor".` message (line 683).
        names = ["_load_state_dict_with_version_validation",
                 "_warn_on_missing_version_tensor_in_load_statedict"]
        name = next((n for n in names
                     if getattr(ER.InferenceExperimentRunner, n, None) is not None), None)
        if name is None:
            upstream_called = {"available": False,
                               "why": f"this tree ({version}) ships no such method"}
        else:
            import torch.nn as nn

            class _LM(nn.Module):             # upstream's key prefix is `model.`
                def __init__(s):
                    super().__init__()
                    s.model = model

            class _Shim:                      # the method only reads self.lightning_module
                lightning_module = _LM()

            fn = getattr(ER.InferenceExperimentRunner, name)
            try:
                fn(_Shim(), {f"model.{k}": v for k, v in sd.items()})
                upstream_called = {"available": True, "method": name, "raised": None}
            except Exception as e:            # noqa: BLE001 - the raise IS the result
                upstream_called = {"available": True, "method": name,
                                   "raised": type(e).__name__, "message": str(e)[:300]}

    inc = model.load_state_dict(
        {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in sd.items()}, strict=False)
    assert sorted(inc.missing_keys) == missing and sorted(inc.unexpected_keys) == unexpected

    rep = {
        "gate": "reference build gate (D23/R126): key sets AND revision window",
        "tag": a.tag, "tree": str(tree), "tree_version": version,
        "tree_properties": props,
        "checkpoint": {"file": a.checkpoint.name, "sha256": sha256_file(a.checkpoint),
                       "n_tensors": len(sd)},
        "build": {"dtype": a.dtype, "seed": a.seed, "num_recycles": a.num_recycles,
                  "n_model_params": sum(1 for _ in model.parameters()),
                  "n_model_state_dict": len(model_keys)},
        "gate1_keys": {
            "missing": {"n": len(missing), "by_leaf": by_leaf(missing), "all": missing},
            "unexpected": {"n": len(unexpected), "by_leaf": by_leaf(unexpected),
                           "all": unexpected},
            "upstream_rule": "InferenceRunner._load_state_dict_with_version_validation, "
                             "0.5.0 entry_points/experiment_runner.py:750",
            "upstream_branch_taken": upstream_branch,
            "upstream_called": upstream_called,
            "pass": not unexpected},
        "gate2_version_window": window,
        "enforced": not a.allow_unexpected,
    }
    gate1 = not unexpected
    gate2 = window.get("inside_window") is not False
    rep["verdict"] = ("PASS" if gate1 and gate2 else
                      "RECORDED (gate not enforced)" if a.allow_unexpected else "FAIL")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")

    print(f"[{a.tag}] tree {tree}  version {version}  "
          f"({props['behaves_as']}; transpose_bias=True x"
          f"{props['transpose_bias_true_in_base_blocks']}, "
          f"DiffusionAttentionPairBias {props['has_DiffusionAttentionPairBias_class']})")
    print(f"[{a.tag}] model {rep['build']['n_model_params']} params / "
          f"{len(model_keys)} state_dict entries, checkpoint {len(sd)} tensors")
    print(f"[{a.tag}] GATE 1 keys: missing={len(missing)} unexpected={len(unexpected)}  "
          f"-> upstream takes branch: {upstream_branch}")
    for lab, keys in (("missing", missing), ("unexpected", unexpected)):
        if keys:
            print(f"[{a.tag}]   {lab} by leaf: {by_leaf(keys)}")
    if upstream_called:
        print(f"[{a.tag}]   upstream's own method: {upstream_called}")
    print(f"[{a.tag}] GATE 2 window: {window.get('registry_name')} declares "
          f"{window.get('declared_version_compatibility')!r}, tree is {version} -> "
          f"inside={window.get('inside_window')}  legacy={window.get('listed_legacy')}")
    print(f"[{a.tag}] {rep['verdict']}  ->  {a.out}")

    if not (gate1 and gate2) and not a.allow_unexpected:
        print(f"[{a.tag}] GATE FAILED: "
              + ("" if gate1 else f"{len(unexpected)} checkpoint tensors have nowhere to go in "
                                  f"this reference and are dropped in silence. ")
              + ("" if gate2 else f"this tree ({version}) is outside the window "
                                  f"{window.get('declared_version_compatibility')!r} the "
                                  f"checkpoint declares. ")
              + "A reference built here is not a reference.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
