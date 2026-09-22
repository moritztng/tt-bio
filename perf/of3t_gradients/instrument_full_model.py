#!/usr/bin/env python3
"""PROTOCOL SS3a over the WHOLE OpenFold3 checkpoint, and SS3b against the frozen bundle.

Three things the campaign has been owed, in one pass over one built model on a card.

1. SS3a's out-of-scope set. K22 mapped 3,275 of their 4,935 tensors and left 1,660 unmapped for
   one reason only: the instrument is CPU-only and tt-bio remaps per device module, so
   `diffusion_module` (763), `sample_diffusion` (763), `input_embedder` (98) and `aux_heads` (16)
   could not be constructed at all. This builds the shipped `OpenFold3` model on the card and
   places their tensors onto the tensors that model HOLDS, by exact bf16 value with every
   candidate confirmed elementwise. A card is what was missing; nothing else changed.

2. Leaves against total for the whole model (K29). Never leaves alone. The walk of the built
   model is the denominator and `Module.torch_to_tt` is recorded beside it so the two numbers
   stay visibly different quantities.

3. SS3b against the FROZEN bundle. `of3t-reference`'s `grad_presence.json` is hash-verified from
   `manifest.json` before it is read, and their presence set is compared against ours AS A SET.
   A missing gradient and a zero gradient are different; nothing is zero-filled.

What this deliberately does NOT do: compare a gradient MAGNITUDE. That needs `grads_f64.pt` and
`w0.pt`, which the manifest declares at 2,947,812,293 and 4,573,705,142 bytes and which are not
on this host. The hash check is run first precisely so that absence is a reported fact rather
than a comparison against nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "perf/of3t_gradients"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
#: Where the large artifacts actually are. The pass-9 amendment names
#: `/home/moritz/of3t_bundle/bundle_min/` as canonical and VERIFIED; on this host that directory
#: does not exist and the files are at the pass-6 path below. Recorded rather than silently
#: followed, because a consumer that guesses the path and finds nothing looks the same as a
#: consumer that found an empty bundle.
BUNDLE = "/home/ttuser/of3t/bundle_min"
#: `of3t-reference`'s published half, read from ITS branch rather than from a copy, so this row
#: cannot drift from the reference it verifies against. `wk/of3t-reference` at `f898ca3c2`.
REF_BRANCH = "origin/wk/of3t-reference"
#: `manifest.json` was renamed `MANIFEST.json` so it could not be mistaken for the publication
#: on a case-sensitive filesystem, and its schema changed from a `files` dict to an `artifacts`
#: list. Both are followed here rather than worked around: a consumer pinned to the old name
#: fails closed, which is what happened on the first run of this pass.
MANIFEST_GIT = "perf/of3t_reference/bundle_min/MANIFEST.json"
#: The presence pattern OF THE VALIDATED GRADIENT, not of the withdrawn random-init one. Git
#: carries only the latter under its original name, so this is read from the host and hashed
#: against the manifest like every other artifact.
PRESENCE_FILE = "grad_presence_recycles0.json"


def from_ref_branch(path):
    """The file as `of3t-reference` published it, never a working-tree copy of it."""
    import subprocess
    out = subprocess.run(["git", "show", f"{REF_BRANCH}:{path}"],
                         capture_output=True, check=True)
    return json.loads(out.stdout)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_bundle(manifest_path):
    """Every declared artifact, hashed before anything reads it. Absence is a result."""
    man = from_ref_branch(manifest_path)
    out = {"manifest": f"{REF_BRANCH}:{manifest_path}", "bundle_dir": BUNDLE, "files": {}}
    decl = {x["file"]: x for x in man.get("artifacts", []) if x.get("sha256")}
    decl = {k: v for k, v in decl.items() if v["sha256"] != "recomputed-on-rename"}
    for name, meta in sorted(decl.items()):
        p = os.path.join(BUNDLE, name)
        if not os.path.isfile(p):
            out["files"][name] = {"present": False, "declared_bytes": meta.get("bytes"),
                                  "declared_sha256": meta.get("sha256")}
            continue
        got = sha256(p)
        out["files"][name] = {"present": True, "bytes": os.path.getsize(p),
                              "declared_sha256": meta.get("sha256"), "sha256": got,
                              "match": got == meta.get("sha256")}
    out["verified"] = sorted(k for k, v in out["files"].items() if v.get("match"))
    out["absent"] = sorted(k for k, v in out["files"].items() if not v["present"])
    out["mismatched"] = sorted(k for k, v in out["files"].items()
                               if v["present"] and not v.get("match"))
    return man, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialise", type=int, default=0, metavar="N",
                    help="drive the trunk pairformer once at N tokens, with the tape hook "
                         "installed and under no_grad, so the weights a module fuses on its "
                         "first call at a given chunk width exist before the walk. of3t-leaves "
                         "measured this as 188 -> 204 on a 4-block stack; 0 skips it.")
    ap.add_argument("--tag", default="of3_full")
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from bijection_device import device_bijection

    t0 = time.perf_counter()
    rep = {"instrument": "PROTOCOL SS3a whole-checkpoint bijection + SS3b presence",
           "checkpoint": CKPT, "host": "qb2", "card": 0, "board": "p300c"}

    # ---- the bundle, hashed before anything reads it -----------------------------------------
    man, ver = verify_bundle(MANIFEST_GIT)
    rep["bundle"] = ver
    rep["bundle_gradient_block"] = man.get("gradient")
    print(f"[{time.perf_counter()-t0:.0f}s] bundle: verified {ver['verified']}, "
          f"absent {ver['absent']}, mismatched {ver['mismatched']}", flush=True)

    # ---- their tensors ------------------------------------------------------------------------
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    atoms = {k: v.to(torch.float32) for k, v in sd.items() if torch.is_tensor(v)}
    rep["their_tensor_count"] = len(atoms)
    # `sample_diffusion.diffusion_module.X` against `diffusion_module.X`, checked rather than
    # assumed. K22 recorded the matching 763 counts as "a coincidence... both prefixes carry
    # distinct parameters", which is refuted below: it compared `X` against
    # `diffusion_module.X` because it stripped only the top-level prefix.
    dmk = {k[len("diffusion_module."):]: v for k, v in atoms.items()
           if k.startswith("diffusion_module.")}
    sdk = {k[len("sample_diffusion.diffusion_module."):]: v for k, v in atoms.items()
           if k.startswith("sample_diffusion.diffusion_module.")}
    both = sorted(set(dmk) & set(sdk))
    rep["sample_diffusion_is_a_copy"] = {
        "diffusion_module": len(dmk), "sample_diffusion_nested": len(sdk),
        "shared_suffixes": len(both),
        "bitwise_identical": sum(1 for k in both if torch.equal(dmk[k], sdk[k])),
        "corrects": "LEDGER K22, which recorded the two prefixes as carrying distinct parameters"}
    tops = sorted({k.split(".")[0] for k in atoms})
    rep["their_top_level"] = {t: sum(1 for k in atoms if k.startswith(t + ".") or k == t)
                              for t in tops}
    print(f"[{time.perf_counter()-t0:.0f}s] their tensors: {len(atoms)} over {len(tops)} "
          f"top-level sections", flush=True)

    # ---- our built model ---------------------------------------------------------------------
    loaded, orig = [], T.Module.torch_to_tt

    def recording(self, key, *ar, **kw):
        t = orig(self, key, *ar, **kw)
        loaded.append(key)
        return t

    T.Module.torch_to_tt = recording
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    from tt_bio.openfold3_fold import OpenFold3
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    model = OpenFold3(sd, ckc)
    # `OpenFold3.__init__` leaves `confidence_head = None` and builds it on first use
    # (`openfold3_fold.py:244-246`), so a census of the constructed model is blind to all 244
    # `aux_heads` tensors -- K29's shape again, at model scope and through a different door.
    # Built here with the same call the fold makes, so the denominator is the model that runs.
    model.confidence_head = OF3ConfidenceHead(model._confidence_sd, model.device, model.ckc)
    T.Module.torch_to_tt = orig
    before_fwd = len(device_weights(model))
    materialise = None
    if a.materialise:
        # Values do not matter for materialising a fused weight; the CHUNK WIDTH does, and it is
        # a function of the token count, so the count is recorded rather than left implicit. The
        # hook is installed because several fused kernels decline while taping and declining
        # fuses a DIFFERENT weight -- of3t-leaves' third ordering trap.
        from tt_bio import autograd as ag
        n = a.materialise
        ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev,
                                       dtype=ttnn.bfloat16)
        pf = model.trunk.pairformer
        c_s = sd["pairformer_stack.blocks.0.attn_pair_bias.layer_norm_a.weight"].shape[0]
        c_z = sd["pairformer_stack.blocks.0.pair_stack.tri_mul_in.layer_norm_in.weight"].shape[0]
        g = torch.Generator().manual_seed(7)
        installed = ag.install() if ag.__dict__.get("grad_hook", lambda: None)() is None else None
        try:
            with ag.no_grad():
                pf(ft(torch.randn(1, n, c_s, generator=g) * 0.05),
                   ft(torch.randn(1, n, n, c_z, generator=g) * 0.05))
        finally:
            try:
                ag.uninstall()
            except Exception:
                pass
        # And the confidence head, whose weights are materialised on its first CALL rather than
        # at construction. Building it is not enough: `aux_heads.distogram.linear.weight` alone
        # is 3.57 % of the reference gradient's squared norm (`reach_by_norm.json`), which made
        # this the single largest lever on SS3a's reach after the diffusion module. The values
        # are arbitrary; the shapes are read off their own checkpoint so a dimension cannot be
        # guessed wrong, and the call is the one `fold` makes.
        conf_err = None
        try:
            pe = "aux_heads.pairformer_embedding."
            c_in = sd[pe + "linear_i.weight"].shape[1]
            c_tr = sd["aux_heads.plddt.layer_norm.weight"].shape[0]
            c_z2 = sd[pe + "linear_i.weight"].shape[0]
            n_bins = sd[pe + "linear_distance.weight"].shape[1]
            with ag.no_grad():
                model.confidence_head.forward_device(
                    ft(torch.randn(1, n, c_in, generator=g) * 0.05),
                    ft(torch.randn(1, n, c_tr, generator=g) * 0.05),
                    ft(torch.randn(1, n, n, c_z2, generator=g) * 0.05),
                    ft(torch.randn(1, n, n, n_bins, generator=g) * 0.05))
        except Exception as e:                       # a gap is a finding, not a crash
            conf_err = f"{type(e).__name__}: {e}"
            print(f"  confidence-head materialise failed: {conf_err}", flush=True)
        materialise = {"tokens": n, "module": "trunk.pairformer + confidence_head",
                       "confidence_head_error": conf_err,
                       "reachable_before": before_fwd,
                       "reachable_after": len(device_weights(model))}
        print(f"[{time.perf_counter()-t0:.0f}s] materialise at {n} tokens: "
              f"{materialise['reachable_before']} -> {materialise['reachable_after']} "
              f"device tensors", flush=True)
    walked = device_weights(model)
    rep["materialise"] = materialise
    rep["ours"] = {"from_loader_hook": len(loaded), "reachable_by_walk": len(walked),
                   "reachable_before_any_forward": before_fwd,
                   "note": "construction only. Weights a module materialises lazily on its first "
                           "call at a given chunk width are absent from BOTH numbers, which is "
                           "of3t-leaves' 188->204 ordering finding and is why this is reported "
                           "as a construction-time denominator rather than a training one."}
    print(f"[{time.perf_counter()-t0:.0f}s] ours: loader hook {len(loaded)}, "
          f"walk of the built model {len(walked)}", flush=True)

    dev_t = {}
    for p, t in walked.items():
        try:
            dev_t[p] = ttnn.to_torch(t).to(torch.float32)
        except Exception as e:                      # a handle we cannot read is a finding
            rep.setdefault("unreadable", []).append({"path": p, "error": f"{type(e).__name__}"})
    print(f"[{time.perf_counter()-t0:.0f}s] read back {len(dev_t)} device tensors", flush=True)

    # ---- the bijection, whole checkpoint -------------------------------------------------------
    bij = device_bijection(dev_t, atoms)
    placed = set(bij["placements"])
    unplaced = sorted(set(atoms) - placed)
    by_top = {}
    for k in unplaced:
        by_top[k.split(".")[0]] = by_top.get(k.split(".")[0], 0) + 1
    rep["bijection"] = {
        "their_placed": len(placed), "their_total": len(atoms),
        "their_unplaced": len(unplaced), "unplaced_by_top_level": by_top,
        "unplaced_sample": unplaced[:60],
        "device_unmatched": len(bij["device_unmatched"]),
        "device_unmatched_sample": bij["device_unmatched"][:40],
        "kinds": {k: sum(1 for m in bij["per_device"].values() if m["kind"] == k)
                  for k in sorted({m["kind"] for m in bij["per_device"].values()})},
        "fused_device_tensors": sum(1 for m in bij["per_device"].values()
                                    if len(m["parts"]) > 1),
        "device_tensors_with_an_unexplained_band": sorted(
            ({"device_path": k, "unexplained": m.get("unexplained_band", 0),
              "shape": None} for k, m in bij["per_device"].items()
             if m.get("unexplained_band")), key=lambda d: -d["unexplained"])[:20],
        "n_device_tensors_with_an_unexplained_band": sum(
            1 for m in bij["per_device"].values() if m.get("unexplained_band")),
        "their_in_a_fusion": sum(1 for v in bij["placements"].values()
                                 if any(p["length"] < max(p["device_shape"]) for p in v))}
    with open("perf/of3t_equivalence/bijection_manifest.json") as fh:
        k22 = json.load(fh)
    rep["bijection"]["k22_in_scope"] = k22["coverage"]["their_tensors_in_covered_stacks"]
    rep["bijection"]["k22_out_of_scope"] = k22["out_of_scope"]["count"]
    rep["bijection"]["k22_out_of_scope_now_placed"] = len(
        [k for k in placed if k not in k22["manifest"]])
    print(f"[{time.perf_counter()-t0:.0f}s] bijection: {len(placed)}/{len(atoms)} of their "
          f"tensors placed, {len(bij['device_unmatched'])} device tensors unmatched; "
          f"{rep['bijection']['k22_out_of_scope_now_placed']} of them were outside K22's scope",
          flush=True)

    # ---- SS3b presence, against the hash-verified frozen set ------------------------------------
    pres = None
    if PRESENCE_FILE in ver["verified"]:
        pres = json.load(open(os.path.join(BUNDLE, PRESENCE_FILE)))
    rep["presence"] = {"reference": "of3t-reference BUNDLE-MIN " + PRESENCE_FILE,
                       "reference_hash_verified": PRESENCE_FILE in ver["verified"]}
    if pres is not None:
        table = pres if isinstance(pres, dict) else {}
        for key in ("presence", "grad_present", "parameters"):
            if isinstance(table.get(key), dict):
                table = table[key]
                break
        theirs_with = {k for k, v in table.items() if v is True or v == "present"}
        theirs_without = {k for k, v in table.items() if k not in theirs_with}
        ours_reachable = placed
        rep["presence"].update(
            their_total=len(table), their_with_gradient=len(theirs_with),
            their_without_gradient=sorted(theirs_without)[:40],
            ours_carried_by_a_device_tensor=len(ours_reachable & set(table)),
            # The FULL set, not a sample. It was truncated to 60 here, and a consumer that
            # read the list rather than the count computed a reach over 60 of 610 missing
            # tensors -- which is exactly the shape of the defect D17 names, arriving through
            # the artifact instead of through the metric. A field whose count and whose length
            # disagree is a trap; the sample now lives under its own name.
            their_with_gradient_we_cannot_carry=sorted(theirs_with - ours_reachable),
            their_with_gradient_we_cannot_carry_sample=sorted(theirs_with - ours_reachable)[:60],
            n_their_with_gradient_we_cannot_carry=len(theirs_with - ours_reachable),
            their_placed_with_gradient=sorted(ours_reachable & theirs_with),
            rule="SS3b: compared as a set, before any magnitude. Nothing was zero-filled.")
        print(f"[{time.perf_counter()-t0:.0f}s] presence: theirs {len(theirs_with)}/{len(table)} "
              f"with a gradient; we carry {len(ours_reachable & set(table))} of them on a device "
              f"tensor, {len(theirs_with - ours_reachable)} we do not", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"full_model_{a.tag}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
