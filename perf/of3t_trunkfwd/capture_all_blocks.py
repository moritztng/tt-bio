#!/usr/bin/env python3
"""Capture EVERY pairformer block's input in upstream's own float64 step, cropped to 64 tokens.

`of3t-gradients` captured blocks 0, 23 and 47 because that is what a per-block gradient arm
needs. Arm 3 of this row needs all 48: the question is whether our per-block forward agrees when
fed block k's CAPTURED input, against the 48-block composition that feeds block k our own block
k-1 output. A per-block arm that only has three inputs cannot answer it.

Two things make this the SAME run as `of3t_gradients/cap` rather than a plausible neighbour:
the seed, the dropout pinning and the draw replay are identical, and the capture is ASSERTED
bit-identical against `cap/block{0,23,47}_boundary.pt` at the end. If that assertion fails the
composition arm is driven by a different function and nothing downstream means anything.

No backward. Arm 3 is a forward question and `loss.backward()` is 1040 s of the 1400 s.

Cropped to the first 64 token positions at capture time, which is what the device arms compare
at. The batch has 56 real tokens inside that window (single_mask.sum() == 56), and the record
already shows the crop is sound: the 48-block masked forward reads 2.7969e-01 at crop 64 and
2.7919e-01 at N=384.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_gradients"))
from capture_trunk_boundary import (BUNDLE, CKPT, ReplayDraws, manifest_from_git,  # noqa: E402
                                    sha256_file, verify)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("/home/ttuser/of3t_trunkfwd/capall"))
    ap.add_argument("--cap", type=Path, default=Path("/home/ttuser/of3t_gradients/cap"),
                    help="the existing 3-block capture this one is asserted against")
    ap.add_argument("--report", type=Path,
                    default=HERE / "CAPTURE_ALL_BLOCKS.json")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    ap.add_argument("--grads", default="grads_f64_recycles0.pt")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
    import bundle_min as BM

    man = manifest_from_git()
    man_src = "origin/wk/of3t-reference:perf/of3t_reference/bundle_min/MANIFEST.json"
    # Only the inputs a FORWARD-ONLY capture actually consumes are hashed. The gradient file
    # and its presence pattern are inputs to a gradient comparison, which this is not, and the
    # manifest has since renamed them: `artifacts` declares `grads_f64_r0.pt` while
    # `validated_gradient.file` still reads `grads_f64_recycles0.pt`, so hashing a file this
    # capture never opens fails closed on somebody else's rename. Reported, not worked around:
    # see NAME_DRIFT in the report.
    files = ["batch_step003.pt", "draws_recycles0.pt"]
    rep = {"instrument": "every pairformer block input of BUNDLE-MIN own step, crop 64",
           "manifest": man_src, "bundle_dir": str(a.bundle), "crop": a.crop}
    rep["hashes"] = verify(man, files, a.bundle, man_src)
    vg = man["validated_gradient"]
    rep["reference"] = {"file": vg["file"], "num_recycles": vg["num_recycles"],
                        "loss": vg["loss"], "global_norm": vg["gradient_global_norm"]}
    _decl = {x["file"] for x in man["artifacts"]}
    rep["NAME_DRIFT"] = {
        "validated_gradient.file": vg["file"],
        "declared_in_artifacts": vg["file"] in _decl,
        "artifacts_gradient_files": sorted(f for f in _decl if f.startswith("grads_f64")),
        "why_it_matters": "A24 identifies an input by digest. The manifest names a gradient file "
                          "its own artifacts list no longer declares, so any row that hashes "
                          "validated_gradient.file against the artifacts list fails closed. This "
                          "capture consumes neither, and says so rather than silencing it."}
    print(f"[{time.time()-t0:.0f}s] bundle verified", flush=True)

    dtype = torch.float64
    raw = torch.load(a.bundle / "batch_step003.pt", weights_only=False)
    draws = torch.load(a.bundle / "draws_recycles0.pt", map_location="cpu", weights_only=False)

    built = BM.build(dtype, a.seed, "cpu", num_recycles=0)
    model, loss_fn = built[1], built[2]
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    rep["checkpoint"] = {"file": CKPT.name, "sha256": sha256_file(CKPT), "n_loaded": len(sd),
                         "n_missing": len(inc.missing_keys),
                         "n_unexpected": len(inc.unexpected_keys)}
    print(f"[{time.time()-t0:.0f}s] model built ({len(inc.missing_keys)} missing, "
          f"{len(inc.unexpected_keys)} unexpected)", flush=True)

    from openfold3.core.model.primitives.dropout import Dropout
    n_drop = 0
    for m in model.modules():
        if isinstance(m, Dropout):
            m.eval()
            n_drop += 1
    rep["dropout"] = {"disabled": True, "modules": n_drop,
                      "why": "identical to the cap/ capture: r = 0 is reproducible and is the "
                             "function our tape computes"}

    blocks = model.pairformer_stack.blocks
    nb = len(blocks)
    rep["n_blocks"] = nb
    c = a.crop
    cap: dict[int, dict] = {}

    def crop_s(x):
        return x[:, :c].detach().clone()

    def crop_z(x):
        return x[:, :c, :c].detach().clone()

    def pre_hook(i):
        def h(mod, args, kwargs):
            d = cap.setdefault(i, {})
            d["s_in"], d["z_in"] = crop_s(args[0]), crop_z(args[1])
            kw = kwargs or {}
            if "single_mask" in kw and torch.is_tensor(kw["single_mask"]):
                d["single_mask"] = crop_s(kw["single_mask"])
            if "pair_mask" in kw and torch.is_tensor(kw["pair_mask"]):
                d["pair_mask"] = crop_z(kw["pair_mask"])
        return h

    def post_hook(i):
        def h(mod, args, kwargs, out):
            d = cap.setdefault(i, {})
            d["s_out"], d["z_out"] = crop_s(out[0]), crop_z(out[1])
        return h

    handles = []
    for i in range(nb):
        handles.append(blocks[i].register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(blocks[i].register_forward_hook(post_hook(i), with_kwargs=True))

    BM.set_rng_state(draws["rng_state_before_step"], model)
    replay = ReplayDraws(draws["torch_randn"], draws["python_random"])
    batch = BM.move(raw, "cpu", dtype)
    t1 = time.time()
    with BM.no_autocast(), replay, torch.no_grad():
        private = copy.deepcopy(batch)
        b, out = model(private)
        loss, breakdown = loss_fn(b, out, _return_breakdown=True)
    for h in handles:
        h.remove()
    rep["replay"] = {"randn_consumed": replay.i_randn, "random_consumed": replay.i_rnd,
                     "n_mismatch": len(replay.mismatch), "mismatches": replay.mismatch[:8]}
    rep["forward"] = {"loss": float(loss), "their_loss": vg["loss"],
                      "rel": abs(float(loss) - vg["loss"]) / abs(vg["loss"]),
                      "seconds": time.time() - t1,
                      "note": "no_grad and no backward: arm 3 is a forward question"}
    print(f"[{time.time()-t0:.0f}s] forward {time.time()-t1:.0f}s loss {float(loss):.15f} "
          f"(their {vg['loss']:.15f}), mismatches {len(replay.mismatch)}", flush=True)

    # ---- the assertion that this is the SAME run as cap/ -------------------------------------
    same = {}
    for i in (0, 23, 47):
        p = a.cap / f"block{i}_boundary.pt"
        if not p.is_file():
            same[str(i)] = {"checked": False, "why": f"{p} absent"}
            continue
        old = torch.load(p, map_location="cpu", weights_only=False)
        o_s, o_z = old["args"][0][:, :c], old["args"][1][:, :c, :c]
        o_so, o_zo = old["out"][0][:, :c], old["out"][1][:, :c, :c]
        same[str(i)] = {
            "checked": True,
            "s_in_bit_identical": bool(torch.equal(o_s.to(torch.float64), cap[i]["s_in"])),
            "z_in_bit_identical": bool(torch.equal(o_z.to(torch.float64), cap[i]["z_in"])),
            "s_out_bit_identical": bool(torch.equal(o_so.to(torch.float64), cap[i]["s_out"])),
            "z_out_bit_identical": bool(torch.equal(o_zo.to(torch.float64), cap[i]["z_out"])),
        }
        del old
        print(f"  block {i} vs cap/: {same[str(i)]}", flush=True)
    rep["bit_identical_to_cap"] = same
    bad = [k for k, v in same.items() if v.get("checked") and not all(
        v[x] for x in ("s_in_bit_identical", "z_in_bit_identical",
                       "s_out_bit_identical", "z_out_bit_identical"))]
    rep["same_run_as_cap"] = not bad
    if bad:
        raise SystemExit(f"blocks {bad} do not reproduce cap/ bit for bit -- this capture is a "
                         f"different function and arm 3 cannot be driven by it")

    # ---- chain consistency: block k output IS block k+1 input --------------------------------
    chain = []
    for i in range(nb - 1):
        chain.append({"k": i,
                      "s": bool(torch.equal(cap[i]["s_out"], cap[i + 1]["s_in"])),
                      "z": bool(torch.equal(cap[i]["z_out"], cap[i + 1]["z_in"]))})
    rep["chain_is_sequential"] = all(x["s"] and x["z"] for x in chain)
    rep["chain_breaks"] = [x for x in chain if not (x["s"] and x["z"])]

    masks = {k: cap[0][k] for k in ("single_mask", "pair_mask") if k in cap[0]}
    rep["masks"] = {k: {"shape": list(v.shape), "sum": float(v.sum())} for k, v in masks.items()}
    blob = {"crop": c, "n_blocks": nb, "masks": masks,
            "s_in": [cap[i]["s_in"] for i in range(nb)],
            "z_in": [cap[i]["z_in"] for i in range(nb)],
            "s_out": [cap[i]["s_out"] for i in range(nb)],
            "z_out": [cap[i]["z_out"] for i in range(nb)]}
    p = a.out / f"blocks_all_crop{c}.pt"
    torch.save(blob, p)
    rep["saved"] = {"file": str(p), "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "s_shape": list(blob["s_in"][0].shape),
                    "z_shape": list(blob["z_in"][0].shape)}
    rep["breakdown"] = {k: float(v) for k, v in breakdown.items()
                        if torch.is_tensor(v) and v.numel() == 1}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {p} ({p.stat().st_size/1e6:.0f} MB) and {a.report}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
