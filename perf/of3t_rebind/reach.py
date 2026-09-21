#!/usr/bin/env python3
"""How far the optimizer/tape defect reaches, and whether the repair moves an inference byte.

Two questions in one process, because both need the same built model and the same card.

REACH. `of3t-modeltraj` measured the defect at `diffusion_module.diffusion_conditioning`, 26
tensors and 36.9462 % of the model squared gradient norm, and argued from the mechanism that it
is not scope-local. An argument is not a measurement (A15). So this walks the whole built model,
hands the walked set to the SHIPPED optimizer, steps it once, and counts how many walked slots
`autograd.parameter_for` can still resolve -- the same instrument `of3t-modeltraj` read as 0 of
26. The count is then converted to a share of the squared gradient norm against upstream 0.4.3's
own float64 reference, `grads_f64_043.pt`, 4,170 tensors and 10.279642678524981 total.

INFERENCE. The tape is open only while training, so a training repair must not move a fold. The
same process folds one structure before touching anything and digests every file the fold wrote;
the caller compares that digest against the same fold on `origin/wk/of3t`.

The fold comes FIRST, deliberately: `device_weights` has to be taken after a forward, because a
module that fuses its weights in its own forward has none to walk before one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

REF_GRADS = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
MODEL_SQ_NORM = 10.279642678524981


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).glob("**/*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-step", action="store_true",
                    help="fold and walk only; for the arm that has no optimizer to run")
    a = ap.parse_args()

    import numpy as np
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)
    import tt_baseline as B

    res = {"model": a.model, "size": a.size, "head": os.environ.get("REBIND_HEAD", "")}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    assert tgt.is_file() and a3m.is_file(), f"no fixture at {tgt}"

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_rb_{a.model}_{a.size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res["card"] = {k: meta.get(k) for k in ("hardware", "grid", "card_type", "card_index",
                                            "board_id", "recycling_steps",
                                            "diffusion_samples")}
    t0 = time.perf_counter()
    # TWICE, in one process. A digest that matches across trees proves nothing if the fold
    # does not reproduce its own bytes, so the A/A floor is taken here rather than assumed
    # from the seed.
    res["folds"] = []
    for i in range(2):
        fold_s, m = one_fold()
        rec = {"run": i, "s": round(fold_s, 2), "n_tokens": m.get("n_tokens"),
               "plddt": m.get("plddt"), "seed": meta["job_cfg"].get("seed"),
               "digest": sha_dir(struct_dir)}
        res["folds"].append(rec)
        print(f"[{time.perf_counter()-t0:.0f}s] fold {i} {fold_s:.1f}s "
              f"n_tokens={m.get('n_tokens')} plddt={m.get('plddt')}", flush=True)
        print("  digest " + json.dumps(rec["digest"]), flush=True)
    res["fold"] = res["folds"][0]
    res["fold_digest"] = res["folds"][0]["digest"]
    res["fold_aa_bit_identical"] = res["folds"][0]["digest"] == res["folds"][1]["digest"]
    print(f"  A/A same process: bit-identical={res['fold_aa_bit_identical']}", flush=True)

    # ---------------------------------------------------------------- the walk, after a forward
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import walk_device_weights
    from tt_bio.train.lora import Parameters
    from tt_bio.train.optim import AdamW

    model = state.model
    walked = list(walk_device_weights(model))
    print(f"[{time.perf_counter()-t0:.0f}s] walked {len(walked)} device tensors from "
          f"{type(model).__name__}", flush=True)

    params = Parameters({path: ag.parameter(t) for path, _o, _k, t in walked},
                        slots={path: (o, k) for path, o, k, _t in walked})
    # `Tensor.free` is the third writer of a leaf's value, and the only one that can fire
    # while a tape is open: it moves an L1-resident tensor to DRAM rather than refusing the
    # free. The registration now follows that write too, but the MODEL's slot does not, so
    # an L1-resident weight would still need a `rebind()`. Whether that is reachable at all
    # is a property of where this model keeps its weights, so it is counted, not assumed.
    l1 = 0
    for _p, _o, _k, t in walked:
        try:
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                l1 += 1
        except Exception:                                                     # noqa: BLE001
            pass
    res["walk"] = {"device_tensors_walked": len(walked), "registered": len(params),
                   "l1_resident_walked_weights": l1}

    def resolved():
        n = 0
        for name, (owner, key) in params.slots.items():
            raw = owner[key] if isinstance(owner, (dict, list)) else getattr(owner, key)
            if ag.parameter_for(raw) is not None:
                n += 1
        return n

    res["walk"]["resolves_before_step"] = resolved()
    print(f"  resolves before any step: {res['walk']['resolves_before_step']} of {len(params)}",
          flush=True)

    if not a.skip_step:
        opt = AdamW(params, lr=1.8e-3, betas=(0.9, 0.95), weight_decay=0.0, clip_norm=0.0)
        print(f"[{time.perf_counter()-t0:.0f}s] optimizer built over {len(params)} tensors",
              flush=True)
        for n, t in params.items():
            t.grad = np.full(opt.master[n].shape, 1e-3, dtype=np.float32)
        opt.step()
        moved = params.rebind()
        res["walk"]["rebound"] = moved
        res["walk"]["resolves_after_step"] = resolved()
        print(f"[{time.perf_counter()-t0:.0f}s] after one step: rebound {moved}, "
              f"resolves {res['walk']['resolves_after_step']} of {len(params)}", flush=True)

    # ------------------------------------------------- the count as a share of the squared norm
    ref = torch.load(REF_GRADS, map_location="cpu", weights_only=False)
    ref_sq = {k: float((v.double() ** 2).sum()) for k, v in ref.items() if torch.is_tensor(v)}
    total_sq = sum(ref_sq.values())

    # Our walked paths carry the device port's own attribute names, upstream's carry the torch
    # module's, and the port transposes on the way to the device. So the bijection is by VALUE,
    # not by name, on quantities a transpose cannot move: element count, sorted shape, L1, L2
    # and the absolute maximum. All four are taken on the CHECKPOINT CAST TO BFLOAT16, because
    # that is what the device holds -- fingerprinting fp32 against a bf16 readback matched 393
    # of 4,170 and the mismatch was the cast, not the model.
    def fp(x):
        x = x.double().reshape(-1)
        return (float(x.abs().sum()), float((x * x).sum()), float(x.abs().max()))

    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    buckets = {}
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point() or k not in ref_sq:
            continue
        buckets.setdefault((v.numel(), tuple(sorted(v.shape))), []).append(
            (k, fp(v.to(torch.bfloat16))))

    dev_fp = {}
    for path, _o, _k, t in walked:
        try:
            x = ttnn.to_torch(t)
        except Exception:                                                     # noqa: BLE001
            dev_fp[path] = None
            continue
        dev_fp[path] = (int(x.numel()), tuple(sorted(x.shape))) + fp(x)

    def close(a, b, tol=2e-2):
        return all(abs(p - q) <= tol * (abs(q) + 1e-12) for p, q in zip(a, b))

    matched, unmatched, ambiguous = {}, [], []
    for path, f in dev_fp.items():
        if f is None:
            unmatched.append(path)
            continue
        cand = [k for k, g in buckets.get((f[0], f[1]), []) if close(f[2:], g)]
        if len(cand) == 1:
            matched[path] = cand[0]
        elif len(cand) > 1:
            ambiguous.append(path)
        else:
            unmatched.append(path)

    hit = set(matched.values())
    missing = sorted(set(ref_sq) - hit)
    res["reach"] = {
        "reference_tensors": len(ref_sq),
        "reference_sq_norm": total_sq,
        "walked_device_tensors": len(walked),
        "matched_to_reference": len(hit),
        "ambiguous_walked_paths": len(ambiguous),
        "unmatched_walked_paths": len(unmatched),
        "sq_norm_matched": sum(ref_sq[n] for n in hit),
        "pct_of_model_sq_grad_norm": 100.0 * sum(ref_sq[n] for n in hit) / total_sq,
        "reference_names_not_matched_count": len(missing),
        "sq_norm_not_matched": sum(ref_sq[n] for n in missing),
        "pct_not_matched": 100.0 * sum(ref_sq[n] for n in missing) / total_sq,
        "reference_names_not_matched_top20_by_sq_norm":
            [[n, ref_sq[n]] for n in sorted(missing, key=lambda n: -ref_sq[n])[:20]],
        "unmatched_walked_paths_sample": unmatched[:20],
    }
    # Dumped so the bijection can be re-derived offline without another card-hour.
    res["device_fingerprints"] = {k: v for k, v in dev_fp.items()}
    res["matched_names"] = matched
    print("[reach] " + json.dumps({k: v for k, v in res["reach"].items()
                                   if not k.startswith("reference_names_not_matched_top")
                                   and k != "unmatched_walked_paths_sample"}, indent=1),
          flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
