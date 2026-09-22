#!/usr/bin/env python3
"""Upstream 0.4.3's own forward AND gradient at the diffusion boundary, out of ONE process.

The campaign has upstream's own bf16 GRADIENT at this boundary
(`perf/of3t_cond043/BARS043.json`, scope median 1.2940662e-01, mass-weighted 1.8817643e-01,
n=761) and it has upstream's forward SECONDS (`FLOOR043_bf16auto.json`, 24.757 s). It does not
have upstream's forward ACCURACY anywhere, and that single missing number is what separates the
two explanations of D30/D58's forward-to-gradient factor:

  a boundary artifact of the D206 class is OURS, so upstream's own ratio would be near 1x;
  a conditioning property belongs to the function, so upstream's own ratio is the same ~11x.

Both halves of the ratio have to come out of one process on one host. An upstream bf16 arm is
host-dependent -- `of3t-bwdaccum` control 7 rebuilt one and read 3.739355e-01 against the
4.007237e-01 on record, 6.7 % apart, while its float64 arm reproduced to 15 digits -- so a
forward measured here divided by a gradient quoted from qb2 would be a cross-frame ratio, which
is the defect `a-reference-is-part-of-the-measurements-identity` names. This file therefore
refuses to read a gradient from an artifact: it recomputes both.

The forward statistic is the one `perf/of3t_diffusion/device_gradient.py:751` uses for our own
arm, so the two are comparable without a conversion: per structure k,
rel_l2(xl[0,k], S["xl_out"][0,k]), median over the 48. The gradient statistic is per-tensor
rel_l2 against S["grad_f64"], median and mass-weighted, reported on the full upstream scope AND
on the 547 tensors our device arm compares (`--match-scope`), because a ratio whose two halves
are on different scopes names two different functions.

Two guards `perf/of3t_condtrans/floor_bf16.py` does not have, both from D141: the checkpoint load
ASSERTS on unexpected keys in the diffusion scope instead of discarding them under strict=False,
and the parameter-name fingerprint of the tree is checked against the capture's own grad_f64 key
set. Silently dropping 24 trained tensors is how the campaign spent two rows on a 6.62x that was
an architecture difference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch


EXPECTED = Path("/home/ttuser/of3t-campaign-refs/EXPECTED_DIGESTS.json")
DIGEST_TO_VERSION = {"of3pkg043": "0.4.3", "of3pkg050": "0.5.0"}


def tree_sha256(root: Path):
    """of3t-campaign-refs/tree_digest.py's rule, recomputed here rather than quoted.

    sha256 over the sorted per-file sha256 hex strings of every .py under the package root,
    concatenated with no separator. The rule is restated because a digest quoted without its
    rule cannot be reproduced, and because a digest with only one carrier is a transcription:
    this one is recomputed from the bytes on disk and compared against EXPECTED_DIGESTS.json,
    which lives outside this namespace and was written by a different row.
    """
    import hashlib
    per = sorted(hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob("*.py"))
    return hashlib.sha256("".join(per).encode()).hexdigest(), len(per)


def upstream_identity():
    """Identify the openfold3 sys.path actually resolved, from the tree itself.

    Two independent pins, because on qb1 the 0.4.3 tree has no co-located metadata at all and
    the arms on record were pinned by metadata on qb2:

      the TREE DIGEST, recomputed in-process under of3t-campaign-refs/tree_digest.py's rule and
      looked up in EXPECTED_DIGESTS.json. This pins the bytes, which for a reference ARE the
      function, and it is strictly stronger than a version string a build wrote.

      the LAYERNORM SIGNATURE, a behaviour no label can lie about: 0.4.3 rounds the affine
      parameter to x.dtype, 0.5.0 upcasts it to float32.

    `importlib.metadata` is deliberately not used: it scans every sys.path entry, so a 0.5.0
    dist-info anywhere behind a 0.4.3 tree makes it report 0.5.0 for code that is 0.4.3.
    """
    spec = importlib.util.find_spec("openfold3")
    if spec is None or not spec.origin:
        raise SystemExit("openfold3 is not importable; PYTHONPATH is wrong")
    pkg = Path(spec.origin).resolve().parent
    root = pkg.parent
    ver = meta = None
    for c in (sorted(root.glob("*.dist-info/METADATA"))
              + sorted(root.glob("*.egg-info/PKG-INFO")) + [root / "PKG-INFO"]):
        if not c.exists():
            continue
        for line in c.read_text(errors="replace").splitlines():
            if line.startswith("Version:"):
                ver, meta = line.split(":", 1)[1].strip(), str(c)
                break
        if ver:
            break
    norm = pkg / "core" / "model" / "primitives" / "normalization.py"
    sig = "normalization.py_absent"
    if norm.exists():
        src = norm.read_text(errors="replace")
        sig = ("layernorm_upcasts_affine_to_float32" if "self.weight.float()" in src
               else "layernorm_keeps_affine_in_x_dtype" if "self.weight.to(dtype=d)" in src
               else "layernorm_unrecognised")
    dg, nfiles = tree_sha256(pkg)
    exp = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    hit = [k for k, v in exp.items() if v == dg]
    dver = DIGEST_TO_VERSION.get(hit[0]) if hit else None
    if ver is None and dver is not None:
        ver, meta = dver, f"tree_sha256={dg} matches {EXPECTED}:{hit[0]} ({nfiles} .py files)"
    return str(root), ver, meta, sig, {"tree_sha256": dg, "n_py_files": nfiles,
                                       "expected_digests": str(EXPECTED),
                                       "digest_matches_entry": hit[0] if hit else None,
                                       "version_from_digest": dver,
                                       "rule": "sha256 of the sorted per-file sha256 hex "
                                               "strings, concatenated, of every .py under the "
                                               "package root"}


def cast(x, dt):
    if torch.is_tensor(x):
        return x.to(dt) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: cast(v, dt) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cast(v, dt) for v in x)
    return x


def rel(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    nb = float(torch.linalg.vector_norm(b))
    return float(torch.linalg.vector_norm(a - b)) / (nb + 1e-300), nb


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def score_grads(grads, ref, names=None, a14=1e-12):
    """Per-tensor rel_l2 against the float64 reference. Median, mass-weighted, A14 accounted."""
    num = den = 0.0
    rels, dropped, worst, worst_n = [], 0, -1.0, None
    per = {}
    for nm, gg in grads.items():
        if names is not None and nm not in names:
            continue
        r = ref.get(nm)
        if gg is None or r is None:
            continue
        rr = r.reshape(-1).double()
        s = float(torch.linalg.vector_norm(rr))
        if s <= a14:
            dropped += 1
            continue
        e = float(torch.linalg.vector_norm(gg.reshape(-1).double() - rr))
        num += e * e
        den += s * s
        rels.append(e / s)
        per[nm] = e / s
        if e / s > worst:
            worst, worst_n = e / s, nm
    return {"n": len(rels), "median_rel": median(rels), "a14_dropped": dropped,
            "mass_weighted_rel": (num / den) ** 0.5 if den else None,
            "error_mass": num, "reference_mass": den,
            "worst_rel": worst if rels else None, "worst_tensor": worst_n,
            "min_ref_norm": min((float(torch.linalg.vector_norm(ref[n].reshape(-1).double()))
                                 for n in per), default=None)}, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=("bf16auto", "f32", "f64"))
    ap.add_argument("--cap", type=Path, required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/ttuser/of3-weights/of3-p2-155k.pt"))
    ap.add_argument("--match-scope", default="",
                    help="a *_per_tensor.json from our own device arm; its tensor names become "
                         "the matched scope the ratio is also reported on")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    t0 = time.time()

    root, ver, meta, ln_sig, dig = upstream_identity()
    print(f"upstream openfold3 {ver} at {root} (from {meta}), LayerNorm: {ln_sig}", flush=True)
    if ver != a.expect_version:
        print(f"HARD FAILURE: arm declares {a.expect_version}, sys.path resolved {ver} "
              f"(digest {dig})", flush=True)
        return 3
    if dig["version_from_digest"] != a.expect_version:
        print(f"HARD FAILURE: the tree's own bytes do not match the campaign's "
              f"{a.expect_version} reference: {json.dumps(dig)}", flush=True)
        return 3

    import bundle_min as BM

    B = torch.load(a.cap / "diffusion_boundary.pt", map_location="cpu", weights_only=False)
    S = torch.load(a.cap / "sub_boundary.pt", map_location="cpu", weights_only=False)
    kwargs, cot, ref_grad, xl_ref = B["kwargs"], B["cot"], S["grad_f64"], S["xl_out"]
    print(f"[{time.time()-t0:.0f}s] boundary loaded, cot norm "
          f"{float(cot.double().norm()):.6e}, {len(ref_grad)} reference tensors, "
          f"xl_out {tuple(xl_ref.shape)} {xl_ref.dtype}", flush=True)
    if xl_ref.dtype != torch.float64:
        print(f"HARD FAILURE: xl_out is {xl_ref.dtype}, not float64; it is not a reference")
        return 4
    del B

    dt = torch.float64 if a.policy == "f64" else torch.float32
    built = BM.build(dt, 20260919, "cpu", num_recycles=0)
    model = built[1]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    got = model.load_state_dict(sd, strict=False)
    dm = model.diffusion_module
    # D141's guard, the one whose absence cost two rows: a key the checkpoint carries that this
    # tree has no home for is a different architecture, not a loose load. Scoped to the
    # diffusion module because that is the only scope this arm measures.
    dm_unexpected = [k for k in got.unexpected_keys if k.startswith("diffusion_module.")]
    dm_missing = [k for k in got.missing_keys if k.startswith("diffusion_module.")]
    tree_names = {n for n, _ in dm.named_parameters()}
    fp = {"n_parameters": len(tree_names),
          "n_perblock_layer_norm_z": sum(1 for n in tree_names
                                         if "blocks." in n and n.endswith("layer_norm_z.weight")),
          "capture_keys_equal_tree_parameters": set(ref_grad) == tree_names,
          "dm_unexpected_keys": len(dm_unexpected), "dm_missing_keys": len(dm_missing),
          "dm_unexpected_sample": dm_unexpected[:4], "dm_missing_sample": dm_missing[:4]}
    print(f"[{time.time()-t0:.0f}s] model at {dt}, fingerprint {json.dumps(fp)}", flush=True)
    if dm_unexpected or not fp["capture_keys_equal_tree_parameters"]:
        print("HARD FAILURE: the tree cannot express the checkpoint's diffusion architecture, "
              "or the capture's reference keys are not this tree's parameters (D141)", flush=True)
        return 5
    del ck, sd

    probe = {}

    def dtype_hook(mod, inp, out):
        if torch.is_tensor(out):
            probe.setdefault("dit_block0_linear_out_dtype", str(out.dtype))

    hs = []
    for name, mod in dm.diffusion_transformer.blocks[0].named_modules():
        if isinstance(mod, torch.nn.Linear):
            probe["probed_module"] = f"diffusion_transformer.blocks.0.{name}"
            hs.append(mod.register_forward_hook(dtype_hook))
            break

    kw = cast(kwargs, dt)
    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if a.policy == "bf16auto"
           else BM.no_autocast() if a.policy == "f64"
           else torch.autocast("cpu", enabled=False))
    load0 = os.getloadavg()
    t1 = time.time()
    with ctx:
        xl = dm(**kw)
    t_fwd = time.time() - t1
    for h in hs:
        h.remove()
    print(f"[{time.time()-t0:.0f}s] forward {t_fwd:.1f}s, xl {tuple(xl.shape)} {xl.dtype}, "
          f"probe {probe}", flush=True)
    if a.policy == "bf16auto" and probe.get("dit_block0_linear_out_dtype") != "torch.bfloat16":
        print("HARD FAILURE: the bf16 recipe did not reach the kernels", flush=True)
        return 2
    if tuple(xl.shape) != tuple(xl_ref.shape):
        print(f"HARD FAILURE: xl {tuple(xl.shape)} is not the reference's "
              f"{tuple(xl_ref.shape)}; these are not the same function's output", flush=True)
        return 6

    # ---- the forward half, in device_gradient.py's own statistic --------------------------
    fwd, fwd_ref_norm = [], []
    for k in range(xl_ref.shape[1]):
        r, nb = rel(xl[0, k].detach(), xl_ref[0, k])
        fwd.append(r)
        fwd_ref_norm.append(nb)
    fwd_med = median(fwd)
    print(f"[{time.time()-t0:.0f}s] FORWARD median {fwd_med:.10e} over {len(fwd)} structures "
          f"(min {min(fwd):.6e}, max {max(fwd):.6e})", flush=True)

    names, params = zip(*[(n, p) for n, p in dm.named_parameters()])
    t1 = time.time()
    g = torch.autograd.grad(xl, tuple(params), grad_outputs=cot.to(xl.dtype),
                            allow_unused=True, retain_graph=False)
    t_bwd = time.time() - t1
    load1 = os.getloadavg()
    grads = {n: (x.detach().to(torch.float64 if a.policy == "f64" else torch.float32)
                 if x is not None else None) for n, x in zip(names, g)}
    print(f"[{time.time()-t0:.0f}s] backward {t_bwd:.1f}s", flush=True)
    if a.dump:
        torch.save({"policy": a.policy, "grads": grads, "cap": str(a.cap),
                    "forward_rel": fwd, "xl": xl.detach()}, a.dump)

    full, _ = score_grads(grads, ref_grad)
    matched = None
    match_names = None
    if a.match_scope:
        j = json.load(open(a.match_scope))
        match_names = {t["tensor"] for t in j["per_tensor"]}
        matched, _ = score_grads(grads, ref_grad, names=match_names)

    def ratio(gr):
        return None if not (gr and gr["median_rel"] and fwd_med) else gr["median_rel"] / fwd_med

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "tag": a.tag, "policy": a.policy, "cap": str(a.cap), "ckpt": str(a.ckpt),
        "upstream_pkg": root, "upstream_version": ver,
        "upstream_version_declared": a.expect_version, "upstream_version_source": meta,
        "LAYERNORM_SIGNATURE": ln_sig, "FINGERPRINT": fp, "TREE_DIGEST": dig,
        "host": socket.gethostname(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__, "python": sys.version.split()[0],
        "loadavg_before_forward": load0, "loadavg_after_backward": load1,
        "forward_seconds": t_fwd, "backward_seconds": t_bwd,
        "xl_dtype": str(xl.dtype), "DTYPE_PROBE": probe,
        "FORWARD": {"statistic": "per structure k, rel_l2(xl[0,k], S['xl_out'][0,k]); the same "
                                 "statistic as perf/of3t_diffusion/device_gradient.py:751",
                    "n_structures": len(fwd), "median": fwd_med,
                    "min": min(fwd), "max": max(fwd), "per_structure": fwd,
                    "min_reference_norm": min(fwd_ref_norm)},
        "GRADIENT_full_scope": full,
        "GRADIENT_matched_scope": matched,
        "matched_scope_source": a.match_scope or None,
        "matched_scope_n_requested": len(match_names) if match_names else None,
        "RATIO_full_scope": ratio(full),
        "RATIO_matched_scope": ratio(matched),
        "RATIO_definition": "gradient median rel_l2 over forward median rel_l2, both against the "
                            "float64 reference of THIS capture, both out of this one process on "
                            "this one host. Comparable to our device arm's 11.026x, which is "
                            "9.344246e-02 / 8.474801e-03 on the same capture and the same "
                            "forward statistic.",
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    show = {k: v for k, v in rep.items() if k not in ("FORWARD",)}
    show["FORWARD"] = {k: v for k, v in rep["FORWARD"].items() if k != "per_structure"}
    print(json.dumps(show, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
