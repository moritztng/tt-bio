#!/usr/bin/env python3
"""D127 Deliverable 1 -- re-derive, from the float64 reference alone, how much of OpenFold3's
squared gradient norm is carried by parameters this port never puts on the device.

`of3t-rebind` published 3.6438 % as the sum of two section totals, `aux_heads` (2.8431 %) and
`input_embedder` (0.8007 %). A section total is not the same quantity as the uncovered mass
inside that section, and neither is a guarantee that the two sections are the only ones. This
script computes the uncovered mass directly and attributes every point of it.

Method, same covering as `perf/of3t_rebind/reach_share.py` and deliberately re-implemented
rather than imported, so the published figure is checked and not echoed: bucket reference
names and walked device tensors by (numel, sorted shape), cluster each bucket into value
classes by an L1/L2/absmax fingerprint taken on the checkpoint CAST TO BFLOAT16, and call a
class covered when it holds at least as many device tensors as reference names.

Per uncovered name it also records WHY it is uncovered:
  hard   -- its value class holds zero device tensors. Nothing on the card has these values.
  short  -- its class holds some device tensors but fewer than reference names, so the
            covering cannot say which of them is the uncovered one. These are reported as a
            range, never as a point.

A24: every input is verified by digest before it is read.
A15: every figure is a share of the reference's squared gradient norm, never a count.
"""
import hashlib
import json
import os
import sys

import torch

REF_GRADS = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
REACH = "perf/of3t_rebind/REACH_fixed.json"

DIGESTS = {
    REF_GRADS: "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4",
    CKPT: "af09eac4f29cef856633af07558cb143226fe95ebbef2c20921769d4a5f4bee4",
}
TOL = 2e-2


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fp(x):
    x = x.double().reshape(-1)
    return (float(x.abs().sum()), float((x * x).sum()), float(x.abs().max()))


def close(a, b, tol=TOL):
    return all(abs(p - q) <= tol * (abs(q) + 1e-12) for p, q in zip(a, b))


# ---- A24: pin before loading -------------------------------------------------------------
pins = {}
for path, want in DIGESTS.items():
    got = digest(path)
    pins[path] = got
    if got != want:
        sys.exit("STOP: digest mismatch on %s\n  expected %s\n  got      %s" % (path, want, got))
pins[REACH] = digest(REACH)
print("digests verified:")
for p, d in pins.items():
    print("   %-64s %s" % (os.path.basename(p), d))

# ---- reference ----------------------------------------------------------------------------
ref = torch.load(REF_GRADS, map_location="cpu", weights_only=False)
ref_sq = {k: float((v.double() ** 2).sum()) for k, v in ref.items() if torch.is_tensor(v)}
TOTAL = sum(ref_sq.values())
del ref
print("\nreference: %d tensors, squared gradient norm %.15f" % (len(ref_sq), TOTAL))

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

dev_fp = json.load(open(REACH))["device_fingerprints"]
print("walked device tensors: %d" % len(dev_fp))

# ---- covering -----------------------------------------------------------------------------
ref_entries, no_weight = [], []
for name in ref_sq:
    v = sd.get(name)
    if v is None or not torch.is_tensor(v) or not v.is_floating_point():
        no_weight.append(name)
        continue
    ref_entries.append((name, (v.numel(), tuple(sorted(v.shape))) + fp(v.to(torch.bfloat16))))
dev_entries = [(p, (f[0], tuple(f[1])) + tuple(f[2:])) for p, f in dev_fp.items() if f]

buckets = {}
for name, f in ref_entries:
    buckets.setdefault(f[:2], {"ref": [], "dev": []})["ref"].append((name, f[2:]))
for path, f in dev_entries:
    buckets.setdefault(f[:2], {"ref": [], "dev": []})["dev"].append((path, f[2:]))

covered, hard, short_names = [], [], []
lo_extra, hi_extra = 0.0, 0.0
short_classes = []
for key, b in buckets.items():
    refs, devs = list(b["ref"]), list(b["dev"])
    if not refs:
        continue
    used = [False] * len(devs)
    seen = [False] * len(refs)
    for i, (rname, rv) in enumerate(refs):
        if seen[i]:
            continue
        cls_ref = [j for j in range(len(refs)) if not seen[j] and close(refs[j][1], rv)]
        for j in cls_ref:
            seen[j] = True
        cls_dev = [k for k in range(len(devs)) if not used[k] and close(devs[k][1], rv)]
        for k in cls_dev:
            used[k] = True
        names = [refs[j][0] for j in cls_ref]
        n = len(cls_dev)
        if n >= len(names):
            covered.extend(names)
        elif n == 0:
            hard.extend(names)
        else:
            by_norm = sorted(names, key=lambda x: ref_sq[x])
            lo_extra += sum(ref_sq[x] for x in by_norm[:n])
            hi_extra += sum(ref_sq[x] for x in by_norm[len(names) - n:])
            short_names.extend(names)
            short_classes.append({"bucket": [key[0], list(key[1])], "n_ref": len(names),
                                  "n_dev": n, "sq": sum(ref_sq[x] for x in names),
                                  "names": names})

sure = sum(ref_sq[n] for n in covered)
hard_sq = sum(ref_sq[n] for n in hard)
short_sq = sum(ref_sq[n] for n in short_names)
print("\ncovering (share of the squared gradient norm, A15):")
print("   fully covered          %5d names   %.6f   %8.4f %%" % (len(covered), sure, 100 * sure / TOTAL))
print("   hard-uncovered         %5d names   %.6f   %8.4f %%" % (len(hard), hard_sq, 100 * hard_sq / TOTAL))
print("   short-class residue    %5d names   %.6f   %8.4f %%  (covered share of it in [%.4f, %.4f] %%)"
      % (len(short_names), short_sq, 100 * short_sq / TOTAL, 100 * lo_extra / TOTAL, 100 * hi_extra / TOTAL))
print("   reach                  fully-covered %.4f %%, bounded [%.4f, %.4f] %%"
      % (100 * sure / TOTAL, 100 * (sure + lo_extra) / TOTAL, 100 * (sure + hi_extra) / TOTAL))
print("   not reached            %.4f %%, bounded [%.4f, %.4f] %%"
      % (100 * (TOTAL - sure) / TOTAL, 100 * (TOTAL - sure - hi_extra) / TOTAL,
         100 * (TOTAL - sure - lo_extra) / TOTAL))

# ---- attribute the hard-uncovered mass ----------------------------------------------------
# Scopes are defined by the SHIPPED CODE that applies the weight on the host, not by the
# reference's section names. The three host families, each named at file and line in WHY.
SCOPES = [
    ("aux_heads output projections",
     lambda n: n.startswith("aux_heads.") and not n.startswith("aux_heads.confidence_head.pf.")),
    ("aux_heads confidence pairformer",
     lambda n: n.startswith("aux_heads.confidence_head.pf.")),
    ("input_embedder atom encoder (ref_atom_feature_embedder)",
     lambda n: n.startswith("input_embedder.atom_attn_enc.ref_atom_feature_embedder.")),
    ("input_embedder atom encoder (pair completion + linear_q)",
     lambda n: n.startswith("input_embedder.atom_attn_enc.") and "ref_atom_feature_embedder" not in n),
    ("input_embedder, rest of section",
     lambda n: n.startswith("input_embedder.") and not n.startswith("input_embedder.atom_attn_enc.")),
    ("diffusion_module atom encoder (ref_atom_feature_embedder)",
     lambda n: n.startswith("diffusion_module.atom_attn_enc.ref_atom_feature_embedder.")),
    ("diffusion_module, rest of section",
     lambda n: n.startswith("diffusion_module.")),
    ("pairformer_stack", lambda n: n.startswith("pairformer_stack.")),
    ("msa_module", lambda n: n.startswith("msa_module.")),
]


def scope_of(name):
    for label, pred in SCOPES:
        if pred(name):
            return label
    return "other: " + name.split(".")[0]


def table(names, title):
    acc = {}
    for n in names:
        s = scope_of(n)
        e = acc.setdefault(s, {"n": 0, "sq": 0.0, "top": ("", 0.0)})
        e["n"] += 1
        e["sq"] += ref_sq[n]
        if ref_sq[n] > e["top"][1]:
            e["top"] = (n, ref_sq[n])
    print("\n%s" % title)
    tot = 0.0
    for s, e in sorted(acc.items(), key=lambda kv: -kv[1]["sq"]):
        tot += e["sq"]
        print("   %-56s %4d  %8.6f  %8.4f %%" % (s, e["n"], e["sq"], 100 * e["sq"] / TOTAL))
        if e["top"][1] > 0:
            print("        heaviest: %-62s %8.4f %%" % (e["top"][0], 100 * e["top"][1] / TOTAL))
    print("   %-56s %4d  %8.6f  %8.4f %%" % ("TOTAL", len(names), tot, 100 * tot / TOTAL))
    return acc


hard_acc = table(hard, "HARD-UNCOVERED by scope (no device tensor holds these values at all):")
short_acc = table(short_names, "SHORT-CLASS RESIDUE by scope (covering cannot resolve; reported as a range):")

# ---- section totals, for comparison with what rebind published ----------------------------
print("\nreference SECTION totals, for comparison with the published figures:")
sec = {}
for n, q in ref_sq.items():
    sec.setdefault(n.split(".")[0], [0, 0.0])
    sec[n.split(".")[0]][0] += 1
    sec[n.split(".")[0]][1] += q
for s, (c, q) in sorted(sec.items(), key=lambda kv: -kv[1][1]):
    hs = sum(ref_sq[n] for n in hard if n.split(".")[0] == s)
    print("   %-24s %5d tensors  %8.4f %% of model   hard-uncovered inside it %8.4f %%"
          % (s, c, 100 * q / TOTAL, 100 * hs / TOTAL))

out = {
    "digests": pins,
    "reference_sq_norm": TOTAL,
    "reference_tensors": len(ref_sq),
    "walked_device_tensors": len(dev_fp),
    "fully_covered_names": len(covered),
    "sq_fully_covered": sure,
    "pct_fully_covered": 100 * sure / TOTAL,
    "pct_reach_lower": 100 * (sure + lo_extra) / TOTAL,
    "pct_reach_upper": 100 * (sure + hi_extra) / TOTAL,
    "hard_uncovered_names": len(hard),
    "sq_hard_uncovered": hard_sq,
    "pct_hard_uncovered": 100 * hard_sq / TOTAL,
    "short_class_names": len(short_names),
    "pct_short_class_total": 100 * short_sq / TOTAL,
    "pct_short_class_uncovered_lower": 100 * (short_sq - hi_extra) / TOTAL,
    "pct_short_class_uncovered_upper": 100 * (short_sq - lo_extra) / TOTAL,
    "hard_by_scope": {k: {"n": v["n"], "sq": v["sq"], "pct": 100 * v["sq"] / TOTAL,
                          "heaviest": v["top"][0], "heaviest_pct": 100 * v["top"][1] / TOTAL}
                      for k, v in hard_acc.items()},
    "short_by_scope": {k: {"n": v["n"], "sq": v["sq"], "pct": 100 * v["sq"] / TOTAL}
                       for k, v in short_acc.items()},
    "hard_uncovered_detail": sorted(
        [[n, ref_sq[n], 100 * ref_sq[n] / TOTAL, scope_of(n)] for n in hard], key=lambda r: -r[1]),
}
os.makedirs("perf/of3t_hostops", exist_ok=True)
json.dump(out, open("perf/of3t_hostops/BOUND.json", "w"), indent=1)
print("\nwrote perf/of3t_hostops/BOUND.json")
