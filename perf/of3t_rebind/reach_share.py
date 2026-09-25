#!/usr/bin/env python3
"""REACH's share of the squared gradient norm, computed as a COVERING rather than a naming.

`reach.py` asked each walked device tensor for a UNIQUE reference name and reported 93.6638 %
as a lower bound, because 829 of 3,932 walked tensors matched a value class with more than one
member. That understates the answer by construction, and the reason is worth stating: two
reference tensors with identical values -- every all-ones LayerNorm weight of a given width --
cannot be told apart by any value fingerprint, and they do not need to be. The question is not
"which name is this tensor" but "is this reference name covered by the walked set", and a value
class with as many device tensors as reference names covers all of them however they are paired.

So: maximum bipartite matching between walked device tensors and reference names, computed per
(numel, sorted shape) bucket where the adjacency is dense within a value class. A class with
n_dev >= n_ref is fully covered and contributes no ambiguity to the SHARE at all. A class with
n_dev < n_ref is genuinely short, and for those the covered norm is reported as a RANGE -- the
n_dev smallest and the n_dev largest -- so an under-saturated class cannot be read as a win.

No card. Reads the fingerprints `reach.py` dumped, so the bijection can be re-derived without
another card-hour.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch                                                                # noqa: E402

REF_GRADS = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
TOL = 2e-2


def fp(x):
    x = x.double().reshape(-1)
    return (float(x.abs().sum()), float((x * x).sum()), float(x.abs().max()))


def close(a, b, tol=TOL):
    return all(abs(p - q) <= tol * (abs(q) + 1e-12) for p, q in zip(a, b))


reach = json.load(open(sys.argv[1]))
dev_fp = reach["device_fingerprints"]

ref = torch.load(REF_GRADS, map_location="cpu", weights_only=False)
ref_sq = {k: float((v.double() ** 2).sum()) for k, v in ref.items() if torch.is_tensor(v)}
total_sq = sum(ref_sq.values())

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}

# Reference side, in the dtype the device holds.
ref_entries, no_weight = [], []
for name in ref_sq:
    v = sd.get(name)
    if v is None or not torch.is_tensor(v) or not v.is_floating_point():
        no_weight.append(name)
        continue
    ref_entries.append((name, (v.numel(), tuple(sorted(v.shape))) + fp(v.to(torch.bfloat16))))

dev_entries = [(p, (f[0], tuple(f[1])) + tuple(f[2:]))
               for p, f in dev_fp.items() if f is not None]
dev_unreadable = [p for p, f in dev_fp.items() if f is None]

buckets = {}
for name, f in ref_entries:
    buckets.setdefault(f[:2], {"ref": [], "dev": []})["ref"].append((name, f[2:]))
for path, f in dev_entries:
    buckets.setdefault(f[:2], {"ref": [], "dev": []})["dev"].append((path, f[2:]))

covered, short_classes, uncovered = [], [], []
lo_extra, hi_extra = 0.0, 0.0
for key, b in buckets.items():
    refs, devs = list(b["ref"]), list(b["dev"])
    if not refs:
        continue
    used = [False] * len(devs)
    # Cluster into value classes: a reference name seeds a class, every ref and dev within
    # tolerance of the seed joins it. Adjacency inside a class is dense, so the matching size
    # is simply min(n_ref, n_dev).
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
        if len(cls_dev) >= len(names):
            covered.extend(names)
        else:
            by_norm = sorted(names, key=lambda n: ref_sq[n])
            n = len(cls_dev)
            covered.extend(by_norm[:0])                       # counted in the range instead
            short_classes.append({"bucket": list(key), "n_ref": len(names),
                                  "n_dev": n,
                                  "names_by_norm": [[x, ref_sq[x]] for x in by_norm]})
            lo_extra += sum(ref_sq[x] for x in by_norm[:n])
            hi_extra += sum(ref_sq[x] for x in by_norm[len(names) - n:])
            uncovered.extend(by_norm[n:] if n else by_norm)

sure = sum(ref_sq[n] for n in covered)
out = {
    "reference_tensors": len(ref_sq),
    "reference_sq_norm": total_sq,
    "walked_device_tensors": len(dev_fp),
    "device_tensors_unreadable": len(dev_unreadable),
    "reference_names_with_no_checkpoint_weight": len(no_weight),
    "sq_norm_of_those": sum(ref_sq[n] for n in no_weight),
    "fully_covered_reference_names": len(covered),
    "sq_norm_fully_covered": sure,
    "pct_fully_covered": 100.0 * sure / total_sq,
    "under_saturated_value_classes": len(short_classes),
    "pct_lower_bound": 100.0 * (sure + lo_extra) / total_sq,
    "pct_upper_bound": 100.0 * (sure + hi_extra) / total_sq,
    "reference_names_not_covered_count": len(set(ref_sq) - set(covered)),
    "sq_norm_not_covered": total_sq - sure,
    "top_uncovered_by_sq_norm":
        [[n, ref_sq[n]] for n in sorted(set(ref_sq) - set(covered),
                                        key=lambda n: -ref_sq[n])[:15]],
    "short_class_detail": short_classes[:10],
}
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("top_uncovered_by_sq_norm", "short_class_detail")}, indent=1))
print("top uncovered by squared norm:")
for n, q in out["top_uncovered_by_sq_norm"]:
    print("   %-72s %.6f  (%.4f %%)" % (n, q, 100.0 * q / total_sq))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("wrote " + sys.argv[2])
