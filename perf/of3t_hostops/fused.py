#!/usr/bin/env python3
"""D127: separate a REACH GAP from a COVERING ARTIFACT.

`bound.py` reports 4.7624 % of the squared gradient norm as hard-uncovered -- no walked device
tensor holds those values. That is the right question for "can the optimizer address this
parameter by name" and the wrong one for "is this parameter on the card at all", because this
port fuses several reference weights into one device tensor. `openfold3_diffusion_transformer.py:125`
says so: "Fused padded qkv: cat([linear_q, linear_k, linear_v]) with head_dim 48->64", and
`protenix_weights.py:26` concatenates the triangle-multiplication projections the same way.

A fused tensor cannot match any single reference name's fingerprint. It can match the GROUP's,
and the test is free of the padding: the fingerprint is (sum |x|, sum x^2, max |x|), all three
invariant to appended zeros and to concatenation order. Group fingerprints are therefore
computed additively from the members and looked up among the walked device tensors.

Two rules keep this from finding what it wants to find:
  * a group needs at least TWO members. A one-member group's fingerprint is just that tensor's,
    so it re-asks the question `bound.py` already answered no to, and the first version of this
    script scored 675 such degenerate matches as hits.
  * each device tensor may be claimed by at most one group.
And the negative control is a SHUFFLE: the same families rebuilt from members taken out of
different blocks. Same shapes, same magnitudes, wrong combination. It must find nothing.
"""
import hashlib
import json
import os
import re
import sys

import torch

REF_GRADS = "/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
REACH = "perf/of3t_rebind/REACH_fixed.json"
BOUND = "perf/of3t_hostops/BOUND.json"
# Two tolerances, not one. The device fingerprints in REACH_fixed.json keep the absolute max of
# the tensor the card was handed, while the reference side is fingerprinted on the checkpoint
# cast to bfloat16, so the two absmaxes differ by one bf16 rounding (2^-9 = 1.95e-3 relative,
# either side, hence 4e-3). The L1 and L2 sums run over ~10^6 elements and that rounding averages
# out: the diffusion block-8 qkv group agrees with `sampler.dm.dit.blocks.8.qkv_w` to 2.9e-6 on
# L1 and 3.2e-6 on L2 while its absmax is 1.55e-3 off. A single tolerance either rejects a real
# fusion (2e-4 loses 105 of them) or admits noise (2e-2 matches 9 of 200 shuffled groups).
TOL_SUM = 1e-4
TOL_MAX = 4e-3
MIN_MEMBERS = 2

FAMILIES = [
    (r"^(?P<p>.*\.mha\.)linear_(q|k|v)\.weight$", "qkv.weight",
     "openfold3_diffusion_transformer.py:125 cat([linear_q, linear_k, linear_v]), head_dim padded"),
    (r"^(?P<p>.*\.mha\.)linear_(q|k|v)\.bias$", "qkv.bias",
     "same site, bias of the fused projection"),
    (r"^(?P<p>.*\.)linear_(a|b)_(p|g)\.weight$", "abpg.weight",
     "protenix_weights.py:26 and the device key trunk.*._gp_cache.(...,('p_a','g_a','p_b','g_b'))"),
]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    return (abs(a[0] - b[0]) <= TOL_SUM * abs(b[0])
            and abs(a[1] - b[1]) <= TOL_SUM * abs(b[1])
            and abs(a[2] - b[2]) <= TOL_MAX * abs(b[2]))


bound = json.load(open(BOUND))
for p, d in bound["digests"].items():
    if os.path.exists(p) and digest(p) != d:
        sys.exit("STOP: %s changed since bound.py ran" % p)
TOTAL = bound["reference_sq_norm"]
hard = {n: sq for n, sq, pct, scope in bound["hard_uncovered_detail"]}
scope = {n: s for n, sq, pct, s in bound["hard_uncovered_detail"]}
print("hard-uncovered going in: %d names, %.4f %% of the squared gradient norm"
      % (len(hard), 100 * sum(hard.values()) / TOTAL))

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
dev_fp = json.load(open(REACH))["device_fingerprints"]
dev_vals = [(p, tuple(f[2:])) for p, f in dev_fp.items() if f]

_cache = {}


def one(n):
    if n not in _cache:
        x = sd[n].to(torch.bfloat16).double().reshape(-1)
        _cache[n] = (float(x.abs().sum()), float((x * x).sum()), float(x.abs().max()))
    return _cache[n]


def group_fp(names):
    l1 = l2 = mx = 0.0
    for n in names:
        a, b, c = one(n)
        l1 += a
        l2 += b
        mx = max(mx, c)
    return (l1, l2, mx)


def probe(groups, claimed):
    hits, misses = [], []
    for g, names in sorted(groups.items()):
        if len(names) < MIN_MEMBERS:
            misses.append((g, names, "degenerate: %d member(s)" % len(names)))
            continue
        gf = group_fp(names)
        match = [p for p, v in dev_vals if close(v, gf) and p not in claimed]
        if match:
            claimed.add(match[0])
            hits.append((g, names, match[0]))
        else:
            misses.append((g, names, "no device tensor with these values"))
    return hits, misses


def build(pattern, label, pool):
    groups = {}
    for n in pool:
        m = re.match(pattern, n)
        if m:
            groups.setdefault(m.group("p") + label, []).append(n)
    return groups


print("\n=== fusion probe ===")
claimed, explained, total_moved, report = set(), set(), 0.0, []
for pattern, label, source in FAMILIES:
    groups = build(pattern, label, list(hard))
    hits, misses = probe(groups, claimed)
    moved = sum(hard.get(n, 0.0) for _, names, _ in hits for n in names)
    for g, names, dpath in hits:
        explained.update(names)
    total_moved += moved
    ok = sum(1 for g, names, why in misses if "degenerate" not in why)
    print("%-14s %3d groups (>=%d members: %d), %3d found on the card   moves %.4f %% out of the gap"
          % (label, len(groups), MIN_MEMBERS,
             sum(1 for v in groups.values() if len(v) >= MIN_MEMBERS), len(hits), 100 * moved / TOTAL))
    print("               source: %s" % source)
    if hits:
        print("               example: %s\n                        -> %s" % (hits[0][0], hits[0][2]))
    real_miss = [m for m in misses if "degenerate" not in m[2]]
    if real_miss:
        print("               %d real misses, first: %s" % (len(real_miss), real_miss[0][0]))
    report.append({"family": label, "source": source, "groups": len(groups),
                   "groups_testable": sum(1 for v in groups.values() if len(v) >= MIN_MEMBERS),
                   "found_on_card": len(hits), "real_misses": len(real_miss),
                   "pct_moved": 100 * moved / TOTAL,
                   "example_group": hits[0][0] if hits else None,
                   "example_device_path": hits[0][2] if hits else None})

# ---- negative control: the same families, members shuffled across blocks -------------------
print("\n=== negative control: same families, members taken from different blocks ===")
ctl_hits = 0
ctl_testable = 0
for pattern, label, source in FAMILIES:
    groups = build(pattern, label, list(hard))
    keys = sorted(g for g, v in groups.items() if len(v) >= MIN_MEMBERS)
    shuffled = {}
    for i, g in enumerate(keys):
        # take member j of group (i+j) mod len(keys): same roles, wrong blocks
        members, ok = [], True
        for j, n in enumerate(sorted(groups[g])):
            donor = groups[keys[(i + j + 1) % len(keys)]]
            role = n.rsplit(".", 2)[-2]
            pick = [m for m in donor if m.rsplit(".", 2)[-2] == role]
            if not pick:
                ok = False
                break
            members.append(pick[0])
        if ok and len(set(members)) == len(members) and len(members) >= MIN_MEMBERS:
            shuffled["SHUF " + g] = members
    h, m = probe(shuffled, set())
    ctl_hits += len(h)
    ctl_testable += len(shuffled)
    print("   %-14s %3d shuffled groups, %d matched a device tensor" % (label, len(shuffled), len(h)))
    if h:
        print("      CONTROL FIRED: %s -> %s" % (h[0][0], h[0][2]))
control_ok = ctl_hits == 0
print("   control: %s (%d of %d shuffled groups matched; must be 0)"
      % ("CLEAN" if control_ok else "FIRED", ctl_hits, ctl_testable))

rest = {n: q for n, q in hard.items() if n not in explained}
print("\n=== reach gap after the fusion probe ===")
print("   hard-uncovered                                 %6.4f %%  over %d names"
      % (100 * sum(hard.values()) / TOTAL, len(hard)))
print("   of that, fused on the card (not addressable)   %6.4f %%  over %d names"
      % (100 * total_moved / TOTAL, len(explained)))
print("   NOT ON THE CARD AT ALL                         %6.4f %%  over %d names"
      % (100 * sum(rest.values()) / TOTAL, len(rest)))
acc = {}
for n, q in rest.items():
    e = acc.setdefault(scope[n], [0, 0.0])
    e[0] += 1
    e[1] += q
print("\n   not on the card, by scope:")
for s, (c, q) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
    print("      %-56s %4d  %8.4f %%" % (s, c, 100 * q / TOTAL))

json.dump({"control_clean": control_ok, "control_hits": ctl_hits,
           "control_groups": ctl_testable, "min_members": MIN_MEMBERS,
           "families": report,
           "pct_hard_before": 100 * sum(hard.values()) / TOTAL,
           "pct_fused_on_card": 100 * total_moved / TOTAL,
           "pct_not_on_card": 100 * sum(rest.values()) / TOTAL,
           "not_on_card_by_scope": {s: {"n": c, "pct": 100 * q / TOTAL}
                                    for s, (c, q) in acc.items()},
           "not_on_card_detail": sorted([[n, q, 100 * q / TOTAL, scope[n]]
                                         for n, q in rest.items()], key=lambda r: -r[1])},
          open("perf/of3t_hostops/FUSED.json", "w"), indent=1)
print("\nwrote perf/of3t_hostops/FUSED.json")
