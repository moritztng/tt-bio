#!/usr/bin/env python3
"""of3t-trajfull deliverable 0: what each of the 180 unscored reference tensors IS.

Host-only. Reads the device census `trajwide.py --census-out` wrote, the reference bank's
own name list and the previous best coupled arm's, and classifies every reference tensor the
old dump missed into the three cases the brief fixes before any of them is scored:

  (a) present in our parameter set and named, never written out;
  (b) present but FUSED, so one device weight stands for several reference ones;
  (c) genuinely absent from the device parameter set.

(c) is the outcome that would make 89.2105 unreachable at this boundary. It is reported as
itself if it happens; it is not engineered around.

An npz is a zip, so the banks are read with stdlib `zipfile` and this runs under qb2's system
python, which has no numpy.
"""
import argparse
import collections
import json
import os
import re
import zipfile

RUNS = "/home/ttuser/of3t_runs/trajwide"
MODEL_SQ_NORM = 10.279642678524981          # `of3t-wholemodel`, grads_f64_043.pt


def npz_names(path):
    return sorted(n[:-4] if n.endswith(".npy") else n
                  for n in zipfile.ZipFile(path).namelist())


def family(n):
    k = re.sub(r"\.\d+\.", ".N.", n)
    return k[:k.index(".N.") + 3] if ".N." in k else k.rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-census", default="perf/of3t_trajfull/DEVICE_CENSUS.json")
    ap.add_argument("--theirs", default=os.path.join(RUNS, "w/theirs/k01.npz"))
    ap.add_argument("--before", default=os.path.join(RUNS, "w/refatom/k01.npz"),
                    help="the previous best coupled arm's dump, the 581")
    ap.add_argument("--families", default="perf/of3t_refatom/traj_refatom.json",
                    help="that arm's scope block, for each missing family's share")
    ap.add_argument("--out", default="perf/of3t_trajfull/CENSUS.json")
    a = ap.parse_args()

    ref = npz_names(a.theirs)
    before = npz_names(a.before)
    dev = json.load(open(a.device_census))
    now = set(dev["w0_dump_keys"])
    missing = sorted(set(ref) - set(before))

    # what the resolver said about each slot, keyed by the reference name it claims
    by_name, unresolved_slots = {}, []
    for r in (dev.get("namemap") or {}).get("rows", []):
        if not r.get("resolved"):
            unresolved_slots.append(r)
            continue
        for n in r.get("names", []):
            by_name.setdefault(n, []).append(r)

    shares = {}
    if os.path.exists(a.families):
        for k, v in (json.load(open(a.families))["scope"]
                     .get("not_covered_families") or {}).items():
            shares[k] = v

    cases = collections.OrderedDict()
    counts = collections.Counter()
    for n in missing:
        rows = by_name.get(n, [])
        if not rows:
            case, why = "c_absent", "no device slot reproduces this tensor"
        elif rows[0].get("kind") in ("qkv_w", "qkv_b"):
            case = "b_fused"
            why = "%s carries it; split round-trips bit-exactly, pad max |x| = %s" % (
                rows[0]["path"], rows[0].get("pad_max_abs"))
        else:
            case = "a_present_not_dumped"
            why = "%s holds it under key %s, transpose=%s, match=%s" % (
                rows[0]["path"], rows[0]["key"], rows[0].get("transpose"),
                (rows[0].get("dtype_match") or [None])[0])
        counts[case] += 1
        cases[n] = {"case": case, "why": why, "in_new_dump": n in now}

    fam = collections.OrderedDict()
    for n in missing:
        e = fam.setdefault(family(n), collections.Counter())
        e[cases[n]["case"]] += 1
        e["total"] += 1
    fam_out = collections.OrderedDict()
    for k in sorted(fam, key=lambda k: -(shares.get(k, {}).get("sq", 0.0))):
        fam_out[k] = dict(fam[k])
        if k in shares:
            fam_out[k]["pct_of_model_sq_grad_norm"] = shares[k]["pct_of_model_sq_grad_norm"]

    added = sorted(now - set(before))
    verdict = ("PASS" if (counts["c_absent"] == 0
                          and not unresolved_slots
                          and set(missing) <= now
                          and not (now - set(ref)))
               else "FAIL")
    res = {
        "what": "of3t-trajfull deliverable 0: the three-way census of the 180 reference "
                "tensors the coupled trajectory did not name, taken BEFORE the arm.",
        "verdict": verdict,
        "reference_tensors": len(ref),
        "dumped_before": len(before),
        "missing_before": len(missing),
        "dumped_now_at_w0": len(now),
        "added_by_the_namemap": len(added),
        "added_are_exactly_the_missing": sorted(added) == missing,
        "nothing_added_outside_the_reference": sorted(now - set(ref)) == [],
        "cases": dict(counts),
        "by_family": fam_out,
        "unresolved_slots": unresolved_slots,
        "device": {k: dev[k] for k in ("walked", "slots", "named_by_fingerprint",
                                       "n_w0_dump_keys") if k in dev},
        "namemap_summary": {k: (dev.get("namemap") or {}).get(k)
                            for k in ("slots_unnamed", "slots_resolved",
                                      "reference_names_added")},
        "emit_evidence": dev.get("emit_evidence"),
        "model_sq_norm": MODEL_SQ_NORM,
        "per_tensor": cases,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_tensor"}, indent=1)[:3000])
    print("wrote", a.out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
