#!/usr/bin/env python3
"""PHASE 0b: prove the N=512 pool nests the N=256 pool, on the targets already at 8 chunks.

The N=512 rung is built by adding chunks 4-7 to the existing chunks 0-3, so three things
must hold for every (model, target) whose 8 chunks are all on disk:

  1. pool(512) has exactly 512 (selector, dockq) pairs and pool(256) has 256
  2. pool(256) is a sub-multiset of pool(512)   -- the ladder is seed-nested
  3. oracle(512) == max dockq over ALL 512 and is >= oracle(256)
  4. the user pick is dockq at argmax(selector) over ALL 512, not per-chunk and not over
     the added 256 only

Reads the same dirs and the same join as scripts/abag_xm_deepn_analysis.py, independently
reimplemented so a shared bug cannot hide. CPU only, no chips, no writes.

usage: verify_n512_nesting.py [--base ~/abag_xm/deepn/galaxy]
"""
import argparse, collections, json, sys
from pathlib import Path

MODELS = {"opendde-abag": ("opendde", "confidence_score"),
          "protenix-v2": ("protenix", "confidence_score"),
          "boltz2": ("boltz2", "confidence_score"),
          "esmfold2": ("esmfold2", "plddt")}
# Mirrors abag_xm_deepn_analysis.py's GALAXY_EXCLUDE + P32_EXTENSION. The join is
# reimplemented here on purpose so a shared bug cannot hide, but the panel must be the
# same one: without these, this script's per-model count is a few cells above the
# analysis's n_targets at 512 and the cross-check reads as a discrepancy that is not one.
# No exclusions. 9i3p, 9ivj and 9q7y were "documented WH DRAM exclusions" and are folded and
# harvested on all four models; 9j4c is still open and now reports as not-yet-8/8, which is the
# honest state rather than a silent drop. 9sbb was carried here with no recorded reason at all.
# A gate that filters its own denominator cannot see a missing cell, which is how this panel
# reported 160/160 while it was short (workstream abag-xm-panel-complete-164).
EXCLUDE: dict[str, set[str]] = {}


def fold_pairs(out_dir: Path, prefix: str, target: str, sel_key: str):
    """One chunk dir -> [(selector, dockq)] joined by rank, or None on any gap."""
    rj = out_dir / f"{prefix}_results_{target}" / "results.json"
    lj = out_dir / "labels.json"
    try:
        runs = json.loads(rj.read_text())[0].get("all_runs", [])
        labs = json.loads(lj.read_text()).get("samples", [])
    except Exception:
        return None
    conf = {int(r["rank"]): float(r[sel_key]) for r in runs if r.get(sel_key) is not None}
    dockq = {}
    for s in labs:
        d = s.get("dockq")
        if isinstance(d, dict) and d.get("dockq") is not None:
            dockq[int(s["rank"])] = float(d["dockq"])
    ranks = sorted(set(conf) & set(dockq))
    return [(conf[r], dockq[r]) for r in ranks] or None


def pool(base: Path, model: str, target: str, rung: int, chunks):
    prefix, sel_key = MODELS[model]
    out = []
    for c in chunks:
        d = base / prefix / f"{target}_n{rung}_c{c}"
        p = fold_pairs(d, prefix, target, sel_key) if d.is_dir() else None
        if p is None:
            return None
        out.extend(p)
    return out


def oracle(p):
    return max(d for _s, d in p)


def user(p):
    return max(p, key=lambda sd: sd[0])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "abag_xm" / "deepn" / "galaxy"))
    ap.add_argument("--yaml-dir", default=None,
                    help="directory of <target>.yaml; sets the denominator so an absent cell "
                         "is reported as incomplete rather than being invisible")
    a = ap.parse_args()
    base = Path(a.base)
    yaml_targets = set()
    if a.yaml_dir:
        yaml_targets = {p.name[:-5] for p in Path(a.yaml_dir).glob("*.yaml")}

    checked = failures = 0
    skipped = []
    per_model = {}
    for model, (prefix, _sel) in MODELS.items():
        mdir = base / prefix
        if not mdir.is_dir():
            continue
        per_model[model] = [0, 0, 0]      # checked, failed, skipped-not-8/8
        # Denominator from the yaml set, NOT from the n512 dirs that happen to exist. A cell
        # with no dir at all used to be absent from `targets` entirely, so it was not even
        # counted as skipped -- an unfolded cell and a nonexistent one printed the same.
        if yaml_targets:
            targets = sorted(yaml_targets)
        else:
            targets = sorted({d.name.split("_n512_c")[0] for d in mdir.glob("*_n512_c*")})
        targets = [t for t in targets if t not in EXCLUDE.get(model, ())]
        for t in targets:
            p512 = pool(base, model, t, 512, range(8))
            p256 = pool(base, model, t, 256, range(4))
            if p512 is None or p256 is None:
                per_model[model][2] += 1
                which = ("512" if p512 is None else "") + ("+256" if p256 is None else "")
                skipped.append(f"{model}/{t} (pool {which} incomplete)")
                continue          # not yet at 8 chunks; the completeness gate drops it
            checked += 1
            per_model[model][0] += 1
            errs, notes = [], []
            if len(p512) != 512:
                errs.append(f"pool(512) has {len(p512)} pairs, want 512")
            if len(p256) != 256:
                errs.append(f"pool(256) has {len(p256)} pairs, want 256")
            c512, c256 = collections.Counter(p512), collections.Counter(p256)
            missing = {k: n - c512[k] for k, n in c256.items() if c512[k] < n}
            if missing:
                errs.append(f"pool(256) is not a sub-multiset of pool(512): "
                            f"{len(missing)} pairs missing, e.g. {list(missing)[:2]}")
            o512, o256 = oracle(p512), oracle(p256)
            if o512 < o256 - 1e-12:
                errs.append(f"oracle(512) {o512:.6f} < oracle(256) {o256:.6f}")
            if abs(o512 - max(d for _s, d in p512)) > 1e-12:
                errs.append("oracle(512) is not the max over all 512")
            u512 = user(p512)
            best_sel = max(s for s, _d in p512)
            tied = [d for s, d in p512 if s == best_sel]
            # A dict keyed by selector would keep the LAST tied entry while user()
            # keeps the first, so on a tie at the max this check used to fail a
            # correct pool. The pick only has to be one of the tied entries.
            if not any(abs(d - u512) <= 1e-12 for d in tied):
                errs.append("user pick is not dockq at argmax(selector) over all 512")
            if len(tied) > 1:
                notes.append(f"selector ties at its max: {len(tied)} samples share "
                             f"{best_sel:.6f}, dockq {min(tied):.4f}..{max(tied):.4f}; "
                             f"the pick is not uniquely defined")
            u_added = user(p512[256:]) if len(p512) > 256 else None
            tag = "" if u_added is None or abs(u_added - u512) > 1e-12 else "  (note: pick also equals the added-256 pick)"
            if errs:
                failures += 1
                per_model[model][1] += 1
                print(f"FAIL {model}/{t}")
                for e in errs:
                    print(f"       {e}")
            else:
                print(f"ok   {model}/{t}  oracle 256->512 {o256:.4f}->{o512:.4f}  "
                      f"user {user(p256):.4f}->{u512:.4f}{tag}")
            for n in notes:
                print(f"       note: {n}")
    print()
    for model, (c, f, s) in sorted(per_model.items()):
        print(f"  {model:<14} checked {c:3d}  failed {f:3d}  not-yet-8/8 {s:3d}")
    print(f"\nchecked {checked} (model,target) cells at 8/8 chunks, {failures} failed")
    if skipped:
        # Name them. A gate that reports "17 skipped" and nothing else cannot be acted on, and
        # a skipped cell reads exactly like a cell that was never meant to exist.
        print(f"\nnot checked ({len(skipped)}), by name:")
        for x in sorted(skipped):
            print(f"  {x}")
    print("per-model 'checked' is the panel the analysis should report as n_targets at 512; "
          "a shortfall means labels are still missing, not that the pool is wrong")
    return 1 if failures or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
