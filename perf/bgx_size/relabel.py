#!/usr/bin/env python3
"""Relabel every rung by the token axis the Evoformer seam actually ran.

`tokens` in a rung.json is `_pad32(target_residues + binder)` off `targets.json`. That is the
arithmetic, and it is wrong wherever the complex BindCraft 2 builds is not
`target_residues + binder` -- which is most targets: hPDL1 contributes 129 tokens where the
file counts 115 residues, so `hPDL1 + 141` folds a 275-residue complex on a 288-token axis and
was filed under 256. Two rungs 32 apart reported an identical 4.0378 GB peak because they were
the same size.

Rungs taken after `f45224e02` stamp `evoformer_axis` themselves. Older ones do not, so they are
relabelled from `axis_census.sh`'s reading for the same (target, binder) pair -- sound because
`binder_lengths` is passed explicitly and the seed is fixed, so a pair has one reproducible
axis, and every census row came back with a single `axis_seen`.

A rung whose axis cannot be established is printed as UNKNOWN rather than given its arithmetic
label back.
"""
import json, glob, pathlib, sys

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load():
    rows = []
    for f in sorted(glob.glob(str(OUT / "*/rung.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        d["_name"] = pathlib.Path(f).parent.name
        rows.append(d)
    return rows


def census(rows):
    """(target, binder) -> true axis, from the census rungs and from any self-stamping rung."""
    m = {}
    for d in rows:
        ax = d.get("evoformer_axis")
        if not ax:
            continue
        key = (d.get("target"), d.get("binder"))
        seen = d.get("axis_seen") or [ax]
        if key in m and m[key] != ax:
            print(f"WARNING {key} reports {m[key]} and {ax}", file=sys.stderr)
        if len(seen) > 1:
            print(f"WARNING {key} saw more than one axis: {seen}", file=sys.stderr)
        m[key] = ax
    return m


def main():
    rows = load()
    m = census(rows)
    print(f"{'rung':22s} {'label':>6s} {'TRUE':>6s} {'peak GB':>8s} {'s/round':>8s} "
          f"{'load1':>6s} {'fused fwd':>12s} outcome")
    out = []
    for d in rows:
        key = (d.get("target"), d.get("binder"))
        true = d.get("evoformer_axis") or m.get(key)
        pr = d.get("per_round") or []
        secs = [r["seconds"] for r in pr[1:]]
        med = round(sorted(secs)[len(secs) // 2], 2) if secs else None
        ld = round(sum(r.get("load1") or 0 for r in pr[1:]) / len(secs), 1) if secs else None
        fh = (d.get("lever_stats") or {}).get("triatt_fused_hifi") or {}
        fused = (f"{fh.get('served')}/{fh.get('declined')}"
                 if fh.get("served") is not None else "-")
        err = str(d.get("error") or "")
        outcome = ("REFUSES" if "Out of Memory" in err
                   else "completes" if "StopAfterRounds" in err or not err else err[:28])
        out.append((true or 0, d["_name"], d.get("tokens"), true, d.get("resident_peak_gb"),
                    med, ld, fused, outcome))
    for _, name, label, true, peak, med, ld, fused, outcome in sorted(out):
        flag = ("  <- axis not established yet" if not true
                else "" if true == label else "  <- RELABELLED")
        print(f"{name:22s} {str(label):>6s} {str(true or 'UNKNOWN'):>6s} {str(peak):>8s} "
              f"{str(med):>8s} {str(ld):>6s} {fused:>12s} {outcome}{flag}")


if __name__ == "__main__":
    main()
