#!/usr/bin/env python3
"""cdk2x2_298 control: all-atom and CA Kabsch RMSD of each flag arm against the shipped default.

`cdk2x2_512` cannot score a non-bit-exact change (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`): it is CDK2 fused to a truncated copy
of itself and its unconstrained inter-domain hinge saturates RMSD for any change, whatever the
cause. `cdk2x2_298` is the same family with one real domain and no hinge, which is why the
thresholds in `state/answered/4649-decision.md` are stated against it: <=0.35 A pass,
0.35-0.60 A hold, >0.60 A reject.

The parser and the superposition come from `perf/other512/cif_rmsd.py` so the numbers stay
comparable across this lineage -- all atoms, equal weights. CA RMSD is the same superposition
restricted to the CA subset, which is the column the 4649 thresholds name.

    score298.py <cifdir> [--ref base_0] [--out results.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms          # noqa: E402

AA_PASS, AA_HOLD = 0.35, 0.60


def verdict(r: float) -> str:
    return "PASS" if r <= AA_PASS else ("HOLD" if r <= AA_HOLD else "REJECT")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cifdir", type=Path)
    ap.add_argument("--ref", default="base_0")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    data = {}
    for d in sorted(a.cifdir.iterdir()):
        if not d.is_dir():
            continue
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        keys, xyz = read_atoms(cifs[0])
        tag = d.name.split("_", 1)[1]          # 298_base_0 -> base_0
        # The prefix carries the FIXTURE, and stripping it makes 298_base_0 and 512_base_0 the
        # same key. A cifdir holding both sizes -- which `ab_arms.py --control` produces, since it
        # keeps 512_* from the timed phase and 298_* from the control phase in one directory --
        # then silently scores the 512 folds under the 298 thresholds. cdk2x2_512 is chimeric and
        # saturates for any change (memory cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-
        # parity), so the table comes out plausible and wrong: measured 2026-09-12, a change that
        # is 0.218 A on cdk2x2_298 read 0.505 A and flipped PASS to HOLD. Refuse loudly instead.
        if tag in data:
            raise SystemExit(
                f"two directories map to the tag {tag!r}: {data[tag]['dir']} and {d.name}. The "
                f"fixture prefix is what differs, so this cifdir holds more than one fixture and "
                f"the thresholds below apply to exactly one. Point this at a single fixture's "
                f"folds.")
        ca = [i for i, k in enumerate(keys) if "CA" in k]
        data[tag] = {"keys": keys, "xyz": xyz, "ca": ca, "cif": cifs[0].name, "dir": d.name}
        print(f"  {tag:10s} {len(xyz):6d} atoms  {len(ca):5d} CA  {cifs[0].name}")
    if len(data) < 2:
        raise SystemExit("need at least two folds on disk")

    ref_keys = data[sorted(data)[0]]["keys"]
    for tag, v in data.items():
        if v["keys"] != ref_keys:
            raise SystemExit(f"atom identity differs in {tag} -- cannot compare by order")
    print(f"\n  atom identity identical across all {len(data)} folds ({len(ref_keys)} atoms), "
          f"so every comparison is atom-for-atom\n")

    def pair(x, y):
        ax, ay = data[x], data[y]
        return (kabsch_rmsd(ax["xyz"], ay["xyz"]),
                kabsch_rmsd(ax["xyz"][ax["ca"]], ay["xyz"][ay["ca"]]))

    out = {"cifdir": str(a.cifdir), "ref": a.ref,
           "thresholds_A": {"pass": AA_PASS, "hold": AA_HOLD,
                            "source": "state/answered/4649-decision.md"},
           "n_atoms": len(ref_keys), "n_ca": len(data[a.ref]["ca"]),
           "pairwise": [], "vs_ref": {}}

    print("  every pair, all-atom / CA Kabsch RMSD in A:")
    for x, y in itertools.combinations(sorted(data), 2):
        aa, ca = pair(x, y)
        kind = "A/A" if x.split("_")[0] == y.split("_")[0] else "A/B"
        out["pairwise"].append({"kind": kind, "a": x, "b": y,
                                "all_atom_A": round(aa, 6), "ca_A": round(ca, 6)})
        print(f"    {kind}  {x:10s} vs {y:10s}  {aa:10.6f} / {ca:10.6f}")

    floor = [p for p in out["pairwise"] if p["kind"] == "A/A"]
    if floor:
        m = max(p["all_atom_A"] for p in floor)
        out["AA_structural_floor_A"] = m
        print(f"\n  A/A structural floor: {m:.6f} A over {len(floor)} pair(s)"
              f"{'  -- NOT ZERO, the measurement is void' if m > 1e-9 else '  (exact, as required)'}")

    print(f"\n  vs `{a.ref}`, against the 4649 bar (<=0.35 pass / 0.35-0.60 hold / >0.60 reject):")
    for tag in sorted(data):
        if tag == a.ref:
            continue
        aa, ca = pair(a.ref, tag)
        out["vs_ref"][tag] = {"all_atom_A": round(aa, 6), "ca_A": round(ca, 6),
                              "verdict_all_atom": verdict(aa), "verdict_ca": verdict(ca)}
        print(f"    {tag:10s} all-atom {aa:9.6f} A [{verdict(aa):6s}]   "
              f"CA {ca:9.6f} A [{verdict(ca):6s}]")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print("\n  wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
