#!/usr/bin/env python3
"""Do the two arms feed BindCraft 2's acceptance the same numbers?

The accept/reject decision is a filter set over per-prediction metrics --
`i_pTM >= 0.7, pTM >= 0.55, i_pAE <= 0.35, Unbound_Binder_pLDDT >= 0.7,
Interface_Residues >= 7, Binder_RMSD <= 3.5, Backbone_Clashes <= 0` -- computed from the
same quantities the design loss is built from. BindCraft 2 stamps every one of them into the
`_bindcraft.*` block of each frame it writes, so the filters' inputs can be read arm against
arm without running a trajectory to a decision.

Round 1 is the paired one: both arms hold the same sequence there, so a difference is the
arithmetic. Later rounds hold different sequences and are shown for the trend only.

    python3 perf/bcx_p10_stack/score_metrics.py perf/bcx_p10_stack/out
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_struct import frames                                        # noqa: E402

KEYS = ("ptm", "iptm", "binder_pae", "interface_pae", "binder_contacts",
        "interface_contacts", "binder_helicity", "compactness",
        "iptm_loss", "plddt_loss")


def metrics(path):
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.match(r"_bindcraft\.(\S+)\s+(\S+)", line.strip())
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="acc_off_a")
    ap.add_argument("--floor", default="acc_off_b")
    ap.add_argument("--lever", default="acc_on")
    a = ap.parse_args()

    base, floor, lever = (frames(a.root, t) for t in (a.base, a.floor, a.lever))
    for k in sorted(set(base) & set(lever)):
        rnd = k[1]
        b, l = metrics(base[k]), metrics(lever[k])
        f = metrics(floor[k]) if k in floor else {}
        tag = "PAIRED, same sequence" if rnd == min(r for _, r, _ in base) else "unpaired"
        print(f"\nround {rnd}  ({tag})")
        print(f"  {'metric':<20}{'all-off':>10}{'all-on':>10}{'delta':>10}{'floor':>10}")
        for key in KEYS:
            if key not in b or key not in l:
                continue
            fl = f.get(key)
            print(f"  {key:<20}{b[key]:>10.4f}{l[key]:>10.4f}{l[key]-b[key]:>+10.4f}"
                  + (f"{fl - b[key]:>+10.4f}" if fl is not None else f"{'--':>10}"))


if __name__ == "__main__":
    main()
