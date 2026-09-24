"""Grade the 512-aa folds per segment instead of over the whole chain.

cdk2x2_512 is one chain: CDK2 (1-298) followed by CDK2 1-214 again (299-512). The whole-chain
RMSD mostly measures where the second copy lands relative to the first, which no structure
constrains. Each copy on its own is CDK2, so each is graded against PDB 1HCL, and the on-vs-off
deviation and seed floor are taken per copy too.

    python3 perf/bcx_oplin/grade512.py perf/bcx_oplin/folds/<model>_512 ...
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fold_ab import ca, rmsd  # noqa: E402

SEGMENTS = {"copy1": (1, 298, 0), "copy2": (299, 512, 298)}   # first, last, offset into 1HCL


def segment(c, first, last, off):
    return {n - off: xyz for (_ch, n), xyz in c.items() if first <= n <= last}


def grade(d):
    truth = {n: xyz for (_ch, n), xyz in ca(HERE / "1hcl.cif", by_resnum=True).items()}
    arms = {f.stem: ca(f, by_resnum=True) for f in sorted(d.glob("s[0-9]_o*.cif"))
            if "_t" not in f.stem}
    out = {}
    for name, (first, last, off) in SEGMENTS.items():
        seg = {k: segment(v, first, last, off) for k, v in arms.items()}
        seeds = sorted({k.split("_")[0] for k in seg})
        out[name] = dict(
            residues=[first, last],
            deviation_A={s: round(rmsd(seg[f"{s}_on"], seg[f"{s}_off"]), 4) for s in seeds},
            seed_floor_off_A={s: round(rmsd(seg[f"{s}_off"], seg["s0_off"]), 4) for s in seeds[1:]},
            seed_floor_on_A={s: round(rmsd(seg[f"{s}_on"], seg["s0_on"]), 4) for s in seeds[1:]},
            vs_1hcl_off=[round(rmsd(seg[f"{s}_off"], truth), 4) for s in seeds],
            vs_1hcl_on=[round(rmsd(seg[f"{s}_on"], truth), 4) for s in seeds],
        )
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        g = grade(Path(p))
        (Path(p) / "segments.json").write_text(json.dumps(g, indent=1) + "\n")
        print(p, json.dumps(g))
