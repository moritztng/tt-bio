#!/usr/bin/env python3
"""Score the seed-floor folds and print the floor tables.

    python3 perf/mgx_seed_floor/collect.py perf/mgx_seed_floor/out [--json]

3abq_1536: every seed against the 3ABQ crystal (score.py's crystal-resolved line), and the
spread across seeds. cdk2x2_1280 has no crystal, so it is the device scored against itself:
seed 1 against seed 0, whole chain and per 298-residue CDK2 copy (segments.py's windows).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "perf/mgx/ref"), str(ROOT / "perf/mgx/acc"), str(ROOT / "tests")]
import score  # noqa: E402
import segments  # noqa: E402


def rows(out: Path) -> dict:
    """Last results row per tag (a retried tag ends in '+')."""
    got = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        got[r["tag"].rstrip("+")] = r
    return got


def cif(out: Path, tag: str) -> str | None:
    hits = sorted((out / "runs").glob(f"{tag}*/**/structures/*.cif"))
    return str(hits[-1]) if hits else None


def per_copy(a: str, b: str, trim: int) -> list:
    x, y = segments.ca(a), segments.ca(b)
    n, w = min(len(x), len(y)), 298
    return [(round(segments.kabsch(x[i:i + w - trim], y[i:i + w - trim]), 3),
             round(segments.lddt(x[i:i + w - trim], y[i:i + w - trim]), 4))
            for i in range(0, n - w + 1, w)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    res = rows(out)
    table = {"3abq_1536": {}, "cdk2x2_1280": {}}
    for tag, r in sorted(res.items()):
        fx, model, seed = tag.split("-", 1)[0], tag.split("-", 1)[1].rsplit("-s", 1)[0], int(tag.rsplit("-s", 1)[1])
        fixture = "3abq_1536" if fx == "3abq" else "cdk2x2_1280"
        path = cif(out, tag)
        e = {"verdict": r.get("verdict"), "runtime_s": r.get("runtime_s") or r.get("wall_s"),
             "card": r.get("card"), "commit": r.get("commit"), "aiclk_mhz": r.get("aiclk_mhz"), "cif": path}
        if path and fixture == "3abq_1536":
            e["crystal"] = score.score(model, fixture, path).get("crystal")
        table[fixture].setdefault(model, {})[seed] = e
    for model, seeds in table["cdk2x2_1280"].items():
        if all(seeds.get(s, {}).get("cif") for s in (0, 1)):
            a, b = seeds[1]["cif"], seeds[0]["cif"]
            seeds["s1_vs_s0"] = {"whole": score.compare(a, b), "per_copy": per_copy(a, b, 0),
                                 "per_copy_trim8": per_copy(a, b, 8)}
    if args.json:
        print(json.dumps(table, indent=1, default=str))
        return 0
    print("3abq_1536 vs crystal: CA-RMSD A / lDDT per seed, spread = max - min")
    for model, seeds in sorted(table["3abq_1536"].items()):
        cells, rm, ld = [], [], []
        for s in sorted(seeds):
            c = seeds[s].get("crystal") or {}
            clk = (seeds[s].get("aiclk_mhz") or {}).get("median")
            if "ca_rmsd_A" in c:
                rm.append(c["ca_rmsd_A"]); ld.append(c["lddt_ca"])
                cells.append(f"s{s} {c['ca_rmsd_A']:.3f}/{c['lddt_ca']:.4f} ({seeds[s]['runtime_s']} s, card {seeds[s]['card']}, AICLK med {clk})")
            else:
                cells.append(f"s{s} {seeds[s]['verdict']}")
        spread = f"spread {max(rm) - min(rm):.3f} A / {max(ld) - min(ld):.4f}" if len(rm) > 1 else ""
        print(f"  {model}: " + " | ".join(cells) + f"  {spread}")
    print("cdk2x2_1280 device seed 1 vs seed 0: whole CA-RMSD/lDDT; per 298-aa copy (trim 8)")
    for model, seeds in sorted(table["cdk2x2_1280"].items()):
        v = seeds.get("s1_vs_s0")
        st = " ".join(f"s{s} {seeds[s]['verdict']} {seeds[s]['runtime_s']} s card {seeds[s]['card']}"
                      f" AICLK med {(seeds[s].get('aiclk_mhz') or {}).get('median')}"
                      for s in sorted(k for k in seeds if isinstance(k, int)))
        if v:
            pc = ", ".join(f"{r:.2f}/{l:.3f}" for r, l in v["per_copy"])
            pt = ", ".join(f"{r:.2f}/{l:.3f}" for r, l in v["per_copy_trim8"])
            print(f"  {model}: whole {v['whole'].get('ca_rmsd_A')}/{v['whole'].get('lddt_ca')}; copies {pc}; trim8 {pt}  [{st}]")
        else:
            print(f"  {model}: [{st}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
