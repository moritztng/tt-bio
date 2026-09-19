"""One row per rung: what ran, on which card, at what clock, and what the design scored.

Written from `out/` rather than by hand. The designs and logs under it are gitignored, so this
file's output is the part of the evidence that survives in the repo, and every field in it is
read back from an artifact the run produced.

PXDesign writes the BINDER ALONE (`write_design_cifs`: a GLY backbone, N/CA/C/O per residue,
placed into the input target's frame). So there are two structural signals here and they answer
different questions:

  * `fit_rmsd`, from designs.json -- the residual of fitting the model's reconstruction of the
    CONDITIONED tokens onto the target coordinates it was conditioned on. This is the one that
    can see a torn conditioning path at a large target, because the binder's own geometry does
    not depend on the target size at all. A broken path lands in the tens of angstroms.
  * `check_structure --kind design` on each written binder -- backbone continuity, CA-CA band,
    radius of gyration, clashes.

A capacity rung that scored only the binder would pass at any target size, which is the same
trap `ladder.check_artifact` closes by reading `conditioned_tokens`.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a design at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}

DESIGN_CHAIN = "A"      # write.py labels the binder chain A; it is the only chain in the file


def window(logs: list[Path], lo: float, hi: float) -> list[list[str]]:
    """Every sample any sampler took inside [lo, hi], deduplicated by timestamp."""
    seen: dict[str, list[str]] = {}
    for p in logs:
        for line in p.read_text().splitlines():
            f = line.split()
            if f and lo <= int(f[0]) <= hi:
                seen[f[0]] = f
    return [seen[k] for k in sorted(seen)]


def score(cif: Path, out: Path) -> dict | None:
    """Backbone continuity and clashes on one designed binder, via the shipped checker."""
    subprocess.run(
        [sys.executable, str(ROOT / "perf" / "wh-correctness" / "check_structure.py"), str(cif),
         "--kind", "design", "--design-chain", DESIGN_CHAIN, "--json", str(out), "--quiet"],
        check=False, capture_output=True)
    return json.loads(out.read_text()) if out.exists() else None


def designs(d: Path, tres: int) -> list[dict]:
    """Every design this rung wrote, with its fit residual, md5 and structure report.

    The md5 is what makes the card-independence claim checkable: PXDesign takes --seed (42 by
    default), so two cards running the same rung must write the same bytes.
    """
    out_dir = d / "work" / f"out_pxdesign_{tres}"
    recs = {}
    j = out_dir / "designs.json"
    if j.is_file():
        for r in json.loads(j.read_text()):
            recs[Path(r["cif"]).name] = r
    rows = []
    for cif in sorted(out_dir.rglob("*.cif")):
        r = recs.get(cif.name, {})
        row = {"cif": cif.name, "md5": hashlib.md5(cif.read_bytes()).hexdigest(),
               "fit_rmsd": r.get("fit_rmsd"), "binder_residues": r.get("binder_residues"),
               "binder_atoms": r.get("binder_atoms"),
               "conditioned_tokens": r.get("conditioned_tokens")}
        rep = score(cif, d / f"struct_{cif.stem}.json")
        if rep:
            ch = rep["checks"]["chains"][0] if rep["checks"].get("chains") else {}
            row.update({"structure_fail": rep["fail"], "breaks": ch.get("breaks"),
                        "step_median": ch.get("step_median"), "in_band": ch.get("in_band_frac"),
                        "rg_ratio": ch.get("rg_ratio"),
                        "clashes": rep["checks"].get("clashes")})
        rows.append(row)
    return rows


def build(r: dict, d: Path, tres: int, binder: int, dev: int) -> dict:
    node = UMD_TO_NODE[dev]
    # The sampler appends, so a re-run rung directory holds both runs' samples and the idle gap
    # between them. The clock quoted has to be the clock DURING the run being reported, so the
    # window is cut from this row's own end stamp and wall time.
    end = calendar.timegm(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ"))
    lo, hi = end - r["wall_s"] - 5, end + 5
    clk = window(sorted(HERE.glob("out/*/aiclk.log")), lo, hi)
    host = window(sorted(HERE.glob("out/*/host.log")), lo, hi)
    row = {
        "target_residues": tres, "binder": binder, "tokens": tres + binder,
        "umd_device": dev, "node": node, "ts": r["ts"],
        "aiclk_median_mhz": (statistics.median(int(x[node + 1]) for x in clk) if clk else None),
        "aiclk_min_mhz": (min(int(x[node + 1]) for x in clk) if clk else None),
        "aiclk_samples": len(clk),
        "runtime_s": r["wall_s"], "rc": r["rc"], "verdict": r["verdict"],
        "mechanism": r.get("mechanism"),
        "host_load_median": (round(statistics.median(float(x[2]) for x in host), 1)
                             if host else None),
        "artifact": r.get("artifact"),
        "designs": designs(d, tres),
    }
    if r.get("diag"):
        row["diag"] = r["diag"][:6]
    return row


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("*_b*_dev*")):
        tres, rest = d.name.split("_b", 1)
        binder, dev = rest.split("_dev", 1)
        tres, binder, dev = int(tres), int(binder), int(dev)
        rj = d / "rung.jsonl"
        if not rj.exists():
            continue
        for r in (json.loads(x) for x in rj.read_text().splitlines() if x.strip()):
            rows.append(build(r, d, tres, binder, dev))
    rows.sort(key=lambda r: (r["tokens"], r["target_residues"], r["ts"]))
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        fits = [x["fit_rmsd"] for x in r["designs"] if x.get("fit_rmsd") is not None]
        fit = f"fit {min(fits):.3f}-{max(fits):.3f} A" if fits else "fit none"
        print(f'{r["tokens"]:5d} tok  {r["target_residues"]:4d}+{r["binder"]:<4d}  '
              f'dev {r["umd_device"]}  {r["aiclk_median_mhz"] or 0:.0f} MHz'
              f'/{r["aiclk_samples"]:3d}  {r["runtime_s"]:7.1f} s  {r["verdict"]:4s}  {fit}  '
              f'breaks {[x.get("breaks") for x in r["designs"]]}  '
              f'fail {[x.get("structure_fail") for x in r["designs"]]}  '
              f'md5 {[str(x["md5"])[:8] for x in r["designs"]]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
