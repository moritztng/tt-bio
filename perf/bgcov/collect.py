"""One row per rung: what ran, on which card, at what clock, and what the design scored.

Written from `out/` and `work/` rather than by hand. Both are gitignored (CIFs, logs,
checkpointed configs), so this file's output is the part of the evidence that survives in the
repo, and every field in it is read back from an artifact the run produced.

BoltzGen writes the COMPLEX into `intermediate_designs/<id>.cif` -- the designed chain next to
the target -- while `out_dir/<id>.cif` is a copy of the INPUT. Scoring the latter scores the
fixture, so the scorer is pointed at the designed complex by name.
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

BINDER_CHAIN = "Z"


def design_cif(rung_dir: Path, tres: int) -> Path | None:
    """The complex BoltzGen designed, not the copy of the input it also writes.

    `out_dir/<id>.cif` is the INPUT copied back, atom for atom; the designed chain lands next
    to the target in `intermediate_designs/<id>.cif`. Scoring the former scores the fixture and
    calls every rung clean.

    The shared `perf/bgcov/work` is the fallback for the rungs launched before rung.sh started
    giving each one its own directory; a rung with its own copy always wins.
    """
    for base in (rung_dir / "work", HERE / "work"):
        d = base / f"out_boltzgen_{tres}" / "intermediate_designs"
        cifs = sorted(d.glob("*.cif")) if d.is_dir() else []
        if cifs:
            return cifs[0]
    return None


def input_cif(rung_dir: Path, tres: int) -> Path | None:
    """The target crop the rung conditioned on -- the control for the clash count."""
    for base in (rung_dir / "work", HERE / "work"):
        p = base / f"bgt{tres}.cif"
        if p.exists():
            return p
    return None


def attribute(cif: Path, control: Path, out: Path) -> dict | None:
    """Clashes split by chain pair, with the same count over the input crop beside them."""
    subprocess.run(
        [sys.executable, str(HERE / "clash_attrib.py"), str(cif),
         "--control", str(control), "--json", str(out)],
        check=False, capture_output=True)
    return json.loads(out.read_text()) if out.exists() else None


def score(cif: Path, out: Path) -> dict | None:
    """Backbone continuity and clashes on the designed complex, via the shipped checker."""
    checker = ROOT / "perf" / "wh-correctness" / "check_structure.py"
    subprocess.run(
        [sys.executable, str(checker), str(cif), "--kind", "design",
         "--design-chain", BINDER_CHAIN, "--json", str(out), "--quiet"],
        check=False, capture_output=True)
    return json.loads(out.read_text()) if out.exists() else None


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("*_dev*")):
        tres, dev = d.name.rsplit("_dev", 1)
        # A rung dir may carry a trailing tree tag (`1831_dev3_c1ffb32d`) when the same rung is
        # replayed against another commit; the card is the first field after `_dev`.
        dev, _, tree = dev.partition("_")
        tres, dev = int(tres), int(dev)
        rj = d / "rung.jsonl"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text().splitlines()[-1])
        node = UMD_TO_NODE[dev]
        # The sampler appends, so a rung dir that has been re-run holds both runs' samples and
        # the idle 800 MHz gap between them. The clock quoted has to be the clock DURING the
        # run being reported, so the window is cut from the row's own end stamp and wall time.
        end = calendar.timegm(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ"))
        lo, hi = end - r["wall_s"] - 5, end + 5
        clk = [x.split() for x in (d / "aiclk.log").read_text().splitlines()
               if x.strip() and lo <= int(x.split()[0]) <= hi]
        host = [x.split() for x in (d / "host.log").read_text().splitlines()
                if x.strip() and lo <= int(x.split()[0]) <= hi]
        # The binder is read back off the artifact rather than assumed: a rung can reach the
        # same token count with a long target and a short design or the other way round, and
        # those are different inputs to the same bar.
        art = r.get("artifact") or {}
        binder = (art["residues"] - art["target_residues"]
                  if "residues" in art and "target_residues" in art else 80)
        row = {
            "target_residues": tres, "binder": binder, "tokens": tres + binder,
            "target_atoms": r.get("target_atoms"),
            "umd_device": dev, "node": node, "tree": tree or "HEAD",
            "aiclk_median_mhz": statistics.median(int(x[node + 1]) for x in clk),
            "aiclk_min_mhz": min(int(x[node + 1]) for x in clk),
            "runtime_s": r["wall_s"], "rc": r["rc"], "verdict": r["verdict"],
            "mechanism": r.get("mechanism"),
            "host_load_median": round(statistics.median(float(x[2]) for x in host), 1),
            "artifact": r.get("artifact"),
        }
        cif = design_cif(d, tres)
        if cif:
            row["design_cif_md5"] = hashlib.md5(cif.read_bytes()).hexdigest()
            rep = score(cif, d / "struct.json")
            if rep:
                row["structure_fail"] = rep["fail"]
                row["chains"] = [
                    {"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                     "step_median": c["step_median"], "in_band": c["in_band_frac"],
                     "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]]
                row["clashes"] = rep["checks"].get("clashes")
            ctrl = input_cif(d, tres)
            if ctrl:
                row["input_cif_md5"] = hashlib.md5(ctrl.read_bytes()).hexdigest()
                row["clash_attribution"] = attribute(cif, ctrl, d / "clash_attrib.json")
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["tokens"], "tok dev", r["umd_device"], r["aiclk_median_mhz"], "MHz",
              r["runtime_s"], "s", r["verdict"], str(r.get("design_cif_md5"))[:8],
              "breaks", [c["breaks"] for c in r.get("chains", [])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
