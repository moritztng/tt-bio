"""One af2ig rung as a row: both axes, the clock it was measured at, and the structure.

The clock window is the FOLD, not the process: `rung.sh` starts its sampler before the CLI
loads 373 MB of parameters and opens the card, and a p150a idles at 800 MHz, so averaging the
whole series reports a clock no trunk pass ever ran at. The window here opens at the first
`trunk 1/N` line the CLI printed and closes at the last progress line, both read off rung.log.

    TT_VISIBLE_DEVICES= python3 perf/af2igcov/collect.py --tag m1536 --tokens 1536 \
        --target-residues 1008 --binder-residues 528
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import statistics as st
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "perf" / "af2igcov" / "out"
CHECK = ROOT / "perf" / "wh-correctness" / "check_structure.py"
CA_BAND = (3.60, 4.10)
CA_BREAK = 5.0


def _log_window(log: pathlib.Path, day: dt.date) -> tuple[float, float, list[tuple[str, float]]]:
    """(first trunk epoch, last progress epoch, [(label, epoch)]) from the CLI's own stamps."""
    marks = []
    for line in log.read_text(errors="replace").splitlines():
        head = line[:8]
        if len(head) == 8 and head[2] == ":" and head[5] == ":" and head[:2].isdigit():
            try:
                t = dt.datetime.strptime(head, "%H:%M:%S").time()
            except ValueError:
                continue
            epoch = dt.datetime.combine(day, t).timestamp()
            marks.append((line[8:].strip(), epoch))
    trunk = [m for m in marks if "trunk" in m[0]]
    if not trunk:
        return (marks[0][1] if marks else 0.0), (marks[-1][1] if marks else 0.0), marks
    return trunk[0][1], marks[-1][1], marks


def _clock(path: pathlib.Path, lo: float, hi: float) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5 or f[1] == "NA":
            continue
        rows.append((float(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4])))
    infold = [r for r in rows if lo <= r[0] <= hi] or rows
    clk = [r[1] for r in infold]
    return {"aiclk_median": st.median(clk), "aiclk_min": min(clk), "aiclk_max": max(clk),
            "aiclk_samples": len(clk), "aiclk_samples_total": len(rows),
            "power_median_w": st.median(r[2] for r in infold),
            "power_max_w": max(r[2] for r in infold),
            "temp_max_c": max(r[3] for r in infold),
            "loadavg_median": st.median(r[4] for r in infold)}


def _structure(cif: pathlib.Path) -> dict:
    import gemmi
    import numpy as np
    st_ = gemmi.read_structure(str(cif))
    st_.setup_entities()
    chains = []
    for ch in st_[0]:
        ca = np.array([[r["CA"][0].pos.x, r["CA"][0].pos.y, r["CA"][0].pos.z]
                       for r in ch if r.find_atom("CA", "*")])
        if len(ca) < 2:
            continue
        d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
        rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
        chains.append({"chain": ch.name, "residues": int(len(ca)),
                       "ca_ca_median": round(float(np.median(d)), 3),
                       "ca_ca_max": round(float(d.max()), 3),
                       "in_band_frac": round(float(((d >= CA_BAND[0]) & (d <= CA_BAND[1])).mean()), 4),
                       "backbone_breaks": int((d > CA_BREAK).sum()),
                       "rg_ratio": round(rg / (2.2 * len(ca) ** 0.38), 3)})
    return {"chains": chains, "n_atoms": sum(1 for _ in st_[0].all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--target-residues", type=int, required=True)
    ap.add_argument("--binder-residues", type=int, required=True)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--engine", default="")
    a = ap.parse_args()

    base = OUT / a.tag
    log = base / "rung.log"
    day = dt.datetime.fromtimestamp(log.stat().st_mtime).date()
    lo, hi, marks = _log_window(log, day)
    row = {"tag": a.tag, "board": "bh-p150a", "host": "pc", "card_umd": a.card,
           "tokens": a.tokens, "target_residues": a.target_residues,
           "binder_residues": a.binder_residues, "msa_rows": 1, "msa": False,
           "engine": a.engine or subprocess.run(
               ["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"],
               capture_output=True, text=True).stdout.strip(),
           "wall_s": int((base / "wall.txt").read_text().split("wall_s=")[1]),
           "rc": int((base / "wall.txt").read_text().split("rc=")[1].split()[0]),
           "progress": [m[0] for m in marks if "trunk" in m[0] or "structure" in m[0]],
           "trunk_step_s": [round(marks[i + 1][1] - marks[i][1], 1)
                            for i in range(len(marks) - 1) if "trunk" in marks[i][0]]}
    row.update(_clock(base / "aiclk.log", lo, hi))

    res = sorted(base.glob("res/af2ig_results_*/results.json"))
    if res:
        doc = json.loads(res[0].read_text())
        rec = doc[0] if isinstance(doc, list) else doc
        row["results"] = rec
    cifs = sorted(base.glob("res/af2ig_results_*/structures/*.cif")) + \
        sorted(base.glob("res/af2ig_results_*/structures/*.pdb"))
    if cifs:
        row["structure_file"] = cifs[0].name
        row["cif_sha256"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()
        row.update(_structure(cifs[0]))
        chk = subprocess.run([sys.executable, str(CHECK), str(cifs[0]), "--quiet",
                              "--json", str(base / "check.json")],
                             capture_output=True, text=True)
        report = json.loads((base / "check.json").read_text())
        row["check_structure_rc"] = chk.returncode
        row["check_structure_verdict"] = report.get("verdict")
        row["check_structure_fails"] = report.get("fail", [])
        row["check_structure_warns"] = report.get("warn", [])
    print(json.dumps(row, indent=1, default=str))
    (base / "row.json").write_text(json.dumps(row, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
