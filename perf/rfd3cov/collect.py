"""One row per RFD3 coverage rung: what ran, on which card, at what clock, and what it designed.

Written from `out/` rather than by hand. The designs and logs under it are gitignored, so this
file's output is the part of the evidence that survives in the repo, and every field in it is
read back from an artifact the run produced.

RFD3 writes the WHOLE complex -- motif plus designed, in contig order -- so unlike PXDesign the
output size does depend on the rung, and `ladder.check_artifact` already refuses a CIF whose
residue count is not the rung's. That is a capacity check and it is not a correctness one, so
there are two more signals here, each answering something the other cannot:

  * `motif_rmsd`. The motif is an INPUT: its coordinates were handed to the model. One rigid fit
    of the output's motif CAs onto the input target's own CAs is therefore the end-to-end test
    that the conditioning path survived the size -- a torn one lands in the tens of angstroms
    while still writing a full-length file. This is the rfd3 analogue of pxdesign's `fit_rmsd`.
  * The DESIGNED WINDOW, scored on its own. This model's one recorded size caveat is a break
    INSIDE the designed backbone at 768 and 832 total residues, where 640/704/896-1024 are
    clean, so a whole-file break count is the wrong instrument: 528 designed residues next to
    1008 motif residues that came out of a crystal structure will look continuous on average
    whatever the design does. The window is taken from the featurizer's own motif mask, not
    from the contig re-parsed here.
"""
from __future__ import annotations

import calendar
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a design at 1350 MHz gets reported as one at 800. Read off
#: /sys/class/tenstorrent/tenstorrent!N/device against tt-smi's bus ids, 2026-09-19.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}

#: The CA-CA distance that counts as a backbone break. Imported rather than restated so this
#: file cannot drift from the checker the rest of the fleet grades with.
from importlib.machinery import SourceFileLoader  # noqa: E402

_cs = SourceFileLoader(
    "check_structure", str(ROOT / "perf" / "wh-correctness" / "check_structure.py")).load_module()
CA_CA_BREAK = _cs.CA_CA_BREAK


def window(logs: list[Path], lo: float, hi: float) -> list[list[str]]:
    """Every sample any sampler took inside [lo, hi], deduplicated by timestamp."""
    seen: dict[str, list[str]] = {}
    for p in logs:
        for line in p.read_text().splitlines():
            f = line.split()
            if f and lo <= int(f[0]) <= hi:
                seen[f[0]] = f
    return [seen[k] for k in sorted(seen)]


# --- geometry ---------------------------------------------------------------------------

def ca_trace(path: Path) -> list[tuple[str, int, tuple[float, float, float]]]:
    """(chain, seq, xyz) per CA, in file order. The column order is READ, never assumed."""
    cols, rows = [], []
    for line in path.read_text().splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
        elif st.startswith(("ATOM", "HETATM")):
            f = st.split()
            if len(f) == len(cols):
                rows.append(f)
    def col(*names):
        for n in names:
            if n in cols:
                return cols.index(n)
        raise KeyError(names)
    an, ch, sq = col("label_atom_id", "auth_atom_id"), col("label_asym_id", "auth_asym_id"), \
        col("label_seq_id", "auth_seq_id")
    x, y, z = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    out = []
    for r in rows:
        if r[an].strip('"') == "CA":
            out.append((r[ch], int(r[sq]), (float(r[x]), float(r[y]), float(r[z]))))
    return out


def kabsch_rmsd(p: list, q: list) -> float:
    """RMSD after the optimal rigid superposition. Plain arithmetic: no scipy on this path."""
    n = len(p)
    cp = [sum(v[i] for v in p) / n for i in range(3)]
    cq = [sum(v[i] for v in q) / n for i in range(3)]
    a = [[v[i] - cp[i] for i in range(3)] for v in p]
    b = [[v[i] - cq[i] for i in range(3)] for v in q]
    import numpy as np
    A, B = np.array(a), np.array(b)
    u, s, vt = np.linalg.svd(A.T @ B)
    d = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((A @ r - B) ** 2).sum() / n))


def steps(tr: list) -> list[float]:
    return [math.dist(tr[i][2], tr[i + 1][2]) for i in range(len(tr) - 1)]


def designed_window(cif: Path, designed: int) -> dict:
    """Continuity of the designed residues alone, plus the motif/design junction.

    The designed region is the tail of the token order in every contig walked here (the motif
    mask off `featurize` is (1,)*motif + (0,)*designed for both routes), and RFD3 writes the CIF
    in token order, so the tail of the trace is the designed chain.
    """
    tr = ca_trace(cif)
    if len(tr) < designed + 2:
        return {"reason": f"trace is {len(tr)} CAs, cannot cut a {designed}-residue tail"}
    tail = tr[-designed:]
    st = steps(tail)
    brk = [round(v, 2) for v in st if v > CA_CA_BREAK]
    return {"designed_residues": len(tail), "designed_chain": tail[0][0],
            "designed_breaks": len(brk), "designed_worst_step": round(max(st), 3),
            "designed_step_median": round(statistics.median(st), 3),
            "designed_breaks_at": brk[:6],
            # The junction is the step the whole-file count would also see; kept separate
            # because a break there is the model failing to attach, not a torn design.
            "junction_step": round(math.dist(tr[-designed - 1][2], tr[-designed][2]), 3)}


def motif_rmsd(cif: Path, target: Path, motif: int) -> dict:
    """One rigid fit of the output's motif CAs onto the input target's own CAs.

    The motif is what the model was CONDITIONED on, so this is the signal that can see a torn
    conditioning path at a size that still writes a full-length file. Paired by ORDER, which is
    what the contig defines: the first `motif` CAs of the output against the CAs the contig
    selected from the target, in the same order.
    """
    out = [c[2] for c in ca_trace(cif)][:motif]
    ref = [c[2] for c in ca_trace(target)]
    if len(out) != motif or len(ref) < motif:
        return {"reason": f"{len(out)} output CAs and {len(ref)} target CAs for a {motif} motif"}
    return {"motif_residues": motif, "motif_rmsd": round(kabsch_rmsd(out, ref[:motif]), 3)}


# --- one rung ---------------------------------------------------------------------------

def score(cif: Path, out: Path, chain: str | None) -> dict | None:
    cmd = [sys.executable, str(ROOT / "perf" / "wh-correctness" / "check_structure.py"), str(cif),
           "--kind", "design", "--json", str(out), "--quiet"]
    if chain:
        cmd += ["--design-chain", chain]
    subprocess.run(cmd, check=False, capture_output=True)
    return json.loads(out.read_text()) if out.exists() else None


def build(r: dict, d: Path, spec: dict) -> dict:
    dev = int(spec["dev"])
    node = UMD_TO_NODE[dev]
    # The sampler appends, so a re-run rung directory holds both runs' samples and the idle gap
    # between them. The clock quoted has to be the clock DURING the run being reported, so the
    # window is cut from this row's own end stamp and wall time.
    end = calendar.timegm(time.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ"))
    lo, hi = end - r["wall_s"] - 5, end + 5
    clk = window(sorted(HERE.glob("out/*/aiclk.log")), lo, hi)
    host = window(sorted(HERE.glob("out/*/host.log")), lo, hi)
    row = {
        "rung": spec["name"], "tokens": spec["total"], "contig": spec["contig"],
        "motif": spec["total"] - spec["designed"], "designed": spec["designed"],
        "target": spec["target"], "steps": spec.get("steps", 100),
        "num_designs": spec.get("designs", 1),
        "umd_device": dev, "node": node, "ts": r["ts"],
        "aiclk_median_mhz": (statistics.median(int(x[node + 1]) for x in clk) if clk else None),
        "aiclk_min_mhz": (min(int(x[node + 1]) for x in clk) if clk else None),
        "aiclk_samples": len(clk),
        "wall_s": r["wall_s"], "rc": r["rc"], "verdict": r["verdict"],
        "mechanism": r.get("mechanism"),
        "host_load_median": (round(statistics.median(float(x[2]) for x in host), 1)
                             if host else None),
        "artifact": r.get("artifact"),
    }
    out_dir = d / "work" / f"out_rfd3_{spec['total']}"
    designs = []
    for cif in sorted(out_dir.rglob("*.cif")):
        one = {"cif": cif.name, "md5": hashlib.md5(cif.read_bytes()).hexdigest()}
        one.update(designed_window(cif, spec["designed"]))
        one.update(motif_rmsd(cif, ROOT / spec["target"], spec["total"] - spec["designed"]))
        rep = score(cif, d / f"struct_{cif.stem}.json", one.get("designed_chain"))
        if rep:
            one["structure_fail"] = rep["fail"]
            one["structure_warn"] = rep.get("warn")
            one["chains"] = [{k: c.get(k) for k in
                              ("chain", "n_res", "breaks", "step_median", "in_band_frac",
                               "rg_ratio")}
                             for c in rep["checks"].get("chains", [])]
            one["clashes"] = rep["checks"].get("clashes")
            one["reasons"] = rep.get("reasons", [])[:4]
        designs.append(one)
    row["designs"] = designs
    if r.get("diag"):
        row["diag"] = r["diag"][:6]
    if r.get("tail"):
        row["tail"] = r["tail"][-1200:]
    return row


def main() -> int:
    specs = json.loads((HERE / "rungs.json").read_text())
    rows = []
    for spec in specs:
        tag = "".join(ch if ch.isalnum() else "_" for ch in spec["contig"])
        # Globbed rather than joined: rung.sh built the tag with `echo | tr` at first, which
        # turned the trailing newline into a fifth underscore, so the directories on disk are
        # not all spelled the same way. A collector that cannot read a rung it already paid for
        # is the expensive kind of typo.
        pat = f"{spec['total']}_{tag}*_n{spec.get('designs', 1)}_dev{spec['dev']}"
        hits = sorted((HERE / "out").glob(pat))
        d = hits[0] if hits else HERE / "out" / pat
        jl = d / "rung.jsonl"
        if not jl.is_file():
            rows.append({"rung": spec["name"], "verdict": "NOT RUN", "dir": d.name})
            continue
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(build(json.loads(line), d, spec))
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(f"{r.get('rung'):>22}  {r.get('verdict'):>7}  {r.get('wall_s')}s  "
              f"clk={r.get('aiclk_median_mhz')}  "
              f"{[ (x.get('designed_breaks'), x.get('motif_rmsd'), x.get('structure_fail')) for x in r.get('designs', []) ]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
