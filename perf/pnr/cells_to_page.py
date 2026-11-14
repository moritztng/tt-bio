#!/usr/bin/env python3
"""Put the Mol-1, BindCraft 2 and RFD4-Proteina rows on site/data/perf-512aa.json from the PNR cell files.

    python3 perf/pnr/cells_to_page.py --gate ~/gates/rel-<sha> [--rows rfd4] [--tt PATH] [--gpu PATH] [--write]
    python3 perf/pnr/cells_to_page.py --fake --out /tmp/perf-fake.json     # local render work only

The cells come from state/pnr/TT-CELLS.json (pnr-tt, qb1 p150a) and state/pnr/GPU-CELLS.json (pnr-gpu,
one rented box per GPU), measured to state/pnr/PROTOCOL.md. Nothing here is typed by hand: the row's
fixed facts (target, steps, batch rule) restate the protocol, every number is read from the two files.
A GPU cell publishes the arm the file marks `chosen`, which must be the fastest arm measured n >= 5 times.
A Tenstorrent cell's parity is the release-gate claim perf/bmk/gate_parity.py builds, or none.

--fake writes clearly fake cells (versions "FAKE", "fake": true) to --out so the render can be developed
before the measurements exist. It refuses to write the real data file, and site_publish_guard.py refuses
any cell marked fake.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "bmk"))
import gate_parity  # noqa: E402

DATA = REPO / "site" / "data" / "perf-512aa.json"
STATE = Path.home() / ".coworker" / "state" / "pnr"
GPUS = ("h200", "b200", "a100")
MIN_N, MIN_AICLK = 5, 1200

# The fixed facts of each row, restated from state/pnr/PROTOCOL.md. `cat` is the page block it lands in.
ROWS = OrderedDict([
    ("mol1", dict(cat="models", unit="s_per_fold", row=OrderedDict(
        id="mol1", name="Mol-1", recycles=20, sampling_steps=200))),
    ("bindcraft2", dict(cat="design", unit="s_per_design", row=OrderedDict(
        id="bindcraft2", name="BindCraft 2",
        target="PD-L1 (hPDL1 chain A, hotspots 54, 56, 66, 115), 80-residue binder; "
               "one design is one 140-iteration trajectory, validation not timed",
        iterations=140, batch=1))),
    ("rfd4", dict(cat="design", unit="s_per_design", row=OrderedDict(
        id="rfd4", name="RFD4-Proteina",
        target="PD-L1 (5O45 chain A, residues 17-131, hotspots A56, A115, A123), 109-residue binder, 224 tokens",
        sampling_steps=60, batch="each side's best"))),
])
IMPL = {
    "mol1": "Mol-1 as released at commit e6e64866, on every GPU, using its fastest shipped option on each: {arms}.",
    "bindcraft2": "BindCraft 2 as released at commit 7a2dfdb8a, on every GPU, using its fastest shipped option "
                  "on each: {arms}.",
    "rfd4": "RFD4-Proteina as released at commit 2db07bb4, on every GPU, at the batch and with the shipped "
            "option that ran fastest on each: {arms}.",
}

# What the design block and the timed-region line say, written from the design rows actually placed, never
# before them: "one design at a time" stays true until RFD4-Proteina, batched on both sides, lands.
def design_prose(ids) -> OrderedDict:
    bc2, rfd4 = "bindcraft2" in ids, "rfd4" in ids
    targets = ["BoltzGen at 514 tokens", "PXDesign at 592"] + (["BindCraft 2 an 80-residue binder to PD-L1"] if bc2
                                                              else []) + (["RFD4-Proteina at 224 tokens"] if rfd4 else [])
    methods = "One fixed target per row at shipped defaults. BoltzGen (500 sampling steps) and PXDesign (400) run " \
              "one design at a time."
    if bc2:
        methods += " A BindCraft 2 design is one 140-iteration design trajectory, one at a time on both sides; the " \
                   "sequence redesign and AlphaFold2 validation after it are not timed."
    if rfd4:
        methods += " RFD4-Proteina (60 steps) grows the batch on each side until it stops paying, prints that " \
                   "batch in its cell and reports wall time divided by designs."
    methods += " Figures are warm medians, with the run count on hover; GPU figures use the same protocol on " \
               "rented single-GPU nodes."
    return OrderedDict(
        cond="Seconds per binder design on one AI Processor, warm, at shipped defaults. Shorter is better. " + (
            "Each row runs one design at a time except RFD4-Proteina, which runs a batch on each side at that "
            "side's best batch size and divides the batch's wall time by its designs." if rfd4
            else "Each row runs one design at a time."),
        note=f"Each row has its own target, {', '.join(targets[:-1])} and {targets[-1]}, so compare each model "
             "only with itself.",
        methods=methods)


TIMED_RFD4 = (" RFD4-Proteina's Tenstorrent cell times featurisation, design, decoding and file writing for each "
              "batch, its GPU cells the generate call alone.")


def need(d: dict, keys, where: str) -> None:
    gone = [k for k in keys if d.get(k) in (None, "")]
    if gone:
        raise SystemExit(f"{where}: missing {', '.join(gone)}")


def tt_cell(mid: str, c: dict, unit: str, parity: str | None) -> OrderedDict:
    where = f"TT-CELLS {mid}"
    need(c, [unit, "n", "aiclk_mhz", "aiclk_min", "versions", "when"] + (["batch"] if mid == "rfd4" else []), where)
    if c["n"] < MIN_N:
        raise SystemExit(f"{where}: n={c['n']} < {MIN_N} (PROTOCOL.md)")
    if c["aiclk_min"] < MIN_AICLK:
        raise SystemExit(f"{where}: AICLK dipped to {c['aiclk_min']} MHz, the cell is void (PROTOCOL.md)")
    out = OrderedDict(status="measured")
    out[unit] = round(float(c[unit]), 3)
    out["n"] = int(c["n"])
    if c.get("spread_pct") is not None:
        out["spread_pct"] = round(float(c["spread_pct"]), 2)
    if c.get("batch") is not None:
        out["batch"] = int(c["batch"])
    out["aiclk_mhz"] = int(round(c["aiclk_mhz"]))
    out["versions"] = c["versions"]
    out["when"] = c["when"]
    if parity:
        out["parity"] = parity
    if c.get("fast"):
        f = c["fast"]
        need(f, [unit, "n", "aiclk_mhz", "versions", "when"], where + " fast")
        out["fast"] = OrderedDict([(unit, round(float(f[unit]), 3)), ("n", int(f["n"]))] +
                                  ([("spread_pct", round(float(f["spread_pct"]), 2))]
                                   if f.get("spread_pct") is not None else []) +
                                  [("aiclk_mhz", int(round(f["aiclk_mhz"]))),
                                   ("versions", f["versions"]), ("when", f["when"])])
    for k in ("fake",):
        if c.get(k):
            out[k] = True
    return out


def gpu_cell(gpu: str, mid: str, m: dict, unit: str) -> tuple[OrderedDict, str]:
    where = f"GPU-CELLS {gpu}/{mid}"
    need(m, ["chosen", "arms"], where)
    arms = m["arms"]
    if m["chosen"] not in arms:
        raise SystemExit(f"{where}: chosen arm {m['chosen']!r} is not among {sorted(arms)}")
    # A single ladder rung (n < MIN_N) is a screen, not a measurement, so it cannot outrank the chosen arm.
    fastest = min((a for a in arms if arms[a]["n"] >= MIN_N), key=lambda a: arms[a][unit], default=m["chosen"])
    if arms[fastest][unit] < arms[m["chosen"]][unit]:
        raise SystemExit(f"{where}: chosen {m['chosen']} is slower than {fastest} (protocol: publish the fastest)")
    a = arms[m["chosen"]]
    need(a, [unit, "n", "sm_clock_mhz", "versions", "when"] + (["batch"] if mid == "rfd4" else []), where)
    if a["n"] < MIN_N:
        raise SystemExit(f"{where}: n={a['n']} < {MIN_N} (PROTOCOL.md)")
    out = OrderedDict(status="measured")
    out[unit] = round(float(a[unit]), 3)
    out["n"] = int(a["n"])
    if a.get("spread_pct") is not None:
        out["spread_pct"] = round(float(a["spread_pct"]), 2)
    if a.get("batch") is not None:
        out["batch"] = int(a["batch"])
    out["clock_mhz"] = int(round(a["sm_clock_mhz"]))
    out["versions"] = f"{m['chosen']}, " + a["versions"]
    out["when"] = a["when"]
    out["ref"] = "stock"
    if a.get("fake") or m.get("fake"):
        out["fake"] = True
    return out, m["chosen"]


def build(tt: dict, gpu: dict, parity: dict, rows=tuple(ROWS)) -> list[tuple[str, OrderedDict]]:
    out = []
    for mid in rows:
        spec = ROWS[mid]
        unit = spec["unit"]
        if mid not in tt:
            raise SystemExit(f"TT-CELLS has no {mid}")
        row = OrderedDict(spec["row"])
        cells = OrderedDict(p150a=tt_cell(mid, tt[mid], unit, parity.get(mid)))
        chosen = []
        for g in GPUS:
            if mid not in gpu.get(g, {}):
                raise SystemExit(f"GPU-CELLS has no {g}/{mid}")
            cells[g], arm = gpu_cell(g, mid, gpu[g][mid], unit)
            chosen.append(f"{arm} on the {g.upper()}" +
                          (f" at batch {cells[g]['batch']}" if mid == "rfd4" else ""))
        row["nvidia_impl"] = IMPL[mid].format(arms=", ".join(chosen))
        row["cells"] = cells
        out.append((spec["cat"], row))
    return out


def place(doc: dict, built: list) -> list[str]:
    changes = []
    for cat, row in built:
        rows = doc[cat] if isinstance(doc[cat], list) else doc[cat]["models"]
        i = next((j for j, r in enumerate(rows) if r["id"] == row["id"]), None)
        if i is None:
            rows.append(row)
        else:
            rows[i] = row
        changes.append(f"{cat}/{row['id']}: " + ", ".join(
            f"{k} {c.get('s_per_fold', c.get('s_per_design'))}" + (f" b{c['batch']}" if "batch" in c else "")
            for k, c in row["cells"].items()))
    ids = {r["id"] for r in doc["design"]["models"]}
    if ids & {"bindcraft2", "rfd4"}:
        doc["design"].update(design_prose(ids))
    if "rfd4" in ids and TIMED_RFD4.strip() not in doc["scope"]["timed_region"]:
        doc["scope"]["timed_region"] += TIMED_RFD4
    return changes


def fake_cells() -> tuple[dict, dict]:
    """Obviously fake numbers in the real files' shape, for rendering before the measurements land."""
    tt, gpu = {}, {g: {} for g in GPUS}
    base = {"mol1": 99.0, "bindcraft2": 88.0, "rfd4": 7.0}
    for mid, spec in ROWS.items():
        unit = spec["unit"]
        tt[mid] = {unit: base[mid], "n": 5, "spread_pct": 1.0, "aiclk_mhz": 1350, "aiclk_min": 1350,
                   "versions": "FAKE local render fixture", "when": "1970-01-01T00:00:00Z", "fake": True,
                   **({"batch": 20} if mid == "rfd4" else {})}
        for i, g in enumerate(GPUS):
            gpu[g][mid] = {"chosen": "fake-arm", "fake": True, "arms": {"fake-arm": {
                unit: base[mid] * (1.5 + i) / 4, "n": 5, "spread_pct": 1.0, "sm_clock_mhz": 1000 + i,
                "versions": "FAKE local render fixture", "when": "1970-01-01T00:00:00Z",
                **({"batch": 256} if mid == "rfd4" else {})}}}
    return tt, gpu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tt", type=Path, default=STATE / "TT-CELLS.json")
    ap.add_argument("--gpu", type=Path, default=STATE / "GPU-CELLS.json")
    ap.add_argument("--gate", type=Path, help="finished release gate dir, ~/gates/rel-<sha>")
    ap.add_argument("--extra", type=Path, help="parity legs run outside the gate (gate_parity --extra)")
    ap.add_argument("--fake", action="store_true", help="fake cells, written only to --out")
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--rows", default=",".join(ROWS), help="comma-separated subset of " + ", ".join(ROWS))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rows = a.rows.split(",")
    if set(rows) - set(ROWS):
        raise SystemExit(f"--rows: unknown {', '.join(sorted(set(rows) - set(ROWS)))}")
    if a.fake:
        if a.out.resolve() == DATA.resolve():
            raise SystemExit("--fake never writes the published data file; pass --out /tmp/...")
        tt, gpu = fake_cells()
        parity = {m: "Passes its release-gate accuracy check (FAKE)" for m in ROWS}
    else:
        if not a.gate:
            raise SystemExit("--gate is required for real cells")
        tt, gpu = json.loads(a.tt.read_text()), json.loads(a.gpu.read_text())
        parity = gate_parity.claims(a.gate.expanduser(), a.extra)
    doc = json.loads(DATA.read_text(), object_pairs_hook=OrderedDict)
    print("\n".join(place(doc, build(tt, gpu, parity, rows))))
    missing = [m for m in rows if m not in parity]
    if missing:
        print(f"no release-gate parity claim for {', '.join(missing)}: render_check will refuse the landing "
              "page's 1:1 accuracy line until the gate carries their legs")
    if a.write or a.fake:
        a.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
