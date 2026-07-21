#!/usr/bin/env python3
"""Render the implementation-parity verdict chart.

Reads the committed result JSONs in docs/implementation-parity-data/ and plots
the gate-metric X/floor ratio per stochastic leg, with the floor = 1.0 line and
bars colored by verdict (green PASS, amber PASS-caveated, red GAP-evidenced).

Deterministic legs (ESMC, SaProt) and special-metric legs (BoltzGen designability,
OpenDDE-abag DockQ) have no X/floor ratio and are not plotted; they are listed in
the verdict table in docs/implementation-parity.md.

Regenerate with:
    python3 scripts/plot_parity.py
        # writes docs/implementation-parity-data/implementation-parity.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent / "docs" / "implementation-parity-data"
OUT = DATA / "implementation-parity.png"

# Each leg: (label, json, verdict, accessor) where accessor returns cross_over_floor
# from the committed JSON. Verdicts are copied verbatim from the verdict table in
# docs/implementation-parity.md; the ratio value is read from the JSON (regenerable).
PASS, CAVEAT, GAP = "PASS", "PASS-caveated", "GAP-evidenced"


def _struct(path, target, metric):
    d = json.loads((DATA / path).read_text())
    return d["targets"][target][metric]["cross_over_floor"]


def _aff(path, metric):
    d = json.loads((DATA / path).read_text())
    return d["metrics"][metric]["cross_over_floor"]


def _esmfold2(protein):
    arr = json.loads((DATA / "esmfold2.json").read_text())
    for p in arr:
        if p["protein"] == protein:
            return p["kabsch_rmsd"]["cross_over_floor"]
    raise KeyError(protein)


def _protenix_7roa():
    d = json.loads((DATA / "protenix-v2.json").read_text())
    return d["parity"]["floor"]["cross_over_floor"]


def _opendde_prod():
    d = json.loads((DATA / "opendde-prod-leg.json").read_text())
    return d["production_rdx"]["rmsd"]["cross_over_floor"]


LEGS = [
    ("ESMFold2 trp-cage L20", "esmfold2", PASS, lambda: _esmfold2("trpcage")),
    ("ESMFold2 GB1 L56", "esmfold2", PASS, lambda: _esmfold2("gb1")),
    ("ESMFold2 ubiquitin L76", "esmfold2", PASS, lambda: _esmfold2("ubiquitin")),
    ("ESMFold2 lysozyme L129", "esmfold2", PASS, lambda: _esmfold2("lysozyme")),
    ("Protenix-v2 7ROA MSA L117", "protenix", PASS, _protenix_7roa),
    ("Protenix-v2 ubq MSA L76", "protenix", PASS, lambda: _struct("protenix-v2-ubiquitin.json", "ubq", "kabsch_rmsd")),
    ("Protenix-v2 HSA MSA L585", "protenix", GAP, lambda: _struct("protenix-v2-hsa.json", "hsa", "rmsd")),
    ("Boltz-2 trp-cage no-MSA L20", "boltz2", PASS, lambda: _struct("boltz2.json", "trpcage", "rmsd")),
    ("Boltz-2 7ROA no-MSA L117", "boltz2", PASS, lambda: _struct("boltz2-prot-nomsa-restore.json", "prot_no_msa", "rmsd")),
    ("Boltz-2 7ROA MSA L117", "boltz2", PASS, lambda: _struct("boltz2.json", "prot_msa", "rmsd")),
    ("Boltz-2 ubiquitin no-MSA L76", "boltz2", CAVEAT, lambda: _struct("boltz2-ubiquitin.json", "ubiquitin_no_msa", "kabsch_rmsd")),
    ("Boltz-2 HSA no-MSA L585", "boltz2", PASS, lambda: _struct("boltz2-hsa.json", "hsa_no_msa", "rmsd")),
    ("OpenDDE trp-cage L20", "opendde", PASS, lambda: _struct("opendde.json", "trpcage", "rmsd")),
    ("OpenDDE 7ROA prod L117", "opendde", PASS, _opendde_prod),
    ("Boltz-2 aff FKBP12 L107", "affinity", CAVEAT, lambda: _aff("boltz2-affinity-fkbp12-5x5.json", "affinity_pred_value")),
    ("Boltz-2 aff DHFR L187", "affinity", CAVEAT, lambda: _aff("boltz2-affinity-dhfr.json", "affinity_pred_value")),
    ("Boltz-2 aff trypsin L223", "affinity", CAVEAT, lambda: _aff("boltz2-affinity-tryp.json", "affinity_pred_value")),
]

COLOR = {PASS: "#2ca02c", CAVEAT: "#ff9f1c", GAP: "#d62728"}


def main() -> None:
    rows = [(label, verdict, acc()) for label, _g, verdict, acc in LEGS]
    labels = [r[0] for r in rows]
    ratios = [r[2] for r in rows]
    colors = [COLOR[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ypos = range(len(labels))
    bars = ax.barh(list(ypos), ratios, color=colors, height=0.62, zorder=3)
    ax.axvline(1.0, color="#444", linestyle="--", linewidth=1.0, zorder=2, label="floor = 1.0")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("device-vs-reference / floor  (X / max(R, D))")
    ax.set_title("Implementation parity: gate-metric X/floor per stochastic leg")
    ax.set_xlim(0, max(2.2, max(ratios) * 1.12))
    ax.grid(axis="x", linestyle=":", color="#bbb", zorder=0)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{r:.2f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[v]) for v in (PASS, CAVEAT, GAP)]
    ax.legend(handles + [plt.Line2D([0], [0], color="#444", linestyle="--")],
              [PASS, CAVEAT, GAP, "floor = 1.0"],
              loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
