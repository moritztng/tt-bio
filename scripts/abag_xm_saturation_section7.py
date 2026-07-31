#!/usr/bin/env python3
"""Render the state doc's §7 RESULTS body from analysis.json.

Every number in §7 is transcribed by this script, never by hand: the campaign's whole
value is that the figures are measured, and hand-copying 200+ cells out of a JSON blob is
exactly where a fabricated digit would enter. Refuses to render a model whose G4 counter
is short of the full 16-target panel unless --allow_partial is given, in which case the
shortfall is printed in the header of every table.

  python3 scripts/abag_xm_saturation_section7.py [--allow_partial] > section7.md
"""
import argparse, json
from pathlib import Path

BASE = Path.home() / "abag_xm" / "saturation"
N_PANEL = 16
MODEL_LABEL = {"opendde": "opendde-abag", "protenix": "protenix-v2", "boltz2": "boltz2"}
SHOW = [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 640, 800, 1000]


def pct(x):
    return "" if x is None else f"{100.0 * x:.1f}%"


def row(label, cells):
    return "| " + label + " | " + " | ".join(cells) + " |"


def head(cols):
    return (row(cols[0], cols[1:]) + "\n"
            + "|" + "---|" * len(cols))


def oracle_tables(models, order, note):
    out = []
    for thr in ("0.23", "0.49", "0.8"):
        out.append(f"\n**Oracle (best-of-N DockQ >= {thr}) vs N{note}**\n")
        out.append(head(["N"] + [MODEL_LABEL[m] for m in order]))
        for m in SHOW:
            cells = []
            for md in order:
                blk = models[md].get(f"thr{thr}", {}).get("oracle")
                cells.append(pct(blk["mean"].get(str(m), blk["mean"].get(m))) if blk else "")
            out.append(row(str(m), cells))
    return out


def ci_table(entry):
    blk = entry["thr0.23"]["oracle"]
    out = ["\n**opendde-abag oracle @0.23 with bootstrap 95% CI over targets**\n",
           head(["N", "oracle", "CI95 lo", "CI95 hi"])]
    for m in SHOW:
        k = str(m) if str(m) in blk["mean"] else m
        lo, hi = blk["ci95"][k]
        out.append(row(str(m), [pct(blk["mean"][k]), pct(lo), pct(hi)]))
    return out


def ranked_table(models, order):
    out = ["\n**Ranked top-1 (confidence_score winner) at DockQ >= 0.23 vs N**\n",
           head(["N"] + [MODEL_LABEL[m] for m in order])]
    for m in SHOW:
        cells = []
        for md in order:
            blk = models[md].get("thr0.23", {}).get("ranked_top1")
            cells.append(pct(blk["mean"].get(str(m), blk["mean"].get(m))) if blk else "")
        out.append(row(str(m), cells))
    return out


def budget_tables(entry, label):
    b = entry["thr0.23"].get("budget")
    if not b:
        return []
    out = [f"\n**Marginal oracle per 1000 additional card-seconds ({label}, thr 0.23)**\n",
           head(["N interval", "dOracle", "d cost (card-s/target)", "pp per 1000 card-s"])]
    for iv in b["intervals"]:
        out.append(row(f"{iv['m'][0]} -> {iv['m'][1]}",
                       [f"{iv['dO_pp']:.2f} pp", f"{iv['d_cost_card_s']:.0f}",
                        "" if iv["pp_per_1000_card_s"] is None
                        else f"{iv['pp_per_1000_card_s']:.2f}"]))
    out += [f"\n**Oracle at a fixed per-target budget ({label}, thr 0.23)**\n",
            head(["budget (card-ks/target)", "oracle"])]
    for ks, v in sorted(b["oracle_at_budget_card_ks"].items(), key=lambda kv: float(kv[0])):
        out.append(row(str(ks), [pct(v)]))
    out.append(f"\n{b['budget_note']}\n")
    return out


def verdicts(models, order):
    out = ["\n**Saturation verdict (pre-registered criterion: per-doubling gain < 1.0 pp AND "
           "the next doubling's bootstrap CI lower bound < 1.0 pp)**\n"]
    for md in order:
        k = models[md].get("thr0.23", {}).get("knee")
        if not k:
            continue
        a, bb, g = k["final_interval"]
        out.append(f"- **{MODEL_LABEL[md]}: {k['verdict']}.** Final measured interval "
                   f"{a} -> {bb} gains {g:.2f} pp "
                   f"({k['final_interval_per_doubling_pp']:.2f} pp per doubling). "
                   f"Per-doubling gains: "
                   + ", ".join(f"{m}->{m2} {gg:.2f} pp" for m, m2, gg in k["doubling_gains_pp"]))
    return out


def q2_block(res):
    q2 = res.get("q2_generators")
    if not q2:
        return []
    out = [f"\n**Q2 — does depth pay equally across generators? "
           f"(shared labeled subset, n={q2['n_shared']}: {', '.join(q2['shared_targets'])})**\n",
           head(["generator", "oracle @N=50", "oracle @N=1000", "pp per doubling, 50->1000"])]
    for md in ("opendde", "protenix", "boltz2"):
        if md not in q2:
            continue
        e = q2[md]
        out.append(row(MODEL_LABEL[md], [pct(e["oracle_50"]), pct(e["oracle_1000"]),
                                         f"{e['pp_per_doubling_50_1000']:.2f}"]))
    return out


def gates(models):
    out = ["\n**Gates**\n"]
    for md, e in models.items():
        n = e["g4_n_targets_labeled"]
        out.append(f"- G4 {MODEL_LABEL[md]}: {n}/{N_PANEL} targets labeled"
                   + ("" if n == N_PANEL else " — PARTIAL, curve is over this subset only"))
    g3 = models.get("opendde", {}).get("g3_continuity")
    if g3 and "omitted" not in g3:
        for m, v in sorted(g3.items(), key=lambda kv: int(kv[0])):
            out.append(f"- G3 continuity @N={m}: saturation panel {v['saturation']:.1f}% "
                       f"(CI {v['ci95'][0]:.1f}-{v['ci95'][1]:.1f}%) vs frontier label-derived "
                       f"{v['s0_reference']:.1f}% — {'agrees' if v['agrees'] else 'DISAGREES'}")
    elif g3:
        out.append(f"- G3: {g3['omitted']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--json", default=str(BASE / "analysis.json"))
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())
    models = res.get("models") or {}
    if not models:
        raise SystemExit("analysis.json has no labeled model yet — nothing to render")
    order = [m for m in ("opendde", "protenix", "boltz2") if m in models]
    short = [m for m in order if models[m]["g4_n_targets_labeled"] != N_PANEL]
    if short and not a.allow_partial:
        raise SystemExit(f"G4: {short} short of the {N_PANEL}-target panel; "
                         f"pass --allow_partial to render anyway")
    note = "" if not short else " (PARTIAL panel — see gates)"
    lines = ["## 7. RESULTS", "",
             f"All figures below are rendered from `saturation/analysis.json` by "
             f"`scripts/abag_xm_saturation_section7.py`; no number is transcribed by hand."]
    lines += oracle_tables(models, order, note)
    if "opendde" in models:
        lines += ci_table(models["opendde"])
    lines += ranked_table(models, order)
    for md in order:
        lines += budget_tables(models[md], MODEL_LABEL[md])
    lines += verdicts(models, order)
    lines += q2_block(res)
    lines += gates(models)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
