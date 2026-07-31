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


def ranked_table(models, order, thr):
    out = [f"\n**Ranked top-1 (confidence_score winner) at DockQ >= {thr} vs N**\n",
           head(["N"] + [MODEL_LABEL[m] for m in order])]
    for m in SHOW:
        cells = []
        for md in order:
            blk = models[md].get(f"thr{thr}", {}).get("ranked_top1")
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
            head(["budget (card-ks/target)", "mean N bought", "oracle", "note"])]
    mb = b.get("m_at_budget") or {}
    for ks, v in sorted(b["oracle_at_budget_card_ks"].items(), key=lambda kv: float(kv[0])):
        info = mb.get(str(ks)) or mb.get(ks) or {}
        n_cl, n_t = info.get("n_clamped"), info.get("n_targets")
        note = ""
        if n_cl:
            note = (f"{n_cl}/{n_t} targets exceed the measured range and are held at N=1000"
                    if n_cl < n_t else
                    "budget exceeds the measured range; every target held at N=1000")
        out.append(row(str(ks), [str(info.get("mean_m", "")), pct(v), note]))
    out.append(f"\n{b['budget_note']}\n")
    return out


def duplicate_block(models, order, thr="0.23"):
    """Disclose duplicate-input groups and show the treatments side by side."""
    out = []
    for md in order:
        e = models[md]
        groups = e.get("duplicate_input_groups") or []
        if not groups:
            continue
        b = e.get(f"thr{thr}", {}).get("oracle", {})
        if not b:
            continue
        names = "; ".join("/".join(g) for g in groups)
        out.append(
            f"\n**{MODEL_LABEL[md]}: duplicate-input disclosure (DockQ >= {thr})** — "
            f"{len(groups)} group(s) fold a byte-identical input ({names}), so the panel's "
            f"{b.get('n_targets')} labeled targets carry "
            f"{b.get('n_distinct_inputs', e.get('n_distinct_inputs'))} distinct inputs. The plain "
            f"panel counts a duplicated curve twice; the variants below do not.\n")
        dist = b.get("mean_distinct_inputs") or {}
        clus = b.get("ci95_clustered") or {}
        out.append(head(["N", "plain mean", "mean over distinct inputs",
                         "95% CI (per target)", "95% CI (clustered)"]))
        for m in SHOW:
            k = str(m)
            ci = b.get("ci95", {}).get(k)
            cc = clus.get(k)
            out.append(row(k, [
                pct(b["mean"].get(k)),
                pct(dist.get(k)) if dist.get(k) is not None else "",
                f"{pct(ci[0])} - {pct(ci[1])}" if ci else "",
                f"{pct(cc[0])} - {pct(cc[1])}" if cc else "",
            ]))
    return out


def verdicts(models, order):
    out = ["\n**Saturation verdict (pre-registered criterion: per-doubling gain < 1.0 pp AND "
           "the next doubling's bootstrap CI lower bound < 1.0 pp). The grid stops at 1000, so "
           "the largest N the criterion can CONFIRM is 200 (confirmed by the 400->800 doubling); "
           "a curve that flattens later is reported as such, not as no-saturation.**\n"]
    for thr in ("0.23", "0.49", "0.8"):
        out.append(f"\n*DockQ >= {thr}*\n")
        for md in order:
            k = models[md].get(f"thr{thr}", {}).get("knee")
            if not k:
                continue
            a, bb, g = k["final_interval"]
            out.append(f"- **{MODEL_LABEL[md]}: {k['verdict']}.** Final measured interval "
                       f"{a} -> {bb} gains {g:.2f} pp "
                       f"({k['final_interval_per_doubling_pp']:.2f} pp per doubling). "
                       f"Per-doubling gains: "
                       + ", ".join(f"{m}->{m2} {gg:.2f} pp"
                                   for m, m2, gg in k["doubling_gains_pp"]))
    return out


def q2_crossover(res):
    """Shared-subset curves side by side, plus the crossover verdict (section 4 item 4)."""
    q2 = res.get("q2_generators")
    if not q2 or "leader_by_n" not in q2:
        return []
    mds = [m for m in MODEL_LABEL if m in q2 and isinstance(q2.get(m), dict)
           and q2[m].get("curve_shared")]
    if not mds:
        return []
    out = [f"\n**Does depth pay equally across generators? Oracle at DockQ >= 0.23 on the "
           f"shared subset (n={q2['n_shared']}), like for like**\n",
           head(["N"] + [MODEL_LABEL[m] for m in mds] + ["leader", "margin"])]
    for m in SHOW:
        k = str(m)
        cells = []
        for md in mds:
            cv = q2[md]["curve_shared"]
            cells.append(pct(cv.get(k, cv.get(m))))
        ld = q2["leader_by_n"].get(k) or q2["leader_by_n"].get(m) or {}
        margin = ld.get("margin_pp")
        if ld.get("leader"):
            who = MODEL_LABEL[ld["leader"]]
        elif ld.get("tied_among"):
            who = "tie: " + ", ".join(MODEL_LABEL.get(x, x) for x in ld["tied_among"])
        else:
            who = ""
        cells += [who, "" if margin is None else f"{margin:.2f} pp"]
        out.append(row(k, cells))
    out.append(f"\n{q2['crossover_verdict']}\n")
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
    for thr in ("0.23", "0.49", "0.8"):
        lines += ranked_table(models, order, thr)
    for md in order:
        lines += budget_tables(models[md], MODEL_LABEL[md])
    lines += duplicate_block(models, order)
    lines += verdicts(models, order)
    lines += q2_block(res)
    lines += q2_crossover(res)
    lines += gates(models)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
