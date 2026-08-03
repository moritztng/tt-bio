#!/usr/bin/env python3
"""AbAg-XM deep-N galaxy cost refit at the N=256 rung (state doc
abag-xm-deepn-saturation-fullpanel, PHASE 2 cost model).

Reads pc staging (~/abag_xm/deepn/galaxy/): fleet_results.jsonl (merged, line-deduped
across windows) + reused_chunks.p27.jsonl (skip-and-link provenance). Separates
FLEET-FOLDED p27 rung-256 chunk records (uncontended timings -- the p26 contention
class was reclassified as the other tenant's disk-fill, pass 29) from REUSED c0
records (seconds paid in earlier windows, never a p27 cost basis), then:

  1. per-model/per-target median chunk seconds from fleet-folded ok records,
  2. N=256 rung projection: sum over the full 1296-task chunk plan of per-target
     median (fallback: model median) -- the rung's total card-h,
  3. N=512 marginal projection: chunks 4-7 fold fresh (same per-target rate),
     chunks 0-3 skip-and-link (zero marginal) => marginal ~= the N=256 fresh total,
  4. contamination check: reused-record share of the rung-256 ok set (those seconds
     must not enter the p27 refit).

Writes costfit_n256.json next to the inputs and prints the table. CPU-only, runs on pc.
"""
import json, statistics, sys
from pathlib import Path

GAL = Path.home() / "abag_xm" / "deepn" / "galaxy"
RUNG = 256
CHUNKS = 4          # p27 rung-256 plan: 4 chunks x 64 samples
N512_FRESH = 4      # p28 folds chunks 4-7 fresh; 0-3 skip-and-link
PLAN = {"boltz2": 164, "opendde-abag": 160}   # targets (od minus 4 WH DRAM exclusions)


def main():
    reused = set()
    rp = GAL / "reused_chunks.p27.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("rung") == RUNG:
                reused.add((r["model"], r["target"], r["chunk"]))

    folded = {}   # (model, target) -> [seconds] fleet-folded ok chunk records at rung 256
    reused_ok = 0
    for line in (GAL / "fleet_results.jsonl").read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("rung") != RUNG or r.get("rc") != 0 or r.get("cifs", 0) != 64:
            continue
        k = (r["model"], r["target"], r["chunk"])
        if k in reused:
            reused_ok += 1
            continue
        folded.setdefault((r["model"], r["target"]), []).append(r["seconds"])

    out = {}
    for model, n_targets in PLAN.items():
        tsec = {t: statistics.median(v) for (m, t), v in folded.items() if m == model}
        if not tsec:
            out[model] = {"error": "no fleet-folded records yet"}
            continue
        model_med = statistics.median(tsec.values())
        # rung total: measured targets at their own median x full chunk plan, the rest at
        # the model median
        seen = sum(v * CHUNKS for v in tsec.values())
        unseen = (n_targets - len(tsec)) * CHUNKS * model_med
        rung_s = seen + unseen
        out[model] = {
            "targets_measured": len(tsec),
            "chunk_median_s": round(model_med, 1),
            "chunk_p90_s": round(sorted(tsec.values())[int(0.9 * len(tsec)) - 1], 1),
            "n256_rung_card_h": round(rung_s / 3600, 1),
            "n512_marginal_card_h": round(rung_s * N512_FRESH / CHUNKS / 3600, 1),
        }
    summary = {
        "rung": RUNG,
        "fleet_folded_ok_chunks": sum(len(v) for v in folded.values()),
        "reused_ok_chunks_excluded": reused_ok,
        "models": out,
        "n256_total_card_h": round(sum(m["n256_rung_card_h"] for m in out.values()
                                       if "n256_rung_card_h" in m), 1),
        "n512_marginal_card_h": round(sum(m["n512_marginal_card_h"] for m in out.values()
                                          if "n512_marginal_card_h" in m), 1),
    }
    (GAL / "costfit_n256.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
