#!/usr/bin/env python3
"""Put the measured batch amortisation into the PXDesign row and the design section note.

Reads CURVES.json (written by curves.py) so the sentences and the numbers cannot drift apart.
Idempotent: the appended text is delimited by a marker sentence and replaced if already present.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PAGE = os.path.join(ROOT, "site", "data", "perf-512aa.json")
MARK_ROW = "Batching is the lever this protocol does not use"
MARK_SEC = "Batching is the one lever these rows leave out"


def fmt(x, n=2):
    return ("%%.%df" % n) % x


def main():
    c = json.load(open(os.path.join(HERE, "CURVES.json")))
    srv, tt, h2, b2 = c["server"], c["tt"], c["h200"], c["b200"]
    tt_best, h_best, b_best = tt["best"], h2["best"], b2["best"]
    tt_a = srv["tt_amortisation_x"]
    tt_s = srv["tt_best_batch_s_per_design"]
    b = srv["b200"]
    ttr = {r["batch"]: r for r in tt["rungs"]}
    hr = {r["batch"]: r for r in h2["rungs"]}
    br = {r["batch"]: r for r in b2["rungs"]}
    worse = [str(k) for k in sorted(ttr) if k > tt_best]
    tt_worse = " and ".join([", ".join(worse[:-1]), worse[-1]]) if len(worse) > 1 else worse[0]
    b_top = max(br)
    vram = max(r["peak_vram_GiB"] for r in b2["rungs"])

    prov = tt.get("provenance") or "on an unnamed card"
    chunk = tt.get("chunk_check")
    chunk_s = ("Capping the model's internal chunk at %d makes a batch of %d read %s s a design "
               "against the batch-of-%d rate of %s s, so a request of any size can hold that rate, "
               "though the CLI has no chunk flag today. "
               % (chunk["max_parallel_samples"], chunk["n_sample"],
                  fmt(chunk["s_per_design_at_400"]), tt_best,
                  fmt(ttr[tt_best].get("measured_s_per_design")
                      or ttr[tt_best]["fitted_s_per_design"]))) if chunk else ""

    row = (
        "%s, and it does not move the two sides equally. A batch of %d is worth %sx on the "
        "Blackhole AI Processor, confirmed at 400 steps %s. The first reading of this lever came "
        "off a p150a in pc whose matmuls are occasionally wrong, so it was a relative result only; "
        "this one is not. Batches of %s all run and all read slower per design than %d, so the "
        "ceiling is arithmetic and not capacity. %sThe same lever is worth %sx at %d on the H200 "
        "and %sx at %d on the B200, which runs one design at %s %% utilisation and %d at %s %%; the "
        "H200 goes from %s %% to %s %%. Peak allocation never passes %s GB of the B200's 183, so "
        "nothing here runs out of memory. At each side's best batch a design costs %s s on a "
        "Blackhole AI Processor against %s s on a B200, so the %sx per-accelerator lead becomes %sx "
        "and the per-server reading falls from %sx to %sx. Upstream recommends collecting 10000+ "
        "designs per target and calls --N_sample 10 a debugging run, so the batched reading is the "
        "campaign-relevant one and the seconds above are latency."
        % (MARK_ROW, tt_best, fmt(tt_a, 3), prov, tt_worse, tt_best, chunk_s,
           fmt(hr[h_best]["amortisation_x"]), h_best, fmt(br[b_best]["amortisation_x"]), b_best,
           fmt(br[1]["util_pct"], 1), b_top, fmt(br[b_top]["util_pct"], 1),
           fmt(hr[1]["util_pct"], 1), fmt(hr[max(hr)]["util_pct"], 1), fmt(vram, 1),
           fmt(tt_s), fmt(b["best_batch_s_per_design"]), fmt(b["per_accelerator_x_b1"]),
           fmt(b["per_accelerator_x_best_batch"]), fmt(b["per_server_x_b1"]),
           fmt(b["per_server_x_best_batch"])))

    sec = ("%s: at each side's best batch PXDesign's per-server reading falls from %sx to %sx and "
           "its per-accelerator reading crosses from %sx to %sx."
           % (MARK_SEC, fmt(b["per_server_x_b1"]), fmt(b["per_server_x_best_batch"]),
              fmt(b["per_accelerator_x_b1"]), fmt(b["per_accelerator_x_best_batch"])))

    doc = json.load(open(PAGE))
    px = next(m for m in doc["design"]["models"] if m["id"] == "pxdesign")
    px["note"] = re.sub(r"\s*" + re.escape(MARK_ROW) + r".*$", "", px["note"]).rstrip() + " " + row
    doc["design"]["note"] = (re.sub(r"\s*" + re.escape(MARK_SEC) + r".*$", "",
                                    doc["design"]["note"]).rstrip() + " " + sec)
    open(PAGE, "w").write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(px["note"][-1400:])
    print()
    print(doc["design"]["note"][-400:])


main()
