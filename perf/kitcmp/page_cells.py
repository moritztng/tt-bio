"""Add the Anthropic-kit A100 columns (exact, fast) to site/data/perf-512aa.json and the page's CELLS list.

Usage: python3 perf/kitcmp/page_cells.py <perf-512aa.json> <index.html>   (both rewritten in place)
Numbers are the warm medians from perf/kitcmp/results (parse.py); every cell carries its ref.
"""
import json
import sys

KIT = "github.com/anthropics/uplifting-biomolecular-modeling"
BOX = ("A100 SXM4 80 GB at a 400 W limit, warm median of n=3 (cold discarded), SM clock median during the arm in the "
       "ref, same fixture bytes as the stock cell (sha256 checked on the box).")

# id -> (exact s, exact spread, fast s, fast spread, box, clock, accuracy note)
CELLS = {
    "boltz2": (8.897, "8.893-8.909", 3.351, "3.345-3.351", "vast 55272030", 1410,
               "Exact is deterministic run to run; fast moves CA 0.187 A against exact."),
    "protenix-v2": (11.216, "11.214-11.270", 9.173, "9.164-9.187", "vast 55272030", 1410,
                    "CA RMSD 21.90 A exact and 21.98 A fast against stock, inside the 20.24 A the stock arm moves "
                    "between two folds of the same seed: the fixture's two CDK2 copies float against each other."),
    "opendde": (23.925, "23.805-23.927", 19.131, "19.055-19.169", "vast 55272030", 1380,
                "CA RMSD 0.414 A exact, 0.539 A fast, against a 0.675 A stock fold-to-fold floor."),
    "openfold3": (5.039, "5.035-5.078", 2.350, "2.345-2.800", "vast 55272030", 1410,
                  "CA RMSD 0.558 A exact, 4.977 A fast, against a 4.553 A stock fold-to-fold floor."),
    "openbind": (5.412, "5.404-5.426", 2.393, "2.387-2.430", "vast 55272030", 1410,
                 "CA RMSD 6.510 A exact, 8.164 A fast, against a 20.11 A stock fold-to-fold floor."),
    "esmfold2": (6.335, "6.302-6.338", 4.709, "4.694-4.721", "vast 55299292", None,
                 "CA RMSD 0.016 A exact, 0.070 A fast, stock floor 0.010 A."),
    "esmfold2-fast": (3.795, "3.788-3.855", 2.623, "2.591-2.627", "vast 55299292", None,
                      "CA RMSD 0.012 A exact, 0.072 A fast, stock floor 0.016 A."),
}
DESIGN = {
    "rfd3": (19.430, "19.37-21.83", 13.840, "13.48-15.51", "vast 55272030", 1410,
             "Batch 0 against stock batch 0, CA over target and design: 4.19 A exact, 1.58 A fast; a different "
             "design from the same stock run sits 9.43 A away. Stock on the same box read 45.53 s."),
    "pxdesign": (19.983, "19.977-20.112", 20.221, "20.217-20.482", "vast 55299292", 1410,
                 "Times InferenceRunner._inference, not the generator-stage region the stock cell publishes; stock on "
                 "the same box and region read 27.306 s, so the kit is 1.37x exact. Fast is no faster at one design: "
                 "the GPU sits at 18 % utilisation and the driver drops it to 1155 MHz. CA RMSD 0.009 A exact, "
                 "0.034 A fast, against 7.40 A between two stock seeds."),
}
BLOCKED = {
    "rf3": "Not comparable on this host. With the kit's input prefetch off (one prediction at a time, like every other "
           "cell) each fold is single-threaded CPU featurisation plus the forward, so the host sets it: on box "
           "55299292 stock read 54.16 s (n=2), kit exact 46.52 s and fast 44.42 s (n=7), 1.16x and 1.22x faster than "
           "stock on the same box, while the published stock cell is 30.89 s on a faster host. Exact is bit-equal "
           "to stock.",
    "boltzgen": "The kit's BoltzGen modes need a batch of at least 2 designs (kit README); this row is one design at a "
                "time.",
    "nesso1": "The kit has no Nesso-1 kit.",
}


def cell(s, spread, box, clock, acc, mode, key):
    clk = f"SM {clock} MHz median" if clock else "SM clock not sampled for this arm"
    return {"status": "measured", key: s,
            "ref": f"Anthropic kit {mode} mode ({KIT}), {box}, {BOX} Warm median {s} s, range {spread} s, {clk}. "
                   f"The kit's own mode switch engaged, logged as ACTIVE mode={mode}. {acc}"}


def main(jpath, hpath):
    d = json.load(open(jpath))
    P = {p["id"]: p for p in d["platforms"]}
    for mode in ("exact", "fast"):
        card = dict(P["a100"], id=f"a100_kit_{mode}", name=f"A100 SXM4 80 GB + Anthropic kit, {mode}",
                    short=f"A100 kit {mode}",
                    note=f"The same A100 SXM4 80 GB part as the stock A100 column, running the Anthropic kit ({KIT}) "
                         f"in its {mode} mode. Measured on vast.ai single-GPU nodes at a 400 W limit.")
        srv = dict(P["dgx_a100"], id=f"dgx_a100_kit_{mode}", name=f"DGX A100 640GB + Anthropic kit, {mode}",
                   short=f"DGX A100 kit {mode}", per_accelerator=card["id"])
        if card["id"] not in P:
            d["platforms"] += [card, srv]
    for sec, rows, key in (("models", CELLS, "s_per_fold"), ("design", DESIGN, "s_per_design")):
        ms = d[sec] if isinstance(d[sec], list) else d[sec]["models"]
        for m in ms:
            if m["id"] in rows:
                e, es, f, fs, box, clk, acc = rows[m["id"]]
                m["cells"]["a100_kit_exact"] = cell(e, es, box, clk, acc, "exact", key)
                m["cells"]["a100_kit_fast"] = cell(f, fs, box, clk if m["id"] != "pxdesign" else 1155, acc, "fast", key)
            elif m["id"] in BLOCKED:
                for mode in ("exact", "fast"):
                    m["cells"][f"a100_kit_{mode}"] = {"status": "blocked", "ref": BLOCKED[m["id"]]}
    for m in d["affinity"]["models"]:
        if m["id"] in BLOCKED:
            for mode in ("exact", "fast"):
                m["cells"][f"a100_kit_{mode}"] = {"status": "blocked", "ref": BLOCKED[m["id"]]}
    json.dump(d, open(jpath, "w"), indent=2, ensure_ascii=False)
    open(jpath, "a").write("\n")
    h = open(hpath).read()
    old_s = 'const SERVER_OF = {p150a:"galaxy_bh", h200:"dgx_h200", b200:"dgx_b200", a100:"dgx_a100"};'
    old_c = 'const CELLS = ["p150a","h200","b200","a100"];'
    assert old_s in h and old_c in h, "index.html CELLS/SERVER_OF moved"
    h = h.replace(old_s, old_s[:-2] + ', a100_kit_exact:"dgx_a100_kit_exact", a100_kit_fast:"dgx_a100_kit_fast"};')
    h = h.replace(old_c, 'const CELLS = ["p150a","h200","b200","a100","a100_kit_exact","a100_kit_fast"];')
    open(hpath, "w").write(h)


if __name__ == "__main__":
    main(*sys.argv[1:3])
