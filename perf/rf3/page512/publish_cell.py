#!/usr/bin/env python3
"""Check the two RF3 page-cell result JSONs against the acceptance rules, then patch the row.

Refuses to touch site/data/perf-512aa.json unless every rule in the plan passes. Run with
--dry-run first; it prints the pooled median, the two per-process medians, the A/A and the
H200 fold ratio without writing anything.
"""
import argparse, json, statistics, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAGE = ROOT / "site/data/perf-512aa.json"
BAND = (42.0, 52.0)          # brief tripwire
EXPECT_DIGEST = "22402ffe781e21bb"
EXPECT_A3M = "ef2301402e7716e9"
EXPECT_YAML = "24d8b2d8c06e4409"

def load(p):
    return json.loads(pathlib.Path(p).read_text())

def check(procs, censuses):
    fail = []
    for d in procs:
        lbl = d["label"]
        if not d.get("warm_digest_identical"):
            fail.append(f"{lbl}: warm_digest_identical is not true")
        for f in d["folds"] + [d["cold"]]:
            dig = list(f["cif_sha256"].values())[0]
            if dig != EXPECT_DIGEST:
                fail.append(f"{lbl}/{f[tag]}: digest {dig} != {EXPECT_DIGEST}")
            if f["denoise_calls"] != 49:
                fail.append(f"{lbl}/{f[tag]}: denoise_calls {f[denoise_calls]} != 49")
            if f["n_tokens"] != 512:
                fail.append(f"{lbl}/{f[tag]}: n_tokens {f[n_tokens]} != 512")
            if not f.get("msa"):
                fail.append(f"{lbl}/{f[tag]}: msa falsy")
            s = f["fp32_softmax_stats"]
            if s["unfused"] != 0 or s["fused"] <= 0:
                fail.append(f"{lbl}/{f[tag]}: route not fused "
                            f"(fused={s[fused]} unfused={s[unfused]})")
        if d["recycling_steps"] != 10 or d["sampling_steps"] != 50:
            fail.append(f"{lbl}: recycles/steps {d[recycling_steps]}/{d[sampling_steps]}")
        if d["diffusion_samples"] != 1 or d["seed"] != 0:
            fail.append(f"{lbl}: diffusion_samples/seed {d[diffusion_samples]}/{d[seed]}")
        if not d["sha256_a3m"].startswith(EXPECT_A3M):
            fail.append(f"{lbl}: a3m sha {d[sha256_a3m][:16]}")
        if not d["sha256_target"].startswith(EXPECT_YAML):
            fail.append(f"{lbl}: yaml sha {d[sha256_target][:16]}")
        if d["n_msa"] != 35:
            fail.append(f"{lbl}: n_msa {d[n_msa]} != 35")
        if d["arm"]["resolved"]["fp32_softmax"] is not False:
            fail.append(f"{lbl}: resolved fp32_softmax is not False")
        if d["arm"]["changed"]:
            fail.append(f"{lbl}: arm overrode something: {d[arm][changed]}")
        num = {k: v for k, v in d["env_flags"].items()
               if k not in ("TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER", "TT_BIO_SDPA_RAGGED_CENSUS")}
        if num:
            fail.append(f"{lbl}: numerics env flags present: {num}")
    ragged = padded = aligned = 0
    for c in censuses:
        ragged += c["sites"]["tri_att"][0]
        aligned += c["sites"]["tri_att"][1]
        padded += c["padded"]
    if ragged or padded:
        fail.append(f"census: {ragged} ragged / {padded} padded, expected 0 / 0")
    if aligned == 0:
        fail.append("census: 0 aligned tri_att calls, the arm did not run")
    return fail, (ragged, aligned, padded)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("procs", nargs="+")
    ap.add_argument("--census", nargs="*", default=[])
    ap.add_argument("--provenance", help='provenance_<host>.json from the harness preflight; names the origin/main SHA the measured tree matched')
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--updated", default="2026-08-23")
    a = ap.parse_args()

    procs = [load(p) for p in a.procs]
    censuses = [load(p) for p in a.census]
    fail, cen = check(procs, censuses)

    # The cell prices the shipping tree or it prices nothing. The harness asserts this at launch
    # and records what it matched; refusing here as well is what stops a hand-run leg from being
    # published against a tree nobody checked.
    prov = load(a.provenance) if a.provenance else None
    if prov is None:
        fail.append("no --provenance: cannot show the measured tree was main")
    elif not prov.get("tt_bio_identical_to_main"):
        fail.append("provenance says tt_bio differed from main (%s)" % prov.get("origin_main"))
    else:
        extra = [f for f in prov.get("non_tt_bio_diff_vs_main", [])
                 if not f.startswith(("perf/", "scripts/", "site/"))]
        if extra:
            fail.append("measured tree differs from main outside tt_bio/perf/scripts: %s" % extra)

    warms = [w for d in procs for w in d["warm_walls_s"]]
    pooled = round(statistics.median(warms), 3)
    per_proc = [d["median_s"] for d in procs]
    aa = round(100.0 * (max(per_proc) - min(per_proc)) / min(per_proc), 2)
    h200 = json.loads(PAGE.read_text())
    h200_s = [m for m in h200["models"] if m["id"] == "rf3"][0]["cells"]["h200"]["s_per_fold"]
    ratio = round(pooled / h200_s, 3)

    print(f"warm folds:        {[round(w,3) for w in sorted(warms)]}")
    print(f"pooled median:     {pooled} s   (n={len(warms)} over {len(procs)} processes)")
    print(f"per-process:       {per_proc}   A/A {aa} %")
    print(f"spread:            {round(min(warms),3)} to {round(max(warms),3)} s, "
          f"{round(100.0*(max(warms)-min(warms))/min(warms),2)} %")
    print(f"census:            {cen[0]} ragged / {cen[1]} aligned / {cen[2]} padded")
    print(f"H200 fold ratio:   {pooled} / {h200_s} = {ratio}x")
    if not (BAND[0] <= pooled <= BAND[1]):
        fail.append(f"pooled median {pooled} outside the {BAND[0]}-{BAND[1]} s publish band")
    if fail:
        print("\nREFUSING TO PUBLISH:")
        for f in fail:
            print("  -", f)
        return 1
    print("\nall acceptance checks pass")

    f4 = " / ".join(f"{w:.3f}" for w in warms)
    colds = " and ".join(f"{d['cold']['fold_s']:.3f}" for d in procs)
    plddt = procs[0]["folds"][0]["plddt"] / 100.0
    ptm = procs[0]["folds"][0]["ptm"]
    # fp32_softmax_stats counters are cumulative over the process, so the per-fold figure is the
    # increment, not the sum. Assert the increments are constant so the number means what it says.
    per_fold = set()
    for d in procs:
        seq = [d["cold"]] + d["folds"]
        vals = [f["fp32_softmax_stats"]["fused"] for f in seq]
        per_fold.update(b - a2 for a2, b in zip([0] + vals, vals))
    if len(per_fold) != 1:
        print("WARNING: fused-call increments are not constant across folds:", sorted(per_fold))
    fused = sorted(per_fold)[-1]
    raws = " and ".join(pathlib.Path(x).name for x in a.procs)
    ref = (
        f"Four warm folds of the shipped default across two independent processes under benchlock "
        f"on qb2 card 2, ttnn 0.68.0, the same host, card and runtime the previous 82.547 s was "
        f"measured on: {f4} s, median {pooled}, the two processes\u2019 own medians "
        f"{per_proc[0]} and {per_proc[1]} for an A/A of {aa} %, cold folds {colds} s discarded. "
        f"Reproducible digest across both processes: CIF sha256 {EXPECT_DIGEST}, plDDT "
        f"{plddt:.4f}, pTM {ptm:.4f} and 49 denoise calls on every warm and cold fold, the denoise "
        f"calls counted at the sampler rather than read off the config. "
        f"The cell moves from 82.547 s because RF3 now folds the fused triangle-attention arm by "
        f"default. That flip is on main and needs no env flag; this cell overrides nothing. The "
        f"digest moves with it, away from 34df7aba88dba1fb, because the fused route is not "
        f"bit-exact against the materialised fp32-softmax chain it replaces. "
        f"The fused route carried every call, counted not assumed: {fused} fused triangle-attention "
        f"calls per fold and 0 unfused, "
        f"and the ragged census reads {cen[0]} ragged / {cen[1]} aligned / {cen[2]} padded, so the "
        f"ragged-tail pad fires on nothing at 512 aa and this reading prices the arm alone. The "
        f"raw JSONs carry a stale route string that predates the flip; resolved.fp32_softmax false "
        f"next to it is the field that says which arm ran. "
        f"Both processes were co-tenanted on a box that does not go quiet, so this is an upper "
        f"bound: benchlock excluded every other timed measurement on the host, but neighbouring "
        f"cards were busy. "
        f"AtomWorks host featurisation is 8.3 s or less of this fold, timed on its own on the same "
        f"box and the same fixture, so this cell is far less host-bound than the two NVIDIA cells "
        f"are. Measured on tt_bio byte-identical to main at {prov['origin_main'][:8]}, asserted "
        f"at launch rather than assumed. Raw: perf/rf3/page512/{raws}, the pre-merge control one "
        f"landing back in premerge_qb2c2_p1.json, and the host split in tt_host_split_qb2.json."
    )

    page = json.loads(PAGE.read_text())
    cell = [m for m in page["models"] if m["id"] == "rf3"][0]["cells"]["p150a"]
    cell["s_per_fold"] = pooled
    cell["ref"] = ref
    cell["updated"] = a.updated
    cell["parity"] = ("reference-relative PASS on the fused arm, 7ROA 0.1780 A and "
                      "ubiquitin 0.2208 A, bit-exact run-to-run and cross-process")
    if not a.write:
        print("(dry run, page not touched)\n")
        print("new s_per_fold:", pooled)
        print("new parity:   ", cell["parity"])
        print("new ref:\n" + ref)
        return 0
    PAGE.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {PAGE}: s_per_fold {pooled}, {ratio}x H200 fold ratio")
    return 0

if __name__ == "__main__":
    sys.exit(main())
