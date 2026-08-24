#!/usr/bin/env python3
"""Check the two RF3 page-cell result JSONs against the acceptance rules, then patch the row.

Refuses to touch site/data/perf-512aa.json unless every rule in the plan passes. Run with
--dry-run first; it prints the pooled median, the two per-process medians, the A/A and the
H200 fold ratio without writing anything.
"""
import argparse, json, statistics, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAGE = ROOT / "site/data/perf-512aa.json"
# The host/device gate pins RF3's device half to the cell, so the cell cannot move without it.
# Its own comment records what happens otherwise: the previous republish left this row alone and
# the gate sat red on main for a day. Moving both in one commit is the whole point of this tool.
GATE = ROOT / "perf/perf-page-host-device-publish/check_numbers.py"
BAND = (42.0, 52.0)          # brief tripwire
EXPECT_DIGEST = "22402ffe781e21bb"
EXPECT_A3M = "ef2301402e7716e9"
EXPECT_YAML = "24d8b2d8c06e4409"

def load(p):
    return json.loads(pathlib.Path(p).read_text())

def check(procs, censuses):
    fail = []
    # The page protocol is two independent processes, and the ref reports both their
    # medians by index. One process would IndexError while writing the page rather than
    # refuse, and three would silently drop one out of the sentence.
    if len(procs) != 2:
        fail.append(f"{len(procs)} result JSONs, the protocol is exactly two processes")
    for d in procs:
        lbl = d["label"]
        if not d.get("warm_digest_identical"):
            fail.append(f"{lbl}: warm_digest_identical is not true")
        for f in d["folds"] + [d["cold"]]:
            dig = list(f["cif_sha256"].values())[0]
            if dig != EXPECT_DIGEST:
                fail.append(f"{lbl}/{f['tag']}: digest {dig} != {EXPECT_DIGEST}")
            if f["denoise_calls"] != 49:
                fail.append(f"{lbl}/{f['tag']}: denoise_calls {f['denoise_calls']} != 49")
            if f["n_tokens"] != 512:
                fail.append(f"{lbl}/{f['tag']}: n_tokens {f['n_tokens']} != 512")
            if not f.get("msa"):
                fail.append(f"{lbl}/{f['tag']}: msa falsy")
            s = f["fp32_softmax_stats"]
            if s["unfused"] != 0 or s["fused"] <= 0:
                fail.append(f"{lbl}/{f['tag']}: route not fused "
                            f"(fused={s['fused']} unfused={s['unfused']})")
        if d["recycling_steps"] != 10 or d["sampling_steps"] != 50:
            fail.append(f"{lbl}: recycles/steps {d['recycling_steps']}/{d['sampling_steps']}")
        if d["diffusion_samples"] != 1 or d["seed"] != 0:
            fail.append(f"{lbl}: diffusion_samples/seed {d['diffusion_samples']}/{d['seed']}")
        if not d["sha256_a3m"].startswith(EXPECT_A3M):
            fail.append(f"{lbl}: a3m sha {d['sha256_a3m'][:16]}")
        if not d["sha256_target"].startswith(EXPECT_YAML):
            fail.append(f"{lbl}: yaml sha {d['sha256_target'][:16]}")
        if d["n_msa"] != 35:
            fail.append(f"{lbl}: n_msa {d['n_msa']} != 35")
        if d["arm"]["resolved"]["fp32_softmax"] is not False:
            fail.append(f"{lbl}: resolved fp32_softmax is not False")
        if d["arm"]["changed"]:
            fail.append(f"{lbl}: arm overrode something: {d['arm']['changed']}")
        num = {k: v for k, v in d["env_flags"].items()
               if k not in ("TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER", "TT_BIO_SDPA_RAGGED_CENSUS")}
        if num:
            fail.append(f"{lbl}: numerics env flags present: {num}")
    # The ref names the host, board, card and runtime. Derive them, and refuse a pair that does
    # not agree on all four: two legs on different cards are not one cell, and a ref that names
    # one of them describes half its own data.
    box = {tuple(str(d[k]) for k in ("host", "card", "ttnn", "card_type")) for d in procs}
    if len(box) > 1:
        fail.append(f"processes disagree on host/card/ttnn/board: {sorted(box)}")
    if len(censuses) != len(procs):
        fail.append(f"{len(censuses)} census files for {len(procs)} processes: expected one each. "
                    f"A leftover ragged_sites_<pid>.json in a reused census dir looks like an "
                    f"extra process.")
    ragged = sum(c["sites"]["tri_att"][0] for c in censuses)
    padded = sum(c["padded"] for c in censuses)
    # Aligned is reported PER PROCESS, not summed: a sum matches no single JSON a reader can open,
    # and an unequal split across processes is itself a finding rather than something to average.
    per = {c["sites"]["tri_att"][1] for c in censuses}
    aligned = sorted(per)[-1] if len(per) == 1 else -1
    if len(per) > 1:
        fail.append(f"census aligned counts differ across processes: {sorted(per)}")
    if ragged or padded:
        fail.append(f"census: {ragged} ragged / {padded} padded, expected 0 / 0")
    if aligned == 0:
        fail.append("census: 0 aligned tri_att calls, the arm did not run")
    return fail, (ragged, aligned, padded), sorted(box)[0] if box else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("procs", nargs="+")
    ap.add_argument("--census", nargs="*", default=[])
    ap.add_argument("--provenance", help='provenance_<host>.json from the harness preflight; names the origin/main SHA the measured tree matched')
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--updated", default="2026-08-23")
    ap.add_argument("--cotenancy", nargs="*", default=[],
                    help="cotenancy_<host>_<leg>.json per process, written by leg_inner.sh inside "
                         "the lock at the timed start. Required: co-tenancy cannot be inferred "
                         "from the per-fold loadavg, which is sampled at fold end, and the two "
                         "legs of a pair do not necessarily see the same box.")
    a = ap.parse_args()

    procs = [load(p) for p in a.procs]
    censuses = [load(p) for p in a.census]
    fail, cen, box = check(procs, censuses)

    # The cell prices the shipping tree or it prices nothing. The harness asserts this at launch
    # and records what it matched; refusing here as well is what stops a hand-run leg from being
    # published against a tree nobody checked.
    prov = load(a.provenance) if a.provenance else None
    # One co-tenancy record per process, matched to it by leg suffix. A pair measured under
    # different box load is a fact about the pair, not something to average into one adjective.
    coten = {}
    for f in a.cotenancy:
        c = load(f)
        coten[c["leg"]] = c
    legs = [d["label"].rsplit("_", 1)[-1] for d in procs]
    missing = [l for l in legs if l not in coten]
    if missing:
        fail.append(f"no co-tenancy record for {missing}: pass --cotenancy "
                    f"cotenancy_<host>_<leg>.json for every process")
    if prov is None:
        fail.append("no --provenance: cannot show the measured tree was main")
    elif not prov.get("tt_bio_identical_to_main"):
        fail.append("provenance says tt_bio differed from main (%s)" % prov.get("origin_main"))
    elif "ours_vs_merge_base" not in prov:
        fail.append("provenance predates the merge-base split; re-run the harness preflight")
    else:
        # What matters is what THIS branch changed. main moving ahead on a doc or a test between
        # the fold and the publish is not a reason to distrust the fold, and reading it as one
        # refused a publish that nothing was wrong with.
        extra = [f for f in prov["ours_vs_merge_base"]
                 if not f.startswith(("perf/", "scripts/", "site/"))]
        if extra:
            fail.append("this branch changed files outside perf/scripts/site: %s" % extra)

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
    print(f"census:            {cen[0]} ragged / {cen[1]} aligned per process / {cen[2]} padded")
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

    # Two sentences below were facts about one particular process, not about the protocol, so they
    # are derived rather than asserted. A ref that describes the data wrongly is worse than a ref
    # that says less.
    #
    # The route label was hardcoded and expired at the flip. It is fixed in the tree now, so the
    # caveat only belongs in the ref when the JSONs actually still carry the stale string.
    stale = any("_fp32_softmax_attention" in d["arm"]["route"] for d in procs)
    route_note = (
        "The raw JSONs carry a stale route string that predates the flip; resolved.fp32_softmax "
        "false next to it is the field that says which arm ran. " if stale else "")
    # Co-tenancy comes from the per-leg records the harness writes inside the lock, never from the
    # per-fold loadavg: that is sampled at fold END, and it read 1.96 for a process whose driver log
    # recorded six concurrent foreign folds, i.e. inferring from it would have published "the box
    # was quiet" about a co-tenanted run. And the two legs need not agree, so neither does the
    # sentence: where they differ, the difference is what explains the A/A.
    peak = max(float(f["loadavg"][0]) for d in procs for f in d["folds"])
    ntenants = [coten[l]["foreign_at_start"] for l in legs]
    tail = ("Benchlock excluded every other timed measurement on the host but not the rest of the "
            "box, so a co-tenanted leg is an upper bound. ")
    if set(ntenants) == {0}:
        quiet_note = (f"No foreign fold was running on any card at either leg start, and benchlock "
                      f"excluded every other timed measurement on the host. 1-minute loadavg "
                      f"peaked at {peak:.2f}. ")
    elif len(set(ntenants)) == 1:
        quiet_note = (f"Both processes ran co-tenanted, {ntenants[0]} foreign folds on neighbouring "
                      f"cards at leg start. " + tail)
    else:
        each = ", ".join(
            f"{l} with " + ("no foreign fold" if n == 0 else
                             f"{n} foreign fold{'' if n == 1 else 's'}") +
            f" on neighbouring cards (1-minute loadavg {coten[l]['loadavg'].split()[0]})"
            for l, n in zip(legs, ntenants))
        quiet_note = (
            f"The two processes did not see the same box: {each}, both at the timed start inside "
            f"the lock. The {aa} % between their medians is that difference and not run-to-run "
            f"variance, and the pooled median sits between them. " + tail)
    host, card, ttnn, board = box
    short = {"tt-quietbox2": "qb2", "tt-quietbox": "qb1"}.get(host, host)
    # Only claim continuity with the previous cell where the box actually is the previous box.
    same_box = (short, card, ttnn) == ("qb2", "2", "0.68.0")
    same_note = (", the same host, board, card and runtime the previous 82.547 s was measured on"
                 if same_box else ", which is NOT the box the previous 82.547 s was measured on "
                 "(qb2 card 2, ttnn 0.68.0)")
    ref = (
        f"Four warm folds of the shipped default across two independent processes under benchlock "
        f"on {short}, one Blackhole AI Processor of a {board} board, physical card {card}, "
        f"ttnn {ttnn}{same_note}: {f4} s, median {pooled}, the two processes\u2019 own medians "
        f"{per_proc[0]:.3f} and {per_proc[1]:.3f} for an A/A of {aa} %, cold folds {colds} s "
        f"discarded. "
        f"Reproducible digest across both processes: CIF sha256 {EXPECT_DIGEST}, plDDT "
        f"{plddt:.4f}, pTM {ptm:.4f} and 49 denoise calls on every warm and cold fold, the denoise "
        f"calls counted at the sampler rather than read off the config. "
        f"The cell moves from 82.547 s because RF3 now folds the fused triangle-attention arm by "
        f"default. That flip is on main and needs no env flag; this cell overrides nothing. The "
        f"digest moves with it, away from 34df7aba88dba1fb, because the fused route is not "
        f"bit-exact against the materialised fp32-softmax chain it replaces. "
        f"The fused route carried every call, counted not assumed: {fused} fused triangle-attention "
        f"calls per fold and 0 unfused, "
        f"and the ragged census reads {cen[0]} ragged / {cen[1]} aligned per process / {cen[2]} "
        f"padded, so the "
        f"ragged-tail pad fires on nothing at 512 aa and this reading prices the arm alone. "
        f"{route_note}{quiet_note}"
        f"AtomWorks host featurisation is 8.3 s or less of this fold, timed on its own on the same "
        f"box and the same fixture, so this cell is far less host-bound than the two NVIDIA cells "
        f"are. Measured on tt_bio byte-identical to main at {prov['origin_main'][:8]}, asserted "
        f"at launch rather than assumed. Raw: perf/rf3/page512/{raws}, the pre-merge control one "
        f"landing back in premerge_qb2c2_p1.json, and the host split in tt_host_split_qb2.json."
    )

    page = json.loads(PAGE.read_text())
    row = [m for m in page["models"] if m["id"] == "rf3"][0]
    cell = row["cells"]["p150a"]
    prev = cell["s_per_fold"]
    cell["s_per_fold"] = pooled
    cell["ref"] = ref
    cell["updated"] = a.updated
    cell["parity"] = ("reference-relative PASS on the fused arm, 7ROA 0.1780 A and "
                      "ubiquitin 0.2208 A, bit-exact run-to-run and cross-process")

    # The gate row, derived from the same fields the page's own JS reads. host_s is carried, not
    # re-measured, so the device half is where the cell's move lands.
    g = row["cells"]["h200"]
    host_tt = cell["split"]["host_s"]
    gate_row = (host_tt, g["split"]["host_s"], round(pooled - host_tt, 3), g["split"]["device_s"],
                round(pooled / g["s_per_fold"], 3),
                round((pooled - host_tt) / g["split"]["device_s"], 3))
    gate_src = GATE.read_text()
    old_line = [l for l in gate_src.splitlines() if l.lstrip().startswith('"rf3":')]
    if len(old_line) != 1:
        print(f"REFUSING TO PUBLISH:\n  - {GATE.name}: found {len(old_line)} rf3 rows, expected 1")
        return 1
    new_line = '    "rf3":         (%s),' % ", ".join(f"{v:.3f}" for v in gate_row)
    gate_new = gate_src.replace(old_line[0], new_line)

    print(f"gate row:          rf3 -> {gate_row}")
    if not a.write:
        print("(dry run, page and gate not touched)\n")
        print("new s_per_fold:", pooled, f"(from {prev})")
        print("new parity:   ", cell["parity"])
        print("new gate line: " + new_line.strip())
        print("new ref:\n" + ref)
        return 0
    PAGE.write_text(json.dumps(page, indent=2, ensure_ascii=False) + "\n")
    GATE.write_text(gate_new)
    print(f"wrote {PAGE}: s_per_fold {pooled}, {ratio}x H200 fold ratio")
    print(f"wrote {GATE}: rf3 device half {gate_row[2]} s")

    # Publishing is not done until the gate that reads the cell agrees with it.
    import subprocess
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
        print("\nPAGE AND GATE WRITTEN BUT THE GATE FAILS. Do not commit; read the rows above.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
