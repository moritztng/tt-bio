#!/usr/bin/env python3
"""of3t-covdefault: the inference A/B for TT_BIO_OF3_DEVICE_REFATOM, one writer, one artifact.

Four things, because none of them is readable alone:

  region    the time the FLAG'S OWN CODE takes inside the shipped fold, at perf_counter
            resolution, from the census hook in `usercustomize.py`. tt_bio's per-target runtime
            is rounded to 0.1 s and the flag's effect is ~5e-2 s warm, so the fold line cannot
            resolve it and the changed region has to be timed where it runs.
  fold      the per-target wall clock tt_bio itself reports, with the AICLK sampled DURING that
            window on the chip that actually ran -- TT_VISIBLE_DEVICES=2 opens the chip whose
            sysfs node is `tenstorrent!3` on this box, so the granted card and the node number
            are not the same integer.
  move      the Angstrom the flag moves the structure, against this target's own seed floor and
            an A/A floor measured in separate processes.
  census    which functions each process called and which branch of the gate they took.

  python3 perf/of3t_covdefault/ab_fold.py
"""
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import gemmi

W = Path(__file__).resolve().parents[2]
COLD = Path("/tmp/of3t/of3t-covdefault/fold")
WARM = Path("/tmp/of3t/of3t-covdefault/warm")
GT = W / "examples/ground_truth_structures/ubiquitin.pdb"
NODE = 3          # sysfs node of the chip TT_VISIBLE_DEVICES=2 opens on qb1
OUT = Path(__file__).with_name("AB_FOLD.json")

FOLD_LINE = re.compile(r"(\d\d):(\d\d):(\d\d)\s+\[[^\]]*\]\s+\S+\s+(\S+)\s+—\s+([\d.]+)s")


def census(d: Path) -> dict:
    """The one process that imported the module -- the spawned worker that folds."""
    for c in sorted(d.glob("census.*.json")):
        j = json.loads(c.read_text())
        if j.get("module_imported"):
            return j
    return {}


def clocks(d: Path):
    rows = [ln.split("\t") for ln in (d / "aiclk.tsv").read_text().splitlines() if "\t" in ln]
    return [(int(r[0]), int(r[NODE + 1])) for r in rows
            if len(r) > NODE + 1 and r[NODE + 1].strip()]


def folds(d: Path):
    """Per-target wall clock and the AICLK over THAT target's window, not the process's."""
    log = (d / "fold.log").read_text()
    samples = clocks(d)
    if not samples:
        return []
    day = datetime.fromtimestamp(samples[0][0], timezone.utc).astimezone()
    out = []
    for hh, mm, ss, name, secs in FOLD_LINE.findall(log):
        secs = float(secs)
        end = day.replace(hour=int(hh), minute=int(mm), second=int(ss),
                          microsecond=0).timestamp()
        win = [c for t, c in samples if end - secs - 1 <= t <= end + 1]
        out.append({"target": name, "fold_s": secs, "aiclk_n": len(win),
                    "aiclk_mean": round(sum(win) / len(win), 1) if win else None,
                    "aiclk_min": min(win) if win else None,
                    "aiclk_max": max(win) if win else None})
    return out


def region(c: dict, n_folds: int):
    """The flag's own region, per fold: the input-embedder leg plus the diffusion ref-atom leg.

    `run_input_atom_encoder` wraps the input embedder's own host `ref_atom_embed` call, which is
    on the host in BOTH arms, so it is inside the region in both and cancels. What the flag
    swaps is the aggregation head inside that call and the diffusion module's leg outside it.
    """
    t = c.get("times_s", {})
    ie = [x * 1e3 for x in t.get("run_input_atom_encoder", [])]
    dev = [x * 1e3 for x in t.get("ref_atom_embed_device", [])]
    host_all = [x * 1e3 for x in t.get("ref_atom_embed", [])]
    if dev:                       # ON: one host call per fold (the input embedder's)
        diff = dev
    else:                         # OFF: two host calls per fold, the SECOND is the diffusion one
        diff = host_all[1::2]
    rows = []
    for i in range(min(n_folds, len(ie), len(diff))):
        rows.append({"run_input_atom_encoder_ms": round(ie[i], 3),
                     "diffusion_ref_atom_leg_ms": round(diff[i], 3),
                     "region_ms": round(ie[i] + diff[i], 3)})
    return rows


def ca(path):
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    return {(ch.name, r.seqid.num): r.find_atom("CA", "*").pos
            for ch in st[0] for r in ch if r.find_atom("CA", "*") is not None}


def rmsd(a, b, reverse=False):
    keys = sorted(set(a) & set(b))
    pa = [a[k] for k in keys]
    pb = [b[k] for k in (keys[::-1] if reverse else keys)]
    return gemmi.superpose_positions(pa, pb).rmsd, len(keys)


def main():
    out = {
        "instrument": "of3t-covdefault ab_fold.py -- the inference A/B for "
                      "TT_BIO_OF3_DEVICE_REFATOM: the changed region timed in situ, the fold "
                      "wall clock with the AICLK sampled DURING it, the Angstrom move against "
                      "this target's seed floor, and the runtime call census",
        "host": "qb1 (tt-quietbox) card 2 (UMD logical; sysfs node tenstorrent!3), "
                "Blackhole p150a, subsystem_device 0x0040",
        "card_numbering_note": "TT_VISIBLE_DEVICES=2 with TT_BIO_LEASE_CARDS=2 opens the chip "
                               "whose sysfs AICLK node is tenstorrent!3. Nodes 0, 1 and 2 stay "
                               "at 800 MHz through every fold in this artifact and node 3 "
                               "boosts to 1350; sampling only node 2 reads 800 MHz for a fold "
                               "that ran at 1350 and the number is an artifact",
        "target": "examples/ubq.yaml, 76 aa, 601 atoms, --single_sequence, 1 sample, 20 steps",
        "cold": {}, "warm": {}, "per_arm_region_ms": {}, "per_arm_fold_s": {},
        "census": {}, "move": {},
    }

    for tag, root, n in [("cold", COLD, 1), ("warm", WARM, 3)]:
        for d in sorted(root.glob("*")):
            if not (d / "fold.log").is_file():
                continue
            c = census(d)
            out[tag][d.name] = {"folds": folds(d), "region": region(c, n)}
            out["census"][f"{tag}:{d.name}"] = {
                "env_TT_BIO_OF3_DEVICE_REFATOM": c.get("env_TT_BIO_OF3_DEVICE_REFATOM"),
                "module_imported": c.get("module_imported"),
                "calls": c.get("calls"), "gate_branch": c.get("gate_branch")}

    # per-arm aggregation, cold: one fold per process, so an arm is its repeats
    for arm in ("of3_off_s0", "of3_on_s0", "ob_off_s0", "ob_on_s0", "of3_off_s1", "ob_off_s1"):
        reg = [r["region_ms"] for k, v in out["cold"].items() if k.startswith(arm)
               for r in v["region"]]
        fs = [f["fold_s"] for k, v in out["cold"].items() if k.startswith(arm)
              for f in v["folds"]]
        clk = [f["aiclk_mean"] for k, v in out["cold"].items() if k.startswith(arm)
               for f in v["folds"] if f["aiclk_mean"]]
        if reg:
            out["per_arm_region_ms"][f"cold:{arm}"] = {
                "n": len(reg), "median": round(statistics.median(reg), 3),
                "min": min(reg), "max": max(reg)}
        if fs:
            out["per_arm_fold_s"][f"cold:{arm}"] = {
                "n": len(fs), "folds_s": sorted(fs),
                "median": round(statistics.median(fs), 3), "min": min(fs), "max": max(fs),
                "aiclk_during_mean": round(sum(clk) / len(clk), 1) if clk else None,
                "aiclk_during_min": min(
                    f["aiclk_min"] for k, v in out["cold"].items() if k.startswith(arm)
                    for f in v["folds"] if f["aiclk_min"]),
                "aiclk_during_max": max(
                    f["aiclk_max"] for k, v in out["cold"].items() if k.startswith(arm)
                    for f in v["folds"] if f["aiclk_max"])}

    # per-arm aggregation, warm: fold 1 of a process is cold, folds 2+ are warm
    for name, v in out["warm"].items():
        if not v["region"]:
            continue
        out["per_arm_region_ms"][f"warm:{name}:fold1_cold"] = v["region"][0]["region_ms"]
        rest = [r["region_ms"] for r in v["region"][1:]]
        if rest:
            out["per_arm_region_ms"][f"warm:{name}:folds2plus"] = {
                "n": len(rest), "median": round(statistics.median(rest), 3),
                "min": min(rest), "max": max(rest)}
        out["per_arm_fold_s"][f"warm:{name}"] = {
            "folds_s": [f["fold_s"] for f in v["folds"]],
            "aiclk_during": [f["aiclk_mean"] for f in v["folds"]]}

    # the deltas the decision rests on
    def med(k):
        r = out["per_arm_region_ms"].get(k)
        return r["median"] if isinstance(r, dict) and "median" in r else None

    out["deltas_ms"] = {
        "openfold3_cold_on_minus_off": round(med("cold:of3_on_s0") - med("cold:of3_off_s0"), 3),
        "openbind_cold_on_minus_off": round(med("cold:ob_on_s0") - med("cold:ob_off_s0"), 3),
        "openfold3_warm_on_minus_off": round(
            med("warm:warm_of3_on:folds2plus") - med("warm:warm_of3_off:folds2plus"), 3),
        "openbind_warm_on_minus_off": round(
            med("warm:warm_ob_on:folds2plus") - med("warm:warm_ob_off:folds2plus"), 3),
        "sign_convention": "positive means the flag ON is SLOWER",
    }

    # ---- the Angstrom move -------------------------------------------------------------------
    runs = {}
    for d in sorted(COLD.glob("*")):
        hits = sorted(d.glob("*_results_ubq/structures/ubq.cif"))
        if hits:
            runs[d.name] = hits[0]
    dig = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in runs.items()}
    crd = {k: ca(v) for k, v in runs.items()}
    out["move"] = {"runs": {k: {"sha256": dig[k], "n_ca": len(crd[k])} for k in runs},
                   "distinct_digests": sorted(set(dig.values())), "pairs": {}}

    def pair(label, x, y, what):
        if x in crd and y in crd:
            r, n = rmsd(crd[x], crd[y])
            out["move"]["pairs"][label] = {"a": x, "b": y, "ca_rmsd_A": r, "n_ca": n,
                                           "bit_identical": dig[x] == dig[y], "what": what}

    for p, m in (("of3", "openfold3"), ("ob", "openbind")):
        for r in ("b", "c"):
            pair(f"{p}_AA_off_a_vs_off_{r}", f"{p}_off_s0_a", f"{p}_off_s0_{r}",
                 f"{m} A/A floor, flag OFF, separate processes")
            pair(f"{p}_AA_on_a_vs_on_{r}", f"{p}_on_s0_a", f"{p}_on_s0_{r}",
                 f"{m} A/A floor, flag ON, separate processes")
        for r in ("a", "b", "c"):
            pair(f"{p}_LEVER_off_vs_on_s0_{r}", f"{p}_off_s0_{r}", f"{p}_on_s0_{r}",
                 f"{m}, the flag, matched seed 0 and matched card")
        pair(f"{p}_SEEDFLOOR_off_s0_vs_s1", f"{p}_off_s0_a", f"{p}_off_s1_a",
             f"{m} seed floor: same arm, seed 0 vs seed 1")
    if "of3_off_s0_a" in crd:
        r0, n0 = rmsd(crd["of3_off_s0_a"], crd["of3_off_s0_a"])
        rb, _ = rmsd(crd["of3_off_s0_a"], crd["of3_off_s0_a"], reverse=True)
        out["move"]["controls"] = {"ZERO_self_vs_self_A": r0,
                                   "BREAK_residue_order_reversed_A": rb, "n_ca": n0}
    if GT.is_file():
        gt = ca(GT)
        out["move"]["vs_experimental"] = {
            k: {"ca_rmsd_A": rmsd(crd[k], gt)[0]} for k in sorted(crd)}
        out["move"]["vs_experimental_source"] = str(GT.relative_to(W))

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"deltas_ms": out["deltas_ms"],
                      "per_arm_region_ms": out["per_arm_region_ms"],
                      "per_arm_fold_s": out["per_arm_fold_s"],
                      "move_pairs": {k: round(v["ca_rmsd_A"], 6)
                                     for k, v in out["move"]["pairs"].items()},
                      "controls": out["move"].get("controls"),
                      "vs_experimental": out["move"].get("vs_experimental")}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
