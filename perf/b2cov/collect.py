"""One JSONL row per rung: what ran, on which card, at what clock, and what the structure scored.

Both axes come out of the run's own results.json -- n_tokens and msa_depth are the shapes the
model was handed (worker.py writes them from feats["restype"] and feats["msa"]), not counts read
off the fixture name or the a3m. The structure is scored with the shared instrument
perf/wh-correctness/check_structure.py, because a completed fold with a torn backbone looks like
success and is worse than a refusal.
"""
from __future__ import annotations
import hashlib, json, re, statistics, subprocess, sys
from pathlib import Path

D = Path("/home/cust-team/mthuening/b2cov")
PY = "/home/cust-team/mthuening/tt-bio/env/bin/python3.10"
CHECK = D / "eng-main" / "perf" / "wh-correctness" / "check_structure.py"
REFUSAL = re.compile(r"Out of Memory|Statically allocated circular buffers|TT_THROW", re.I)
LEGS = (("b1024", "cdk2x2_1024_d8192"), ("b1024b", "cdk2x2_1024_d8192"),
        ("b1920", "cdk2x2_1920_d8192"), ("b1920b", "cdk2x2_1920_d8192"),
        ("w2048", "cdk2x2_2048_d8192"))


def pwr(tag):
    f = D / "logs" / f"{tag}.pwr"
    if not f.exists():
        return {}
    rows = [l.split() for l in f.read_text().splitlines() if "A=" in l]
    a = [int(r[3][2:]) for r in rows if r[3][2:].isdigit()]
    p = [int(r[1][2:]) / 1e6 for r in rows if r[1][2:].isdigit()]
    ld = [float(r[4][2:]) for r in rows if len(r) > 4]
    # The first sample is taken before the device is opened and the last after it is released;
    # both read 500 MHz idle. Drop both and report them separately so an idle reading can never
    # be mistaken for an in-fold one.
    inf = a[1:-1] if len(a) > 2 else a
    return {"aiclk_samples": len(a), "aiclk_in_fold_samples": len(inf),
            "aiclk_median_mhz": statistics.median(inf) if inf else None,
            "aiclk_min_mhz": min(inf) if inf else None, "aiclk_max_mhz": max(inf) if inf else None,
            "aiclk_first_sample_mhz": a[0] if a else None,
            "aiclk_last_sample_mhz": a[-1] if a else None,
            "power_median_w": round(statistics.median(p), 1) if p else None,
            "power_max_w": round(max(p), 1) if p else None,
            "host_load_median": round(statistics.median(ld), 1) if ld else None,
            "host_load_max": max(ld) if ld else None}


def trunk_cadence(tag):
    """Seconds between consecutive `trunk i/N` ticks: forward progress, witnessed."""
    import datetime as dt
    ts = []
    for l in (D / "logs" / f"{tag}.log").read_text(errors="replace").splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d)\s+\[.*trunk (\d+)/(\d+)", l)
        if m:
            ts.append((int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)),
                       int(m.group(4))))
    return {"trunk_ticks": [t[1] for t in ts],
            "trunk_gaps_s": [b[0] - a[0] for a, b in zip(ts, ts[1:])]}


def structure(tag, cif, yml):
    out = D / "out" / tag / "struct.json"
    subprocess.run([PY, str(CHECK), str(cif), "--input", str(yml), "--json", str(out), "--quiet"],
                   capture_output=True)
    if not out.exists():
        return {}
    rep = json.loads(out.read_text())
    return {"struct_verdict": rep.get("verdict"),
            "chains": [{"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                        "step_median": c["step_median"], "in_band": c["in_band_frac"],
                        "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]],
            "clashes": rep["checks"].get("clashes")}


#: n_tokens and MSA depth AT THE MODEL, from axes.py: tt_bio.main.prepare_features at the served
#: flags, CPU only. boltz2 writes neither into results.json, so they cannot be read off the run
#: the way rf3s are, and a count taken off the fixture name would measure the generator.
AXES = {"cdk2x2_1024_d8192": {"n_tokens": 1024, "msa_depth_at_model": 8192, "n_atoms_feat": 8239},
        "cdk2x2_1920_d8192": {"n_tokens": 1920, "msa_depth_at_model": 8192, "n_atoms_feat": 15447},
        "cdk2x2_2048_d8192": {"n_tokens": 2048, "msa_depth_at_model": 8192, "n_atoms_feat": 16473}}

rows = []
for tag, rung in LEGS:
    runf = D / "logs" / f"{tag}.run"
    if not runf.exists():
        continue
    run = runf.read_text()
    log = (D / "logs" / f"{tag}.log").read_text(errors="replace")
    row = {"tag": tag, "rung": rung, "engine_sha": re.search(r"sha=(\S+)", run).group(1),
           "umd_device": int(re.search(r"card=(\d+)", run).group(1)),
           "node": int(re.search(r"node=(\d+)", run).group(1)), **pwr(tag),
           **trunk_cadence(tag),
           "refusals_in_log": sorted(set(REFUSAL.findall(log)))}
    m = re.search(r"rc=(\d+) wall=(\d+)s", run)
    row["finished"] = bool(m)
    if m:
        row |= {"rc": int(m.group(1)), "wall_s": int(m.group(2))}
    res = list((D / "out" / tag).rglob("results.json"))
    if res:
        r = json.loads(res[0].read_text())[0]
        row |= {"status": r["status"], "runtime_s": r.get("runtime_s"),
                "n_tokens": r.get("n_tokens"), "n_residues": r.get("n_residues"),
                "msa": r.get("msa"), "msa_depth": r.get("msa_depth"),
                "n_atoms": r.get("n_atoms"), "samples": r.get("samples"),
                "plddt": round(r["plddt"], 4) if r.get("plddt") is not None else None,
                "ptm": round(r["ptm"], 4) if r.get("ptm") is not None else None}
        cifs = sorted((D / "out" / tag).rglob("*.cif"))
        if cifs:
            row["cif"] = cifs[0].name
            row["cif_sha256"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16]
            row |= structure(tag, cifs[0], D / "rungs" / f"{rung}.yaml")
    else:
        row["status"] = "running" if not m else ("failed" if m.group(1) != "0"
                                                  else "no_results_json")
        # The allocator says it in its own words, plus the tt_bio origin line naming the site. A
        # wall is only comparable across trees if the request, the per-bank share and the largest
        # free block are all quoted, not just the size.

        row["last_log"] = "\n".join(log.splitlines()[-8:])
    oom = [l for l in log.splitlines() if "Out of Memory: Not enough space" in l]
    org = [l for l in log.splitlines() if "[tt_bio origin:" in l]
    if row.get("rc"):
        # The allocator says it in its own words, plus the tt_bio origin line naming the site. A
        # wall is only comparable across trees if the request, the per-bank share and the largest
        # free block are all quoted, not just the size.
        row["wall_alloc_message"] = oom[-1].split("| ")[-1].strip() if oom else None
        row["wall_site"] = org[-1].strip() if org else None
    row["caught_oom_lines"] = len(oom)
    row |= AXES.get(rung, {})
    rows.append(row)

(D / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
for r in rows:
    print(r["tag"], "dev", r["umd_device"], r.get("aiclk_median_mhz"), "MHz",
          r.get("n_tokens"), "tok", r.get("msa_depth"), "rows", r.get("runtime_s"), "s",
          "plddt", r.get("plddt"), r.get("struct_verdict"), r["status"],
          "ticks", r.get("trunk_ticks", [])[-1:] , r.get("trunk_gaps_s", [])[-3:])
