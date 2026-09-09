#!/usr/bin/env python3
"""One (model, token-count) rung of the Blackhole 1536 ladder.

Runs the fold as a subprocess, then judges it on the ARTIFACT: a CIF with the
expected number of residues in it, and a results.json whose token count matches
the fixture. An exit status of 0 is not accepted as evidence, and neither is a
"status": "ok" field -- both have shipped a green run over an empty output
folder before. Peak host RSS and MemAvailable floor are sampled while it runs,
because a host OOM has taken one of these boxes down.

Appends one JSON object per rung to results.jsonl and one line to sweep.log.
"""
import argparse, json, os, re, resource, shutil, subprocess, sys, time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
OUTROOT = WT / "perf" / "bh1536"
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"

# The refusals we classify. DRAM and L1 are different walls with different fixes.
OOM_PATTERNS = [
    (re.compile(r"Not enough space to allocate (\d+) B DRAM buffer across (\d+) banks.*?"
                r"each bank needs to store (\d+) B, but bank size is (\d+) B", re.S), "dram"),
    (re.compile(r"Not enough space to allocate (\d+) B L1 buffer across (\d+) banks.*?"
                r"each bank needs to store (\d+) B, but bank size is (\d+) B", re.S), "l1"),
    (re.compile(r"grow to (\d+) B .*?beyond max L1 size of (\d+) B", re.S), "l1_cb"),
]


def sample_host():
    rss = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "statm").read_text().split()
            cmd = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if b"tt_bio" in cmd or b"python" in cmd:
            rss += int(st[1]) * resource.getpagesize()
    avail = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    return rss, avail


def cif_residues(path: Path) -> int:
    """Distinct (chain, seq id) in the atom records. Reads the structure, not a header."""
    seen = set()
    text = path.read_text(errors="replace")
    if "_atom_site." in text:                      # mmCIF loop
        cols, rows = [], []
        in_loop = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1]); in_loop = True; continue
            if in_loop:
                if s.startswith("#") or s.startswith("_") or s.startswith("loop_"):
                    if rows: break
                    in_loop = False; cols = []; continue
                if s: rows.append(s.split())
        try:
            ch, ri = cols.index("label_asym_id"), cols.index("label_seq_id")
        except ValueError:
            return 0
        for r in rows:
            if len(r) > max(ch, ri):
                seen.add((r[ch], r[ri]))
    else:                                          # PDB
        for line in text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                seen.add((line[21], line[22:27]))
    return len(seen)


def _oom_of(text):
    for pat, kind in OOM_PATTERNS:
        m = pat.search(text)
        if m:
            return {"class": kind, "groups": [int(g) for g in m.groups()],
                    "text": m.group(0)[:400].replace("\n", " ")}
    return None


def _write(row):
    OUTROOT.mkdir(parents=True, exist_ok=True)
    with (OUTROOT / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    line = (f"{row['when']} {row['model']:14s} {row['size']:5d}tok {row['verdict']:8s} "
            f"wall={row['wall_s']:7.1f}s engine={row['engine_runtime_s']} "
            f"cif_res={row['cif_residues']}/{row['size']} ntok={row['n_tokens']} "
            f"rss={row['peak_host_rss_gib']}G "
            f"oom={row['oom']['class'] if row['oom'] else '-'}"
            f"{'(fatal)' if row['fatal_oom'] else ''}"
            + (f" breaks={row['struct'].get('ca_breaks')} "
               f"worst_ca={row['struct'].get('worst_ca_ca')} "
               f"clash={row['struct'].get('clash_frac')}" if row.get("struct") else ""))
    with (OUTROOT / "sweep.log").open("a") as fh:
        fh.write(line + "\n")
    print(line)


def _judge_affinity(a, out, text, wall, proc, killed, peak_rss, floor_avail):
    """nesso1 writes no structure. The artifact is the scalar, so read it and require a number."""
    scores = sorted(out.rglob("*_affinity.json"))
    value = None
    if scores:
        try:
            d = json.loads(scores[0].read_text())
            for k in ("affinity_pred_value", "affinity", "pred_value", "value"):
                if isinstance(d.get(k), (int, float)):
                    value = float(d[k]); break
            if value is None:
                for v in d.values():
                    if isinstance(v, (int, float)):
                        value = float(v); break
        except Exception:
            value = None
    oom = _oom_of(text)
    ok = value is not None
    verdict = "PASS" if ok else ("TIMEOUT" if killed else ("OOM" if oom else "FAIL"))
    row = {"model": a.model, "size": a.size, "tag": a.tag, "task": "affinity",
           "verdict": verdict, "wall_s": round(wall, 1), "engine_runtime_s": None,
           "cif": str(scores[0]) if scores else None,
           "cif_residues": a.size if ok else 0, "n_tokens": None,
           "affinity": value, "exit": proc.returncode, "killed": killed,
           "single_sequence": True, "sampling_steps": None,
           "recycling_steps": a.recycling_steps,
           "peak_host_rss_gib": round(peak_rss / 2**30, 2),
           "floor_memavail_gib": round(floor_avail / 2**30, 2),
           "oom": oom, "fatal_oom": bool(oom and not ok),
           "tail": text[-1200:].replace("\n", " | ") if verdict != "PASS" else "",
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(row)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--budget", type=int, default=2400, help="seconds before the rung is killed")
    ap.add_argument("--sampling_steps", type=int, default=20)
    ap.add_argument("--recycling_steps", type=int, default=None)
    ap.add_argument("--single_sequence", action="store_true", default=True)
    ap.add_argument("--msa", dest="single_sequence", action="store_false")
    ap.add_argument("--tag", default="")
    ap.add_argument("--task", choices=("predict", "affinity"), default="predict",
                    help="affinity scores a ligand and writes no structure, so it is judged on "
                         "its scalar rather than on a CIF")
    a = ap.parse_args()

    fixture = (WT / "perf" / "bh1536" / "fixtures" / f"aff_{a.size}.yaml" if a.task == "affinity"
               else WT / "perf" / "size512" / "fixtures" / f"cdk2x2_{a.size}.yaml")
    if not fixture.is_file():
        sys.exit(f"no fixture {fixture}")
    label = f"{a.model}_{a.size}" + (f"_{a.tag}" if a.tag else "")
    out = OUTROOT / "runs" / label
    if out.exists():
        shutil.rmtree(out)          # a stale results folder has been read as this run's before
    out.mkdir(parents=True)
    log = out / "fold.log"

    cmd = [PY, "-m", "tt_bio.main", a.task, str(fixture), "--model", a.model,
           "--out_dir", str(out), "--accelerator", "tenstorrent"]
    if a.task == "predict":
        cmd += ["--sampling_steps", str(a.sampling_steps), "--diffusion_samples", "1",
                "--seed", "0", "--output_format", "cif"]
        if a.single_sequence:
            cmd.append("--single_sequence")
    if a.recycling_steps is not None:
        cmd += ["--recycling_steps", str(a.recycling_steps)]

    env = dict(os.environ)
    env.update(PYTHONPATH=str(WT), TT_VISIBLE_DEVICES="0", TT_BIO_LEASE_CARDS="0",
               TT_BIO_LEASE_HOLDER="worker:bh-1536-structure")

    t0 = time.time()
    peak_rss, floor_avail = 0, 1 << 62
    with log.open("wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(WT), env=env)
        killed = False
        while proc.poll() is None:
            time.sleep(2)
            r, av = sample_host()
            peak_rss = max(peak_rss, r); floor_avail = min(floor_avail, av)
            if time.time() - t0 > a.budget:
                proc.kill(); killed = True; break
        proc.wait()
    wall = time.time() - t0
    text = log.read_text(errors="replace")

    # --- judge the artifact, never the exit code -------------------------------------------
    if a.task == "affinity":
        return _judge_affinity(a, out, text, wall, proc, killed, peak_rss, floor_avail)
    cifs = sorted(out.rglob("structures/*.cif")) + sorted(out.rglob("structures/*.pdb"))
    res_json = sorted(out.rglob("results.json"))
    nres = cif_residues(cifs[0]) if cifs else 0
    metrics = {}
    if res_json:
        try:
            metrics = json.loads(res_json[0].read_text())
        except Exception:
            metrics = {}
    if isinstance(metrics, list):          # boltz2 writes a list, one entry per target
        metrics = metrics[0] if metrics else {}
    ntok = None
    for k in ("n_tokens", "num_tokens", "tokens"):
        v = metrics.get(k) if isinstance(metrics, dict) else None
        if isinstance(v, int):
            ntok = v; break
    runtime_s = metrics.get("runtime_s") if isinstance(metrics, dict) else None
    plddt = next((metrics[k] for k in ("complex_plddt", "plddt", "mean_plddt")
                  if isinstance(metrics, dict) and isinstance(metrics.get(k), (int, float))), None)

    oom = _oom_of(text)
    # An L1 refusal that the blocking path retried past is not a failure; only count one
    # that the run did not survive.
    fatal_oom = oom if (oom and nres == 0) else None

    # A CIF that exists is not a fold. `perf/ceilings/struct_signal.py` is the repo's one
    # structural instrument (it imports the release gate's own thresholds), so a rung that
    # returns coordinates still has to show a continuous backbone before it counts as PASS.
    signal = {}
    if cifs:
        try:
            sig = subprocess.run([PY, str(WT / "perf/ceilings/struct_signal.py"), str(out)],
                                 capture_output=True, text=True, cwd=str(WT), env=env,
                                 timeout=1800)
            signal = json.loads(sig.stdout.strip().splitlines()[-1])
        except Exception as exc:
            signal = {"scored": 0, "error": str(exc)[:200]}

    ok = bool(cifs) and nres == a.size
    verdict = "PASS" if ok else ("TIMEOUT" if killed else ("OOM" if fatal_oom else "FAIL"))
    row = {"model": a.model, "size": a.size, "tag": a.tag, "task": "predict",
           "verdict": verdict, "wall_s": round(wall, 1), "engine_runtime_s": runtime_s,
           "cif": str(cifs[0]) if cifs else None, "cif_residues": nres,
           "n_tokens": ntok, "plddt": plddt, "struct": signal,
           "exit": proc.returncode, "killed": killed,
           "single_sequence": a.single_sequence, "sampling_steps": a.sampling_steps,
           "recycling_steps": a.recycling_steps,
           "peak_host_rss_gib": round(peak_rss / 2**30, 2),
           "floor_memavail_gib": round(floor_avail / 2**30, 2),
           "oom": oom, "fatal_oom": bool(fatal_oom),
           "tail": text[-1200:].replace("\n", " | ")[-1200:] if verdict != "PASS" else "",
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(row)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
