"""Table of one kitcmp session: per page cell x mode, warm median, n, min-max, cold, rc, the kit's ACTIVE line, the timed
function, and the SM clock / power / memory nvidia-smi read during the arm (samples above 100 W only).

Kit arms (results/<model>/<mode>/) are timed by kctime.py (times.jsonl); RoseTTAFold3 by the gaps between consecutive structure
dumps; RFdiffusion3 by the engine's own per-batch line; ESMC by esmc_gpu.py's records. Stock arms (results/stock-<row>/) are read
from the report of the tt-bio harness that ran them (bgg_bench.py, gpu_embed_bench.py, gpu_nesso1_run.py).
Units: fold s/prediction, design s/design, affinity s/prediction, embed s/batch.

    python3 perf/kitcmp/parse.py perf/kitcmp/results/<label>/results [--json]
"""
import csv, json, re, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

# result dir -> BOARD cell id
CELL = {"boltz2": "boltz2", "protenix_v2": "protenix-v2", "protenix_v1": "protenix-v1", "opendde": "opendde",
        "openfold3": "openfold3", "openfold3_ob0": "openbind", "esmfold2-full_nomsa": "esmfold2",
        "esmfold2-fast": "esmfold2-fast", "rfdiffusion3": "rfd3", "pxdesign": "pxdesign"}


def ts(p):
    return datetime.strptime(p.read_text().strip(), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


def smi(path):
    out = []
    if path.exists():
        for r in csv.reader(path.open()):
            try:
                t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                out.append((t, float(r[1].split()[0]), float(r[3].split()[0]), float(r[8].split()[0]) / 1024, float(r[6].split()[0])))
            except (ValueError, IndexError):
                pass
    return out


def clocks(samples, d, folds=()):
    """SM MHz median / min, W median, GB max during the warm folds ((start, end) epochs), else over the arm's window with the
    GPU busy (utilization >= 50 %; a power floor does not work: an idle B200 draws ~197 W)."""
    if not samples or not (d / "t_start").exists() or not (d / "t_end").exists():
        return None, None, None, None
    if folds:
        w = [x for x in samples if any(a <= x[0] <= b for a, b in folds)]
    else:
        a, b = ts(d / "t_start"), ts(d / "t_end")
        w = [x for x in samples if a <= x[0] <= b and x[4] >= 50]
    if not w:
        return None, None, None, None
    c = [x[1] for x in w]
    return statistics.median(c), min(c), statistics.median(x[2] for x in w), max(x[3] for x in w)


def row(cell, mode, warm, cold, rc, fn, clk, engaged="-", **extra):
    return dict(cell=cell, mode=mode, warm=[round(x, 4) for x in warm], n=len(warm),
                median=round(statistics.median(warm), 4) if warm else None, cold=cold, rc=rc, fn=fn,
                sm_mhz=clk[0], sm_min=clk[1], watts=clk[2], mem_gb=clk[3], engaged=engaged, **extra)


def kit_arms(m, samples):
    for mode in ("off", "exact", "fast", "off-det", "exact-det"):
        d = m / mode
        if not d.exists():
            continue
        rc = (d / "rc").read_text().strip() if (d / "rc").exists() else "running"
        if (d / "SERVER-MSA").exists() or any(d.glob("pred/*/msa/**/msa.sh")):
            rc += " SERVER-MSA"
        log = "".join(f.read_text(errors="replace") for f in (d / "stderr.log", d / "stdout.log") if f.exists())
        act = next((l.strip()[:90] for l in log.splitlines() if "ACTIVE" in l or l.startswith("stock:")), "-")
        if m.name == "esmc":
            clk = clocks(samples, d)  # one record per variant, warm list inside
            for f in sorted(d.glob("esmc-*.jsonl")):
                r = json.loads(f.read_text().splitlines()[-1])
                yield row(f"esmc-{r['variant']}", mode, r.get("warm_s", []), round(r["cold_s"], 4), rc, "esmc_gpu.forward",
                          clk, act, digest=r["pooled_digest"], batch=r["batch"])
            continue
        rows = [json.loads(l) for l in (d / "times.jsonl").read_text().splitlines()] if (d / "times.jsonl").exists() else []
        s = [r["s"] for r in rows]
        fn = ".".join(rows[0]["fn"].rsplit(".", 2)[-2:]) if rows else "-"
        if m.name.startswith("rosettafold3") and rows:  # wrapped RF3Output.dump inside one engine.run: fold i = t[i] - t[i-1]
            t = [r["t"] for r in rows]
            s = [round(t[0] - ts(d / "t_start"), 3)] + [round(b - a, 4) for a, b in zip(t, t[1:])]
        if m.name == "rfdiffusion3":  # the engine's own per-batch line, wrapping the sampler loop
            s = [float(x) for x in re.findall(r"Finished inference batch in ([\d.]+) seconds", log)]
            fn = "engine-log"
        # warm fold i ends at its record's t and lasts s[i] (rfdiffusion3's log lines carry no time: the arm window)
        folds = [(r["t"] - x, r["t"]) for r, x in zip(rows[1:], s[1:])] if rows and m.name != "rfdiffusion3" else ()
        clk = clocks(samples, d, folds)
        cell = "rf3" if m.name.startswith("rosettafold3") else CELL.get(m.name, m.name)
        yield row(cell, mode, s[1:], s[0] if s else None, rc, fn, clk, act, arm=m.name)


def stock_arms(m, samples):
    rc = (m / "rc").read_text().strip() if (m / "rc").exists() else "running"
    clk = clocks(samples, m)
    what = m.name[len("stock-"):]
    if what == "boltzgen":
        for f in m.glob("bgg_*.jsonl"):
            r = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            r = [x for x in r if x.get("arm") == "headline" and x.get("batch") == 1]
            if r:
                x = r[-1]
                yield row("boltzgen", "stock", x.get("s_per_design_all", []), x.get("cold_loop_s"), rc if x.get("ok", True) else "not-ok",
                          "Boltz.predict_step", clk, f"batch {x['batch']}, {x.get('n_warm_loops', '?')} warm loops")
    elif what == "saprot":
        for f in sorted(m.glob("gpu_embed_saprot-*.json")):
            j = json.loads(f.read_text())
            s = next((v for v in _walk(j) if "warm_times_s" in v), {})
            yield row(f.name.split("_")[2], "stock", s.get("warm_times_s", []), s.get("cold_s"), rc, "forward", clk,
                      f"sdpa engaged={next((v['attn_engaged'] for v in _walk(j) if 'attn_engaged' in v), '?')}")
    elif what == "nesso1":
        legs = [json.loads(f.read_text()) for f in sorted(m.glob("nesso1_ladder_aa512_*.json"))]
        for k, mode in (("cueq", "stock"), ("torch", "stock-nokernels")):
            ls = [j for j in legs if j["label"] == f"ladder_aa512_{k}"]
            if ls:
                warm = [x for j in ls for x in j["rep_s"][1:]]
                yield row("nesso1", mode, warm, ls[0]["rep_s"][0], rc if all(j["ok"] for j in ls) else "not-ok",
                          "invocation-wall", clk, f"{len(ls)} legs, affinity sha {sorted({j['affinity']['sha256_of_values'][:12] for j in ls})}")


def _walk(j):
    if isinstance(j, dict):
        yield j
        for v in j.values():
            yield from _walk(v)
    elif isinstance(j, list):
        for v in j:
            yield from _walk(v)


def arms(root):
    root = Path(root)
    for m in sorted(p for p in root.iterdir() if p.is_dir()):
        if m.name.startswith("stock-"):
            yield from stock_arms(m, smi(m / "smi.csv"))
        else:  # esmfold2-full_nomsa / esmfold2-fast: one box_arms.sh run, its clock trace sits in esmfold2/
            yield from kit_arms(m, smi(next((d / "smi.csv" for d in (m, root / m.name.split("-")[0]) if (d / "smi.csv").exists()),
                                            m / "smi.csv")))


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    rs = list(arms(sys.argv[1]))
    if "--json" in sys.argv:
        print(json.dumps(rs, indent=1))
        sys.exit(0)
    print("| cell | mode | warm median | n | min-max | cold | rc | timed | SM MHz med (min) | W med | mem GB max | engaged |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        rng = f"{min(r['warm']):.3f}-{max(r['warm']):.3f}" if r["warm"] else "-"
        clk = f"{r['sm_mhz']:.0f} ({r['sm_min']:.0f})" if r["sm_mhz"] else "-"
        print(f"| {r['cell']} | {r['mode']} | {fmt(r['median'])} | {r['n']} | {rng} | {r['cold'] if r['cold'] is not None else '-'} | "
              f"{r['rc']} | {r['fn']} | {clk} | {fmt(r['watts'], 0)} | {fmt(r['mem_gb'], 1)} | {r['engaged']} |")
