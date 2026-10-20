"""BC4 rule 4 on the box: every shipped BindCraft 2 / JAX setting that does the SAME work, ranked, then the fastest
one rerun long enough to be a result. Each arm is one bc2_gpu.py campaign (PD-L1, binder 80, seed 101, one trajectory
at a time); nothing here touches iteration count, stages, recycles, models or MPNN sequences, so every arm does the
same work and the iteration distribution is checked, not assumed.

    python bc4_arms.py --bc2 /root/bc2 --out /root/kc/bc4 [--screen-warm 2] [--final-warm 9] [--deadline <epoch>]

Writes <out>/arms/<arm>/runs.jsonl per arm, <out>/winner/runs.jsonl for the headline run and <out>/ARMS.json with
every arm's median seconds per design iteration, n, spread, SM clock and iteration distribution. The losers stay.
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# arm -> (BindCraft 2 settings, BC2_XLA_EXTRA). Attention arms are mutually exclusive; 'stock' resolves to cudnn.
ARMS = [
    ("stock", [], None),
    ("cueq", ["use_cueq=true"], None),
    ("triton", [], "--xla_gpu_enable_triton_gemm=true"),
    ("nosubbatch", ["subbatch_size=null"], None),
    ("attn-xla", ["attention_backend=xla"], None),
    ("attn-stock", ["attention_backend=stock"], None),
]
ATTENTION = {"attn-xla", "attn-stock"}

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--smi", type=Path, help="the 1 Hz nvidia-smi log to link into every arm's directory")
ap.add_argument("--screen-warm", type=int, default=2)
ap.add_argument("--final-warm", type=int, default=9)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--python", default="/opt/bc2-venv/bin/python")
ap.add_argument("--deadline", type=float, default=0, help="epoch seconds; stop starting arms that cannot finish")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


def log(*x):
    print(f"{time.strftime('%FT%TZ', time.gmtime())} bc4_arms:", *x, flush=True)


def run(arm, settings, xla, warm, d):
    """One campaign. Returns the arm's summary over its admissible warm trajectories."""
    d.mkdir(parents=True, exist_ok=True)
    if a.smi:
        (d / "smi.csv").unlink(missing_ok=True)
        (d / "smi.csv").symlink_to(a.smi)
    env = dict(os.environ)
    if xla:
        env["BC2_XLA_EXTRA"] = xla
    else:
        env.pop("BC2_XLA_EXTRA", None)
    cmd = [a.python, str(HERE / "bc2_gpu.py"), "--bc2", a.bc2, "--out", str(d), "--seed", str(a.seed),
           "--warm", str(warm), "--binder-len", "80"]
    for s in settings:
        cmd += ["--set", s]
    t0 = time.time()
    with open(d / "stdout.log", "w") as o, open(d / "stderr.log", "w") as e:
        rc = subprocess.call(cmd, env=env, stdout=o, stderr=e)
    (d / "rc").write_text(f"{rc}\n")
    s = summarise(arm, d, xla, settings)
    s["wall_s"], s["rc"] = round(time.time() - t0, 1), rc
    log(f"arm {arm}: rc={rc} n={s['n']} median={s['median']} wall={s['wall_s']}s")
    return s


def summarise(arm, d, xla, settings):
    reps, start = [], {}
    for line in (d / "runs.jsonl").read_text().splitlines() if (d / "runs.jsonl").exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ev") == "start":
            start = r
        if r.get("ev") == "rep" and r.get("kind") == "warm" and not r.get("err") \
                and (r.get("design_s") or 0) > 0 and (r.get("iterations") or 0) > 0:
            reps.append(r)
    # protocol rule 4: a trajectory whose SM clock fell below 1800 MHz is not admissible
    adm = [r for r in reps if not (r.get("sm") or {}).get("median") or r["sm"]["median"] >= 1800]
    ts = sorted(r["design_s"] / r["iterations"] for r in adm)
    sm = sorted((r.get("sm") or {}).get("median") for r in adm if (r.get("sm") or {}).get("median"))
    return dict(arm=arm, settings=settings, xla=xla, evidence=str(d / "runs.jsonl"), n=len(ts),
                dropped_low_clock=len(reps) - len(adm),
                median=round(statistics.median(ts), 4) if ts else None,
                spread=[round(ts[0], 4), round(ts[-1], 4)] if ts else None,
                iterations=sorted({r["iterations"] for r in adm}),
                terminated=sorted({str(r.get("terminated")) for r in adm}),
                design_s=[round(r["design_s"], 2) for r in adm],
                sm_mhz=sm[len(sm) // 2] if sm else None,
                gpu=start.get("gpu"), jax=start.get("jax"), bc2_sha=start.get("bc2_sha"),
                cpu=start.get("cpu"), ncpu=start.get("ncpu"))


def save():
    (a.out / "ARMS.json").write_text(json.dumps(res, indent=1) + "\n")


def reserve():
    """Seconds to keep for the headline run, from the stock arm's measured pace."""
    s = res.get("stock") or {}
    per = (s.get("median") or 0.4) * max(s.get("iterations") or [140])
    return per * (a.final_warm + 1) * 1.5 + 300


def ok(s):
    """An arm is comparable only if it completed and did the same work as stock (same iteration counts)."""
    st = res.get("stock") or {}
    return s and s.get("rc") == 0 and s.get("n") and s.get("median") and \
        (not st.get("iterations") or s["iterations"] == st["iterations"])


res, skipped = {}, []
for arm, settings, xla in ARMS:
    if a.deadline and res.get("stock", {}).get("wall_s") and \
            time.time() + res["stock"]["wall_s"] * 1.3 + reserve() > a.deadline:
        log(f"skipping arm {arm}: no time left before the deadline")
        skipped.append(arm)
        continue
    res[arm] = run(arm, settings, xla, a.screen_warm, a.out / "arms" / arm)
    save()

# the stack: every lever that beat stock on its own, together, and at most one attention backend (the fastest).
base = (res.get("stock") or {}).get("median")
wins = [n for n in res if n != "stock" and ok(res[n]) and base and res[n]["median"] < base * 0.99]
attn = [n for n in wins if n in ATTENTION]
if len(attn) > 1:
    slowest = sorted(attn, key=lambda n: res[n]["median"])[1:]
    wins = [n for n in wins if n not in slowest]
if len(wins) > 1:
    settings = [s for n, st, _ in ARMS if n in wins for s in st]
    xla = " ".join(x for n, _, x in ARMS if n in wins and x) or None
    if not a.deadline or time.time() + (res["stock"]["wall_s"] * 1.3 if res.get("stock") else 0) + reserve() <= a.deadline:
        res["best"] = run("best", settings, xla, a.screen_warm, a.out / "arms" / "best")
        res["best"]["stacked"] = wins
        save()
    else:
        log(f"no time to screen the stack {wins}")

ranked = sorted((n for n in res if ok(res[n])), key=lambda n: res[n]["median"])
log("ranking:", [(n, res[n]["median"]) for n in ranked])
res["_skipped"], res["_ranking"] = skipped, ranked
save()

if ranked:
    w = res[ranked[0]]
    log(f"winner {w['arm']} at {w['median']} s/it; rerunning at --warm {a.final_warm}")
    res["_winner"] = run(w["arm"], w["settings"], w["xla"], a.final_warm, a.out / "winner")
    res["_winner"]["arm"] = w["arm"]
    res["_winner"]["stacked"] = w.get("stacked")
    save()
log("done")
