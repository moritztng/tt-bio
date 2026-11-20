"""Startup-time report from a perf/startup/run.sh output dir: phase table (stdout, JSON) and a 2-page PDF.

    python perf/startup/report.py RUNDIR OUT.pdf

Per run (one fresh process) the tracer's summary is turned into phases, from process start to the end of the first
prediction. Per model, cold and warm runs of the same round are paired: kernel compile = cold total - warm total
(the only difference between the two runs is the empty TT_METAL_CACHE). The cold bar is the warm run's phases plus
that compile segment, medians over rounds. Steady = the second prediction in the same process.
"""
import json, statistics, sys
from pathlib import Path

RUN, PDF = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = {"esmc": "ESMC-300M", "esmfold2": "ESMFold2", "boltz2": "Boltz-2", "protenix-v2": "Protenix-v2",
         "opendde": "OpenDDE", "openfold3": "OpenFold3", "rfd4": "RFD4", "bindcraft2": "BindCraft 2"}
UNIT = {"rfd4": "design", "bindcraft2": "trajectory", "esmc": "embed"}
# bar segments, in stacking order (categorical slots 1..8)
SEG = ["imports", "device", "weights_read", "weights_upload", "setup", "compile", "first_extra", "compute"]
LABEL = {"imports": "Python imports", "device": "Device open", "weights_read": "Checkpoint check + disk read",
         "weights_upload": "Weight prep + upload", "setup": "Other setup + featurize",
         "compile": "Kernel compile (cold cache)", "first_extra": "First-call extra (trace capture, caches)",
         "compute": "Prediction compute"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def summary(d):
    p = d / "trace.jsonl"
    if not p.exists():
        return None
    for line in p.open():
        e = json.loads(line)
        if e.get("ev") == "summary":
            return e
    return None


def phases(s):
    """Fine phases (s) of one run up to the end of its first prediction, plus steady and totals."""
    tops = {t[0]: t for t in s["tops"]}
    if "job#1" not in tops or tops["job#1"][2] is None:
        return None
    share = {}
    for top, cls, n in s["samples"]:
        share.setdefault(top, {}).setdefault(cls, 0)
        share[top][cls] += n
    f = dict(import_torch=0, import_ttnn=0, import_other=s["age_at_site"], device=0, ckpt=0, read=0, upload=0,
             build=0, setup=0, featurize=0, trace=0, first=0)
    for top, cat, sec in s["acc"]:
        if top not in ("pre", "device_open", "load", "job#1"):
            continue
        if cat == "import_torch":
            f["import_torch"] += sec
        elif cat == "import_ttnn":
            f["import_ttnn"] += sec
        elif cat.startswith("import_"):
            f["import_other"] += sec
        elif cat in ("device", "device_open"):
            f["device"] += sec
        elif cat in ("ckpt", "read", "featurize"):
            f[cat] += sec
        elif cat == "trace_capture":
            f["trace"] += sec
        elif cat == "upload":
            f["first" if top == "job#1" else "upload"] += sec
        else:  # time no inner wrap claimed: split by the stack sampler into imports and the phase's own work
            sh = share.get(top, {})
            imp = sh.get("import", 0) / max(1, sum(sh.values()))
            f["import_other"] += sec * imp
            f[{"pre": "setup", "load": "build", "job#1": "first", "device_open": "device"}[top]] += sec * (1 - imp)
    # a model that loads weights lazily inside its first prediction (ESMFold2's 6B LM): the upload prediction 1
    # does beyond prediction 2's is weight upload, not prediction work
    up = {t: sum(sec for tt, cat, sec in s["acc"] if tt == t and cat == "upload") for t in ("job#1", "job#2")}
    if "job#2" in tops:
        lazy = max(0.0, up["job#1"] - up["job#2"])
        f["first"] -= lazy
        f["upload"] += lazy
    total = tops["job#1"][2]
    f["setup"] += max(0.0, total - sum(f.values()))   # interpreter bookkeeping the tracer did not see
    j2 = tops.get("job#2")
    return dict(fine=f, total=total, steady=(j2[2] - j2[1]) if j2 and j2[2] else None,
                first_job=tops["job#1"][2] - tops["job#1"][1])


def coarse(f, steady):
    c = dict(imports=f["import_torch"] + f["import_ttnn"] + f["import_other"], device=f["device"],
             weights_read=f["ckpt"] + f["read"], weights_upload=f["upload"] + f["build"],
             setup=f["setup"] + f["featurize"])
    first = f["first"] + f["trace"]
    c["compute"] = min(first, steady) if steady else first
    c["first_extra"] = first - c["compute"]
    return c


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


models, rows = [m for m in NAMES if (RUN / m).is_dir()], {}
for m in models:
    pairs = []
    for r in sorted({p.name.split("-")[0] for p in (RUN / m).iterdir() if p.is_dir()}):
        sc, sw = summary(RUN / m / f"{r}-cold"), summary(RUN / m / f"{r}-warm")
        pc, pw = (phases(sc) if sc else None), (phases(sw) if sw else None)
        if pc and pw:
            pairs.append((r, pc, pw))
    if not pairs:
        continue
    if any(r != "r1" for r, _, _ in pairs):  # r1 cold ran before any weight page-cache drop; r2+ drop both
        pairs = [p for p in pairs if p[0] != "r1"]
    fine = {k: med([pw["fine"][k] for _, _, pw in pairs]) for k in pairs[0][2]["fine"]}
    steady = med([pw["steady"] for _, _, pw in pairs] + [pc["steady"] for _, pc, _ in pairs])
    c = coarse(fine, steady)
    c["compile"] = med([max(0.0, pc["total"] - pw["total"]) for _, pc, pw in pairs])
    rows[m] = dict(n=len(pairs), rounds=[r for r, _, _ in pairs], fine=fine, coarse=c, steady=steady,
                   cold_total=med([pc["total"] for _, pc, _ in pairs]), warm_total=med([pw["total"] for _, _, pw in pairs]),
                   cold_first_job=med([pc["first_job"] for _, pc, _ in pairs]),
                   warm_first_job=med([pw["first_job"] for _, _, pw in pairs]),
                   cold_device=med([pc["fine"]["device"] for _, pc, _ in pairs]),
                   cold_runs=[round(pc["total"], 1) for _, pc, _ in pairs], warm_runs=[round(pw["total"], 1) for _, _, pw in pairs])
json.dump(rows, open(PDF.with_suffix(".json"), "w"), indent=1)
for m, r in rows.items():
    print(f"{m:12s} n={r['n']} cold={r['cold_total']:.1f} warm={r['warm_total']:.1f} steady={r['steady'] or 0:.1f} "
          + " ".join(f"{k}={v:.1f}" for k, v in r["coarse"].items()))
if len(sys.argv) > 3 and sys.argv[3] == "--no-pdf":
    sys.exit(0)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

META = json.loads((RUN / "meta.json").read_text()) if (RUN / "meta.json").exists() else {}
TAKE = (RUN / "takeaways.txt").read_text().strip().split("\n") if (RUN / "takeaways.txt").exists() else []
INK, INK2, INK3, SURF = "#0b0b0b", "#52514e", "#8a8984", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})
order = sorted(rows, key=lambda m: rows[m]["cold_total"])
with PdfPages(PDF) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.955, "Startup time on Tenstorrent, 8 models", fontsize=17, weight="bold")
    fig.text(0.06, 0.933, META.get("subtitle", ""), fontsize=9, color=INK2)
    ax = fig.add_axes([0.2, 0.47, 0.62, 0.42])
    xmax = max(rows[m]["cold_total"] for m in order)
    for i, m in enumerate(order):
        left = 0.0
        for k, col in zip(SEG, COLORS):
            w = rows[m]["coarse"][k]
            if w > 0:
                ax.barh(i, w, left=left, height=0.62, color=col, edgecolor=SURF, linewidth=1.2)
            left += w
        ax.text(left + xmax * 0.012, i + 0.06, f"{rows[m]['cold_total']:.0f} s", va="center", fontsize=10, weight="bold")
        ax.text(left + xmax * 0.012, i - 0.30, f"warm cache {rows[m]['warm_total']:.0f} s", va="center", fontsize=7.5,
                color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{NAMES[m]}" for m in order], fontsize=10)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("seconds from process start to first result ready (cold kernel cache, median of runs)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d7d6d1")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, [LABEL[k] for k in SEG], loc="upper left", bbox_to_anchor=(0.05, 0.925), ncol=2, frameon=False,
               fontsize=8.5, handlelength=1.2, columnspacing=1.5)
    y = 0.405
    fig.text(0.06, y, "Takeaways", fontsize=12, weight="bold")
    import textwrap
    for t in TAKE:
        y -= 0.012
        for j, line in enumerate(textwrap.wrap(t, 95)):
            y -= 0.017
            fig.text(0.075 if j else 0.06, y, ("• " if j == 0 else "") + line, fontsize=9.5)
    fig.text(0.06, 0.025, META.get("footer", ""), fontsize=7, color=INK3)
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.955, "Per-phase seconds (medians) and method", fontsize=14, weight="bold")
    cols = ["torch\nimport", "ttnn\nimport", "other\nimports", "device\nopen", "ckpt\ncheck", "disk\nread",
            "upload", "host\nbuild", "setup +\nfeaturize", "trace\ncapture", "kernel\ncompile", "1st pred\n(warm)",
            "steady\npred", "warm\ntotal", "cold\ntotal", "n"]
    cell = []
    for m in order:
        r, f = rows[m], rows[m]["fine"]
        vals = [f["import_torch"], f["import_ttnn"], f["import_other"], f["device"], f["ckpt"], f["read"], f["upload"],
                f["build"], f["setup"] + f["featurize"], f["trace"], r["coarse"]["compile"],
                r["warm_first_job"], r["steady"] or 0, r["warm_total"], r["cold_total"]]
        cell.append([f"{v:.1f}" for v in vals] + [str(r["n"])])
    ax = fig.add_axes([0.03, 0.62, 0.94, 0.3])
    ax.axis("off")
    tb = ax.table(cellText=cell, rowLabels=[NAMES[m] for m in order], colLabels=cols, loc="upper center",
                  cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 2.0)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#e4e3df")
        if i == 0:
            c.set_text_props(color=INK2, fontsize=6.5)
            c.set_height(c.get_height() * 1.5)
    method = META.get("method", "")
    y = 0.58
    for para in method.split("\n"):
        for line in textwrap.wrap(para, 118) or [""]:
            fig.text(0.06, y, line, fontsize=7.6, color=INK2)
            y -= 0.0145
        y -= 0.004
    pdf.savefig(fig)
    plt.close(fig)
print("wrote", PDF)
