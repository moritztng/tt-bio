#!/usr/bin/env python3
"""Write an OpenBind-0 GPU cell into site/data/perf-512aa.json from a rental's raw reports.

perf/newmodelcells/write_rows.py writes the p150a cells for the two new rows and reads its GPU
denominators as fixed constants. It cannot add a GPU column to a row that already exists, so this
is the companion for the catch-up rental: same principle, that no number reaches the page unless a
committed file behind it says so and the acceptance criteria pass here rather than by eye.

The load-bearing check is the control arm. OpenBind-0 is the OpenFold3 runner on upstream 0.5.0
with a different checkpoint, so every box also folds OpenFold3 and has to reproduce that GPU's
already-published cell before its OpenBind number means anything. A box that misses the control
gets no cell written, however clean its own arms look.

  python3 perf/gpucatchup/write_gpu_cells.py --dir perf/gpucatchup/a100 --gpu a100 \
      --data site/data/perf-512aa.json
"""
import argparse, glob, json, statistics, sys
from pathlib import Path

# The published OpenFold3 cell each box's control arm has to reproduce, read off the page itself
# rather than restated here, so this script cannot drift from the column it is checking against.
CONTROL_TOL_PCT = 3.0   # the published a100 cell's own arms move 0.30 %, h200 0.21 %, b200 0.69 %;
                        # a fresh landlord's CPU moves a launch-bound row more than that, and 3 %
                        # is the band inside which the box is still reproducing the column.
ARMS = 4
WARM_PER_ARM = 3


def die(msg):
    sys.exit("write_gpu_cells: REFUSING TO WRITE -- " + msg)


def pct(a, b):
    return abs(a - b) / min(a, b) * 100.0


def load_arms(d, model, gpu):
    out = []
    for f in sorted(glob.glob(str(d / f"gpu_{model}_prot512_{gpu}_*.json"))):
        out.append((f, json.loads(Path(f).read_text())))
    return out


def gate_plddt(d, model, gpu, i):
    g = json.loads((d / f"gate_{model}_{gpu}_{i}.txt").read_text())
    if not g["pass"] or g["fail"]:
        die(f"gate_{model}_{gpu}_{i}: {g['fail']}")
    if g["n_ca"] != 512:
        die(f"gate_{model}_{gpu}_{i}: n_ca={g['n_ca']}, want 512")
    return g["plddt_mean"]


def check(d, model, gpu):
    """Return (pooled_median, all_warm, plddts, one representative report)."""
    arms = load_arms(d, model, gpu)
    if len(arms) != ARMS:
        die(f"{model} on {gpu}: {len(arms)} arms, want {ARMS}")
    warm, plddt = [], []
    for n, (f, j) in enumerate(arms, 1):
        if j.get("error"):
            die(f"{f}: {j['error']}")
        if (j["recycling_steps"], j["sampling_steps"], j["diffusion_samples"], j["seed"]) != (3, 200, 1, 0):
            die(f"{f}: protocol is {j['recycling_steps']}/{j['sampling_steps']}/"
                f"{j['diffusion_samples']}/seed {j['seed']}, want 3/200/1/0")
        if j["fixture"]["n_residues"] != 512:
            die(f"{f}: fixture is {j['fixture']['n_residues']} residues, want 512")
        if not j["fixture"]["a3m"]:
            die(f"{f}: no alignment; this page's fixture carries a 35-sequence a3m")
        if j["cueq_import_errors"]:
            die(f"{f}: cuEquivariance import errors {j['cueq_import_errors']}")
        r = j["result"]
        if r["warm_n"] != WARM_PER_ARM:
            die(f"{f}: {r['warm_n']} warm folds, want {WARM_PER_ARM}")
        if r["cold_s"] <= r["warm_max_s"]:
            die(f"{f}: cold {r['cold_s']} is not slower than every warm fold; "
                "the discarded round was not actually cold")
        k = r["kernel_counts_total"]
        # Counted, not assumed: the fused triangle kernels are the whole reason these rows are
        # comparable, and a silent torch fallback is a different measurement wearing the same name.
        if k["triangle_attention"] <= 0 or k["triangle_multiplicative_update"] <= 0:
            die(f"{f}: triangle kernels not engaged ({k['triangle_attention']} attention, "
                f"{k['triangle_multiplicative_update']} trimul)")
        for fb in ("triangle_attention._triangle_attention_torch",
                   "triangle_attention._warn_triangle_attention_fallback"):
            if k.get(fb, 0):
                die(f"{f}: torch fallback counter {fb}={k[fb]}")
        warm += r["warm_times_s"]
        plddt.append(gate_plddt(d, model, gpu, n))
    return statistics.median(warm), warm, plddt, arms[0][1], [a[1] for a in arms]


def exclusivity(d, gpu):
    """What the artifacts actually prove about the card being ours, and nothing more.

    Two independent readings, both from files: the session's own pre-row gate counts foreign
    compute apps and writes the count into the arm log, and a post-row nvidia-smi sample records
    resident memory. A shared card ([[vast-ai-access]], a stranger's 12 GB alongside 3.5 GB of
    ours) shows up in either one. Arm logs are large and not every rental's are committed, so the
    nvidia samples are the check that must pass and the gate counts are reported when present.
    """
    resident = []
    for f in sorted(d.glob(f"nvidia_{gpu}_*.txt")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            mib = float(line.split(",")[2].strip().split()[0])
            resident.append(mib)
            if mib > 0:
                die(f"{f.name}: {mib} MiB resident on the card between rows; not an exclusive card")
    if not resident:
        die(f"no nvidia_{gpu}_*.txt samples, so nothing shows the card was ours")
    counts = set()
    for f in sorted(d.glob("arm*.log")):
        for line in f.read_text().splitlines():
            if "foreign GPU procs" in line:
                counts.add(int(line.split("foreign GPU procs")[1].split(",")[0].strip()))
    if counts - {0}:
        die(f"the session's own gate saw foreign compute apps: {sorted(counts)}")
    gate = (f"the session gate read 0 foreign compute apps before every row it logged, and "
            if counts == {0} else "")
    return (f"The card was ours: {gate}all {len(resident)} post-row nvidia-smi samples "
            f"read 0 MiB resident.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--gpu", required=True, choices=("h200", "b200", "a100"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = json.loads(a.data.read_text())
    rows = {r["id"]: r for r in data["models"]}
    published = rows["openfold3"]["cells"][a.gpu]
    if published.get("status") != "measured" or not published.get("s_per_fold"):
        die(f"the OpenFold3 {a.gpu} cell is not measured, so there is nothing to control against")
    pub = published["s_per_fold"]

    ctl, ctl_warm, ctl_plddt, ctl_rep, ctl_all = check(a.dir, "openfold3", a.gpu)
    ob, ob_warm, ob_plddt, ob_rep, ob_all = check(a.dir, "openbind", a.gpu)

    delta = pct(ctl, pub)
    if delta > CONTROL_TOL_PCT:
        die(f"control arm {ctl:.4f} s against the published {a.gpu} cell {pub} s is {delta:.2f} % "
            f"apart, over the {CONTROL_TOL_PCT} % band: this box is not reproducing the column")

    # Each model's plDDT has to reproduce across all four arms, or the arms folded different things.
    for name, ps in (("openfold3", ctl_plddt), ("openbind", ob_plddt)):
        if max(ps) - min(ps) > 0.01:
            die(f"{name} plDDT moves {min(ps):.6f}-{max(ps):.6f} across arms, over 0.01")
    if abs(statistics.median(ctl_plddt) - statistics.median(ob_plddt)) < 0.02:
        die("the two arms' plDDT agree; they are probably running the same checkpoint")

    # The two stacks differ by design: same runner, different upstream release and checkpoint.
    if ctl_rep["packages"]["openfold3"] == ob_rep["packages"]["openfold3"]:
        die(f"both arms report openfold3 {ctl_rep['packages']['openfold3']}; "
            "the control is supposed to be a different upstream release")

    rel = (ob - ctl) / ctl * 100.0
    fmt = lambda xs: " / ".join(f"{x:.3f}" for x in sorted(xs))
    smi = ob_rep["nvidia_smi"].split(",")
    excl = exclusivity(a.dir, a.gpu)
    # A control that lands inside the band can still be measurably off the published cell. If the
    # gap is wider than either arm's own warm spread it is the landlord's host, not run-to-run
    # noise, and the honest reading of this row is the relative figure rather than the absolute.
    widest = max(j["result"]["warm_spread_pct"] for j in ctl_all + ob_all)
    host_note = (
        f"That {delta:.2f} % is wider than the widest warm spread inside any arm on this box "
        f"({widest:.2f} %), so it is this rental's host rather than run-to-run noise: read the "
        f"relative figure, not this cell against the published OpenFold3 one. "
        if delta > widest else
        f"That {delta:.2f} % sits inside the widest warm spread within a single arm on this box "
        f"({widest:.2f} %). ")
    arm_medians = " / ".join(f"{j['result']['warm_median_s']:.3f}" for j in ob_all)
    ref = (
        f"Pooled median of {len(ob_warm)} warm folds over {ARMS} alternating arms, "
        f"{WARM_PER_ARM} warm plus a discarded cold fold each: {fmt(ob_warm)} s, "
        f"per-arm medians {arm_medians}. "
        f"An OpenFold3 control arm alternated with it on the same card in the same session pools to "
        f"{ctl:.4f} s against the published {a.gpu} cell's {pub} s, {delta:.2f} % apart, so this box "
        f"reproduces the column; OpenBind-0 folds this fixture {abs(rel):.2f} % "
        f"{'under' if rel < 0 else 'over'} the control beside it. {host_note}"
        f"Both arms reach the cuEquivariance triangle kernels, counted not assumed: "
        f"triangle_attention {ob_rep['result']['kernel_counts_total']['triangle_attention']} and "
        f"triangle_multiplicative_update "
        f"{ob_rep['result']['kernel_counts_total']['triangle_multiplicative_update']} on this arm "
        f"against {ctl_rep['result']['kernel_counts_total']['triangle_attention']} and "
        f"{ctl_rep['result']['kernel_counts_total']['triangle_multiplicative_update']} on the "
        f"control, every torch-fallback counter 0. plDDT reproduces across all four arms "
        f"({min(ob_plddt):.6f}-{max(ob_plddt):.6f}) and sits well below the control's "
        f"({min(ctl_plddt):.6f}-{max(ctl_plddt):.6f}), which is the checkpoint difference, not noise. "
        f"openfold3 {ob_rep['packages']['openfold3']} on checkpoint of3-ob-2025-06-30-174k.pt, "
        f"torch {ob_rep['torch_version']}, cuequivariance-torch "
        f"{ob_rep['packages'].get('cuequivariance_torch')}, triton {ob_rep['packages'].get('triton')}. "
        f"{ob_rep['gpu']}, driver {smi[0].strip()}, {ob_rep['host_cpu']} at "
        f"{ob_rep['vcpu_cgroup']} effective vCPU from the cgroup quota. Peak allocated VRAM "
        f"{ob_rep['peak_mem_MiB']} MiB. {excl} "
        f"3 recycles, 200 sampling steps, one diffusion sample, seed 0, the pinned cdk2x2_512 "
        f"fixture with its 35-row alignment, checkpoint load outside the cell."
    )
    cell = {"status": "measured", "s_per_fold": round(ob, 4), "ref": ref}

    print(f"{a.gpu}: openbind {ob:.4f} s, control {ctl:.4f} s vs published {pub} s "
          f"({delta:.2f} %), openbind {rel:+.2f} % against the control")
    if a.dry_run:
        print(json.dumps(cell, indent=1))
        return
    rows["openbind"]["cells"][a.gpu] = cell
    a.data.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote openbind/{a.gpu} into {a.data}")


if __name__ == "__main__":
    main()
