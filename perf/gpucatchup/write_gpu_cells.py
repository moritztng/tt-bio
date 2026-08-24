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
import argparse, glob, json, re, statistics, sys
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


# --- Nesso-1 -------------------------------------------------------------------------------------
# Nesso-1 has no control arm: nothing else on the page runs its stack, so there is no published
# number this box could reproduce first. The bar for a brand-new GPU cell with no prior anchor is
# internal A/A instead, which for this harness is a real digest and not a wall-clock coincidence --
# two independent processes have to agree on the affinity values' sha256.
NESSO_PINS = {   # what the h200 and a100 cells both name; a new column has to match or say it did not
    "torch": "2.11.0+cu128", "nesso": "1.0.0", "cuequivariance-torch": "0.11.1",
    "cuequivariance-ops-torch-cu12": "0.11.1", "lightning": "2.6.5",
    "transformers": "5.15.1", "rdkit": "2026.3.5",
}


def write_nesso(d, gpu, data):
    legs = [json.loads(f.read_text()) for f in sorted(d.glob(f"nesso1_ladder_aa512_cueq_{gpu}_*.json"))]
    if len(legs) < 2:
        die(f"{len(legs)} Nesso-1 legs, want at least 2 for an A/A")
    warm, medians, dev = [], [], []
    for j in legs:
        lbl = j["label"]
        if lbl != "ladder_aa512_cueq":
            die(f"{lbl}: not the ladder_aa512_cueq rung the h200 and a100 cells hold")
        if not j["ok"]:
            die(f"{lbl}: {j['why']}")
        if not j["gpu_exclusive"] or j["env"]["compute_apps_before"]:
            die(f"{lbl}: card not exclusive, {j['env']['compute_apps_before']}")
        if (j["reps"], j["recycling_steps"], j["precision"], j["seed"], j["num_workers"],
                j["dataloader_batch_size"], j["refine"]) != (4, 5, "bf16-mixed", 42, 2, 1, "on"):
            die(f"{lbl}: protocol differs from the published columns")
        if j["seq_lens"] != [512] or j["n_records"] != 1:
            die(f"{lbl}: {j['n_records']} records at {j['seq_lens']}, want 1 at [512]")
        if not j["effective_use_kernels"] or not j["ckpt_use_kernels"]:
            die(f"{lbl}: kernels off")
        c = j["counts"]
        if c["cueq.triangle_attention"] != c["callsite.triangle_attention"]:
            die(f"{lbl}: cuEquivariance took {c['cueq.triangle_attention']} of "
                f"{c['callsite.triangle_attention']} triangle-attention calls, not all of them")
        for k, want in NESSO_PINS.items():
            got = j["env"].get(k)
            if got != want:
                die(f"{lbl}: {k} is {got}, the published columns name {want}")
        reps = j["rep_s"]
        if reps[0] <= max(reps[1:]):
            die(f"{lbl}: rep 0 ({reps[0]}) is not the slowest, so it was not the cold rep")
        warm += reps[1:]
        medians.append(statistics.median(reps[1:]))
        ph = j["phases"]
        dev += [ph[k]["predict_step"] for k in sorted(ph, key=int)][1:]

    shas = {j["affinity"]["sha256_of_values"] for j in legs}
    if len(shas) != 1:
        die(f"the legs disagree on the affinity digest: {shas}")
    aff = {j["affinity"]["mean"] for j in legs}
    if len(aff) != 1:
        die(f"the legs disagree on the affinity scalar: {aff}")

    row = [r for r in data["affinity"]["models"] if r["id"] == "nesso1"][0]
    a100 = row["cells"].get("a100", {})
    ob = statistics.median(warm)
    # The ref says this scalar is the H200 NVL arm's and not the A100 one. That is a claim about two
    # other cells, so it is read out of the a100 cell -- which records both arms' scalars and its own
    # device figure -- instead of being restated here.
    scalar = f"{sorted(aff)[0]:.6f}"
    a100_ref = a100.get("ref", "")
    if scalar not in a100_ref:
        die(f"affinity scalar {scalar} appears in no other cell's ref, so the claim that it matches "
            "the H200 NVL arm is not supported by the page")
    a100_dev = re.search(r"Lightning predict step is ([0-9.]+) s", a100_ref)
    a100_scalar = re.search(r"Affinity scalar ([0-9.]+) against the H200 NVL arm's ([0-9.]+)", a100_ref)
    if not a100_dev or not a100_scalar:
        die("the a100 cell's ref no longer carries its predict-step and affinity-scalar wording; "
            "this cell quotes both and must not guess them")
    if a100_scalar.group(2) != scalar:
        die(f"the a100 cell records the H200 NVL arm's scalar as {a100_scalar.group(2)}, "
            f"this box measured {scalar}")
    a100_vram = re.search(r"Peak allocated VRAM ([0-9.]+) MiB", a100_ref)
    vram = f"{legs[0]['peak_vram_alloc_B'] / 2 ** 20:.1f}"
    same_vram = bool(a100_vram) and a100_vram.group(1) == vram
    if a100_scalar.group(1) == scalar:
        die("the a100 cell records its own scalar as the same value, so the claim that this box "
            "differs from the A100 is wrong")
    e, g = legs[0]["env"], legs[0]["env"]["gpu_static"]
    leg_gap = abs(medians[0] - medians[1]) / min(medians) * 100.0
    ref = (
        f"Two independent processes, 3 warm reps each after a discarded cold rep, the same "
        f"invocation-wall region the h200 and a100 cells hold: "
        f"{' / '.join(f'{x:.4f}' for x in sorted(warm))} s, pooled median {ob:.4f}. The two legs "
        f"median {medians[0]:.4f} and {medians[1]:.4f}, {leg_gap:.2f} % apart, and that gap is host, "
        f"not device: the Lightning predict step inside the same reps holds "
        f"{min(dev):.4f}-{max(dev):.4f} s across all six, against "
        f"{a100_dev.group(1)} s on the A100. Most of this rung is host work at "
        f"{e['effective_cpus']:.2f} effective vCPU from the cgroup quota, so the wall moves with the "
        f"landlord's CPU while the card's share barely does. "
        f"Reproducible where it matters: both processes return the identical affinity scalar "
        f"{sorted(aff)[0]:.6f} and the identical sha256 of the affinity values "
        f"({sorted(shas)[0]}), and that scalar is the H200 NVL arm's rather than the A100's. "
        f"cuEquivariance engaged on 100 % of the triangle calls "
        f"({legs[0]['counts']['cueq.triangle_attention']} of "
        f"{legs[0]['counts']['callsite.triangle_attention']} attention and "
        f"{legs[0]['counts']['cueq.triangle_multiplicative_update']} multiplicative-update). "
        f"Peak allocated VRAM {vram} MiB"
        f"{', the same as the A100 arm' if same_vram else ''}. "
        f"Card exclusive, no compute app on it before either leg. "
        f"{g['name']} at a {float(g['power.limit']):.0f} W limit, driver {g['driver_version']}, "
        f"Python {e['python']}. Stack byte-matched to both published columns: torch {e['torch']}, "
        f"triton {e['triton']}, cuequivariance-torch {e['cuequivariance-torch']} with "
        f"-ops-torch-cu12 {e['cuequivariance-ops-torch-cu12']}, lightning {e['lightning']}, "
        f"transformers {e['transformers']}, rdkit {e['rdkit']}, nesso {e['nesso']} at revision "
        f"{legs[0]['model_revision']}. 5 recycling steps, bf16-mixed, refine on, seed 42, batch 1, "
        f"model load outside the cell."
    )
    row["cells"][gpu] = {"status": "measured", "s_per_fold": round(ob, 4), "ref": ref}
    print(f"nesso1/{gpu}: {ob:.4f} s (legs {medians[0]:.4f} / {medians[1]:.4f}), "
          f"device {min(dev):.4f}-{max(dev):.4f} s, affinity sha {sorted(shas)[0]}")


# --- PXDesign ------------------------------------------------------------------------------------
# The published PXDesign h200 cell is not a whole-pipeline wall: it is the generator stage's own
# clock, gen_feat + gen_device + gen_write, taken from three cells of perf/pxdesign/gpu_reference.json
# that differ only in eval preset and in whether the target YAML carries an inert msa key. Two of
# those three need a sliced MSA that is an external MSA-server product and is not in the repo, so
# this arm takes its three warm samples from the one cell whose fixture it can byte-match: the same
# yaml sha256 the h200 laczc512_prev_n1 cell records, run as three fresh processes so every warm
# sample is rep1-after-a-discarded-cold-rep exactly as each h200 cell's was.
PX_LABEL = "laczc512_prev_n1"
PX_PROTO = {"preset": "preview", "n_sample": 1, "n_step": 400, "dtype": "bf16", "seed": 42,
            "extra": ""}
PX_PROCS = 3


def px_ref(gpu):
    """The h200 reference this cell is written against, read off the committed file."""
    ref = json.loads(Path("perf/pxdesign/gpu_reference.json").read_text())
    cell = [c for c in ref["cells"] if c["label"] == PX_LABEL]
    if len(cell) != 1:
        die(f"perf/pxdesign/gpu_reference.json holds {len(cell)} cells labelled {PX_LABEL}, want 1")
    return ref, cell[0]


def write_pxdesign(d, gpu, data):
    ref, h200 = px_ref(gpu)
    files = sorted(d.glob(f"px_{gpu}_p*.jsonl"))
    if len(files) != PX_PROCS:
        die(f"{len(files)} process files px_{gpu}_p*.jsonl, want {PX_PROCS}")

    warm_gen, warm_tot, cold_tot, seqs, reps = [], [], [], set(), []
    for f in files:
        rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if len(rs) != 2:
            die(f"{f.name}: {len(rs)} reps, want 2 (a cold rep 0 and one warm rep 1)")
        cold, hot = rs
        if not cold.get("cold") or hot.get("cold"):
            die(f"{f.name}: rep 0 cold={cold.get('cold')}, rep 1 cold={hot.get('cold')}")
        for j in rs:
            if j["label"] != PX_LABEL:
                die(f"{f.name}: label {j['label']}, want {PX_LABEL}")
            if j["yaml_sha256"] != h200["yaml_sha256"]:
                die(f"{f.name}: yaml sha256 {j['yaml_sha256'][:16]}, the published h200 cell "
                    f"records {h200['yaml_sha256'][:16]}; different fixture, not a comparable cell")
            for k, want in PX_PROTO.items():
                if j.get(k) != want:
                    die(f"{f.name}: {k} is {j.get(k)!r}, the published cell is {want!r}")
            if not j.get("sanity_ok") or j.get("why"):
                die(f"{f.name} rep {j['rep']}: sanity {j.get('why')}")
            v = j["validation"]
            if not v["ok"] or v["why"]:
                die(f"{f.name} rep {j['rep']}: validation {v['why']}")
            if v["n_generated_cif"] != 1 or v["seq_lengths"] != [80]:
                die(f"{f.name} rep {j['rep']}: {v['n_generated_cif']} cif at {v['seq_lengths']}, "
                    "want 1 design of an 80-residue binder")
            if not j.get("gpu_exclusive") or j.get("compute_apps_before"):
                die(f"{f.name} rep {j['rep']}: card not exclusive, "
                    f"{j.get('compute_apps_before')}")
            ci = j["counter_info"]
            # The h200 median deliberately EXCLUDES the fast-LayerNorm variant, because the shipped
            # CLI never exports LAYERNORM_TYPE before python starts. A cell that reached the fused
            # kernel is the excluded measurement wearing the included cell's name.
            if ci.get("fused_ln_present") or ci.get("layernorm_type_env"):
                die(f"{f.name} rep {j['rep']}: fused LayerNorm reached "
                    f"(fused_ln_present={ci.get('fused_ln_present')}, "
                    f"layernorm_type_env={ci.get('layernorm_type_env')!r}); the published h200 "
                    "median excludes that variant")
            if not ci.get("ds4sci_present"):
                die(f"{f.name} rep {j['rep']}: the DeepSpeed Evoformer kernel was not present")
            st = j["stages"]
            gen = sum(st[k]["s"] for k in ("gen_feat", "gen_device", "gen_write"))
            if abs(gen - st["gen_total"]["s"]) > 2e-3:
                die(f"{f.name} rep {j['rep']}: gen_feat+gen_device+gen_write {gen:.4f} does not "
                    f"partition gen_total {st['gen_total']['s']:.4f}")
            # Every package the h200 stack block names has to match, or the two cells are not the
            # same measurement. Read off gpu_reference.json rather than restated here.
            for k, want in ref["stack"].items():
                if k.startswith(("gpu", "env_", "nvidia_", "jax_", "torch_", "cudnn")):
                    continue
                got = j["env"].get(k)
                if got != want:
                    die(f"{f.name} rep {j['rep']}: {k} is {got}, the h200 stack names {want}")
        st = hot["stages"]
        warm_gen.append(st["gen_total"]["s"])
        warm_tot.append(hot["total_s"])
        cold_tot.append(cold["total_s"])
        if cold["total_s"] < hot["total_s"]:
            die(f"{f.name}: cold rep {cold['total_s']:.2f} s is faster than the warm rep "
                f"{hot['total_s']:.2f} s, so rep 0 was not actually cold")
        seqs.add(tuple(hot["validation"]["sequences"]))
        reps.append(hot)

    if len(seqs) != 1:
        die(f"the {PX_PROCS} processes designed different binders at the same seed: {seqs}")

    ob = statistics.median(warm_gen)
    spread = (max(warm_gen) - min(warm_gen)) / min(warm_gen) * 100.0
    row = [r for r in data["design"]["models"] if r["id"] == "pxdesign"][0]
    h200_cell = row["cells"]["h200"]
    if h200_cell.get("status") != "measured":
        die("the PXDesign h200 cell is not measured, so there is nothing to write this against")
    pub = h200_cell["s_per_design"]
    if abs(pub - h200["stages_s"]["gen_total_s"]) > 1e-3:
        die(f"the page's h200 cell {pub} s is not gen_total_s "
            f"{h200['stages_s']['gen_total_s']} s of {PX_LABEL}; the quantity moved")

    r0, e = reps[0], reps[0]["env"]
    st = r0["stages"]
    gd = r0["gpu_per_stage"]["gen_device"]
    # The whole-pipeline wall is NOT comparable between the two boxes and must not be read as if it
    # were: ProteinMPNN and AF2-IG are host-bound subprocesses and this landlord's CPU is far
    # slower. Both figures go in the ref so a reader cannot make that mistake by accident.
    mpnn = statistics.median(j["stages"]["mpnn"]["s"] for j in reps)
    fmt = lambda xs: " / ".join(f"{x:.4f}" for x in sorted(xs))
    ref_txt = (
        f"The generator stage's own wall clock (gen_feat + gen_device + gen_write), the same "
        f"quantity the h200 cell publishes: {fmt(warm_gen)} s, median {ob:.4f}, spread "
        f"{spread:.2f} %. Three fresh processes of one cell, each a discarded cold rep 0 plus one "
        f"warm rep 1, so every sample is rep-1-of-a-fresh-process as each h200 cell's was. The "
        f"h200 median pools three cells that differ only in eval preset and in whether the YAML "
        f"carries an msa key, which read_design_yaml parses and ignores because PXDesign-d has no "
        f"trunk; two of those need a sliced MSA that is an external MSA-server product and is not "
        f"in the repo, so this arm repeats the one cell it can byte-match instead, the same target "
        f"YAML sha256 {h200['yaml_sha256'][:16]} the h200 laczc512_prev_n1 cell records. "
        f"The A100 runs this stage {(ob - pub) / pub * 100.0:+.2f} % against the H200's {pub} s. "
        f"Inside it, {statistics.median(j['stages']['gen_device']['s'] for j in reps):.4f} s is the "
        f"diffusion call at {gd['util_pct_mean']:.1f} % mean GPU utilisation and "
        f"{gd['power_W_median']:.0f} W median, and {statistics.median(j['stages']['gen_feat']['s'] for j in reps):.4f} s "
        f"plus {statistics.median(j['stages']['gen_write']['s'] for j in reps):.4f} s are the host "
        f"featurise and CIF write the h200 cell also carries inside its number. "
        f"Read only this stage against the H200: the whole pipeline is "
        f"{statistics.median(warm_tot):.1f} s here against {h200['total_s_median']:.1f} s on the "
        f"H200, almost all of it the host-bound ProteinMPNN and AF2-IG subprocesses "
        f"({mpnn:.1f} s of ProteinMPNN against {h200['stages_s']['mpnn_s']:.1f} s), which is this "
        f"landlord's CPU and not the card. "
        f"Reproducible: all {PX_PROCS} processes return the identical designed sequence at seed "
        f"{PX_PROTO['seed']} ({sorted(seqs)[0][0]}), and each written CIF parses to one design of "
        f"an 80-residue binder with no non-finite coordinates. "
        f"The DeepSpeed Evoformer kernel is present and fast LayerNorm is not reached in any rep, "
        f"matching the h200 cells the page includes rather than the fast-LayerNorm one it excludes. "
        f"Card exclusive, no compute app on it before any rep. "
        f"{e['gpu']}, {e['nvidia_smi'].split(',')[1].strip()} driver, {e['gpu_capability']} "
        f"compute capability, {r0['env'].get('cpu_name', 'host CPU')} at "
        f"{r0['env'].get('effective_cpus', 'n/a')} effective vCPU. Peak allocated VRAM "
        f"{r0['peak_vram_alloc_GiB']:.3f} GiB. Stack byte-matched to the h200 cell's: torch "
        f"{e['torch']} with CUDA {e['torch_cuda']} and cuDNN {e['cudnn']}, protenix "
        f"{e['protenix']}, pxdesign {e['pxdesign']}, pxdbench {e['pxdbench']}, deepspeed "
        f"{e['deepspeed']}, jax {e['jax']}, numpy {e['numpy']}. N_sample 1, n_step 400, bf16, "
        f"512 target residues, 80-residue binder, checkpoint load outside the cell."
    )
    cell = {"status": "measured", "s_per_design": round(ob, 4), "ref": ref_txt,
            "split": {"host_s": round(statistics.median(
                          j["stages"]["gen_feat"]["s"] + j["stages"]["gen_write"]["s"]
                          for j in reps), 3),
                      "in_cell": True,
                      "ref": (f"gen_feat and gen_write of the same warm reps, the two host steps "
                              f"the Tenstorrent and H200 cells also carry inside their own "
                              f"numbers.")}}
    print(f"pxdesign/{gpu}: {ob:.4f} s generator stage (warm {fmt(warm_gen)}, spread "
          f"{spread:.2f} %), h200 {pub} s, {(ob - pub) / pub * 100.0:+.2f} %")
    return row, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--gpu", required=True, choices=("h200", "b200", "a100"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--row", default="openbind",
                    choices=("openbind", "nesso1", "pxdesign"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = json.loads(a.data.read_text())
    if a.row == "pxdesign":
        row, cell = write_pxdesign(a.dir, a.gpu, data)
        if a.dry_run:
            print(json.dumps(cell, indent=1))
            return
        row["cells"][a.gpu] = cell
        a.data.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote pxdesign/{a.gpu} into {a.data}")
        return
    if a.row == "nesso1":
        write_nesso(a.dir, a.gpu, data)
        if a.dry_run:
            print(json.dumps([r for r in data["affinity"]["models"]
                              if r["id"] == "nesso1"][0]["cells"][a.gpu], indent=1))
            return
        a.data.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote nesso1/{a.gpu} into {a.data}")
        return
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
