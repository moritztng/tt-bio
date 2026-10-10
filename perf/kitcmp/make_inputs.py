"""Write the tt-bio.com perf-page fixture (cdk2x2_512.yaml + cdk2x2_512.a3m, 512 aa, 35 MSA rows) in each kit's input
format, four copies under four names (fold0..fold3) so one process does 1 cold + 3 warm folds with one weight load,
exactly as scripts/gpu_vs_tt/gpu_bench.py and gpu5_bench.py did for the published stock cells.

    python perf/kitcmp/make_inputs.py OUTDIR [--root /root/kc/in]

The a3m is copied byte for byte (as msa/uniref90_hits.a3m, the slot OF3 reads an unpaired main MSA from); sha256 of
both fixtures goes to OUTDIR/SHA256SUMS so the box re-verifies them.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--root", default="/root/kc/in")
a = ap.parse_args()
fx = Path(__file__).resolve().parents[1] / "size512/fixtures"
y, m = fx / "cdk2x2_512.yaml", fx / "cdk2x2_512.a3m"
seq = next(l.split(":", 1)[1].strip() for l in y.read_text().splitlines() if l.strip().startswith("sequence:"))
assert len(seq) == 512 and m.read_text().splitlines()[1] == seq
(a.out / "msa").mkdir(parents=True, exist_ok=True)
shutil.copyfile(m, a.out / "msa/uniref90_hits.a3m")
shutil.copyfile(y, a.out / "cdk2x2_512.yaml")
R, N = a.root, 4
A3M = f"{R}/msa/uniref90_hits.a3m"
(a.out / "boltz2").mkdir(exist_ok=True)
for i in range(N):  # the fixture yaml plus the msa key, the only change, as gpu5_bench run_boltz
    (a.out / f"boltz2/fold{i}.yaml").write_text(y.read_text().replace(f"sequence: {seq}", f"sequence: {seq}\n      msa: {A3M}"))
# unpairedMsaPath alone, as perf/mgx/ref/ref_fold.py: any other msa key that names no existing file (pairedMsaPath "")
# makes protenix/opendde need_msa_search() fetch a server MSA (24009 rows instead of the fixture's 35)
job = lambda i: {"name": f"fold{i}", "modelSeeds": [0], "sequences": [{"proteinChain": {
    "sequence": seq, "count": 1, "unpairedMsaPath": A3M}}]}
(a.out / "protenix.json").write_text(json.dumps([job(i) for i in range(N)], indent=1))
(a.out / "openfold3.json").write_text(json.dumps({"seeds": [0], "queries": {f"fold{i}": {
    "chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq, "main_msa_file_paths": [A3M]}],
    "use_msas": True, "use_paired_msas": False, "use_main_msas": True} for i in range(N)}}, indent=1))
(a.out / "openfold3.yaml").write_text("experiment_settings:\n  seeds: [0]\nmodel_update:\n  presets: [\"predict\"]\n"
    "  custom:\n    settings:\n      memory:\n        eval:\n          use_cueq_triangle_kernels: true\n")
(a.out / "esmfold2.json").write_text(json.dumps([{"id": f"fold{i}", "sequences": [{"type": "protein", "id": "A",
    "sequence": seq, "msa": None}], "seeds": [0], "num_diffusion_samples": 1} for i in range(N)], indent=1))
(a.out / "rf3.json").write_text(json.dumps([{"name": f"fold{i}", "components": [{"seq": seq, "chain_id": "A"}]}
    for i in range(2 * N)], indent=1))  # eight names in one engine.run: the kit's warm-up spans two folds, so four left 2 warm gaps
# RFdiffusion3: the published fixture staged at a root, so its repo-relative target path resolves (scripts/gpu_vs_tt/rfd3_prod_run.sh)
rf = Path(__file__).resolve().parents[1] / "dsfix"
(a.out / "rfd3/perf/dsfix/targets").mkdir(parents=True, exist_ok=True)
shutil.copyfile(rf / "fixtures/rfd3_R4_gpu.json", a.out / "rfd3/rfd3_R4_gpu.json")
shutil.copyfile(rf / "targets/R4_9q6y_A.pdb", a.out / "rfd3/perf/dsfix/targets/R4_9q6y_A.pdb")
px = Path(__file__).resolve().parents[1] / "pxdesign/targets"  # yaml byte for byte; box_arms.sh puts the CIF where it points
(a.out / "pxdesign").mkdir(exist_ok=True)
for n in ("laczc_512_nomsa.yaml", "laczc_512.cif"):
    shutil.copyfile(px / n, a.out / "pxdesign" / n)
(a.out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {R}/{n}\n"
    for p, n in ((m, "msa/uniref90_hits.a3m"), (y, "cdk2x2_512.yaml"), (rf / "fixtures/rfd3_R4_gpu.json", "rfd3/rfd3_R4_gpu.json"),
                 (rf / "targets/R4_9q6y_A.pdb", "rfd3/perf/dsfix/targets/R4_9q6y_A.pdb"),
                 (px / "laczc_512_nomsa.yaml", "pxdesign/laczc_512_nomsa.yaml"), (px / "laczc_512.cif", "pxdesign/laczc_512.cif"))))
print("wrote", a.out, len(seq), "aa")
