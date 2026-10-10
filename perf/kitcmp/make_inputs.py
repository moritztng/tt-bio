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
job = lambda i: {"name": f"fold{i}", "modelSeeds": [0], "sequences": [{"proteinChain": {
    "sequence": seq, "count": 1, "msa": {"precomputed_msa_dir": f"{R}/msa", "pairing_db": "uniref100"},
    "unpairedMsaPath": A3M, "pairedMsaPath": "", "templatesPath": ""}}]}
(a.out / "protenix.json").write_text(json.dumps([job(i) for i in range(N)], indent=1))
(a.out / "openfold3.json").write_text(json.dumps({"seeds": [0], "queries": {f"fold{i}": {
    "chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq, "main_msa_file_paths": [A3M]}],
    "use_msas": True, "use_paired_msas": False, "use_main_msas": True} for i in range(N)}}, indent=1))
(a.out / "openfold3.yaml").write_text("experiment_settings:\n  seeds: [0]\nmodel_update:\n  presets: [\"predict\"]\n"
    "  custom:\n    settings:\n      memory:\n        eval:\n          use_cueq_triangle_kernels: true\n")
(a.out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {R}/{n}\n"
    for p, n in ((m, "msa/uniref90_hits.a3m"), (y, "cdk2x2_512.yaml"))))
print("wrote", a.out, len(seq), "aa")
