"""Does the device arm's miss ever flip a PXDesign filter decision?

`af2_easy` is PXDesign's success counter: pLDDT > 0.8, i_pTM > 0.5, i_pAE < 0.35 and
bound-unbound RMSD < 3.5, all four (`pxdbench/pxd_configs/eval.py:96-108`). The device arm's
accuracy question is therefore not a structure metric, it is whether the miss moves designs across
those lines. This harness scores a population on one arm, applies or measures the other, and diffs
the verdict. The RMSD criterion needs a second binder-only pass and is not wired up; the three
confidence criteria are.

The measured delta is read from the committed device-arm report by `measured_delta()` and is never
written into this file. The frozen number here until pass 12 was pass 6's 0.0846 of i_pTM, which
pass 9's float32 residual fix had already cut 32x to 0.0026, so a hardcoded delta had already gone
stale once.

Three populations, in the order they were tried:

* `--mode scramble` (pass 7). Chain A is the `laczc_128` crop, chain B the 64 residues that follow
  it in `laczc_256`, binder sequence randomised in nine steps. Flat and rejected at every rung: a
  sequence-contiguous chain split is not an interface, so there is nothing packed to lose and
  sequence quality stops being a lever (`parity_artifacts/laczc128_b80/filter_tolerance_scramble.json`).
* `--mode pose` (pass 8). The parity fixture's binder walked in from 60 A to interpenetration.
  Also flat, i_pTM spanning 0.0081 over the whole ladder, because `rm_template_ic=True` strips the
  template's interchain information and four recycles forget the initial guess
  (`filter_tolerance_pose.jsonl`).
* `--mode designs` (pass 12). Real designs, which is what the two flat ladders were standing in
  for: `design_population.py` writes a generated binder backbone against a real target with
  ProteinMPNN sequences, and that is the input the production filter grades. Rows come from its
  `population.jsonl`.

`--arm device` scores the same row on the ttnn trunk, so the delta is measured per design instead
of assumed constant across a population.

    PYTHONPATH=. python3 -u scripts/af2_port/filter_tolerance.py --mode designs \\
        --population .af2ig_p12/population.jsonl --arm torch --out scores_torch.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from af2_fixture import read_crop_cif, seq_of, write_complex_pdb  # noqa: E402

TARGET_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_256.cif"
FIXTURE_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_128.cif"
FIXTURE_BINDER = 80         # == the parity fixture, af2_fixture.build_fixture's default
TARGET_SLICE = (64, 192)    # == laczc_128.cif, verified equal residue for residue
BINDER_SLICE = (192, 256)   # the 64 residues that follow the crop, native pose
DEFAULT_PARAMS = "/home/ttuser/pxd_tool_weights/af2/params_model_1_ptm.npz"

# pxdbench/pxd_configs/eval.py:96-108. af2_easy is the success counter.
AF2_EASY = {"plddt": (">", 0.8), "i_ptm": (">", 0.5), "i_pae": ("<", 0.35)}

ARTIFACTS = Path(__file__).resolve().parent / "parity_artifacts" / "laczc128_b80"
# The device arm's own report is the only place this delta may come from. Hardcoding it went stale
# once already: the number in this file until pass 12 was 0.0846 of i_pTM, which pass 9's residual
# fix had already reduced 32x.
DEVICE_ARM = ARTIFACTS / "device_trunk_complex_rne_residual.json"
REFERENCE_ARM = ARTIFACTS / "torch_trunk_complex.json"


def measured_delta(path: Path = DEVICE_ARM) -> dict:
    """Signed device-minus-JAX delta per scalar, from the committed device-arm report."""
    report = json.loads(path.read_text())
    return {row["scalar"]: round(row["got"] - row["want"], 8) for row in report["scalars"]}

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def mutate(sequence: str, fraction: float, seed: int) -> str:
    """Randomise `fraction` of the positions. Seeded, so a rung is reproducible."""
    if fraction <= 0:
        return sequence
    rng = np.random.default_rng(seed)
    out = list(sequence)
    count = int(round(fraction * len(sequence)))
    for i in rng.choice(len(sequence), size=count, replace=False):
        out[i] = AA20[int(rng.integers(20))]
    return "".join(out)


def min_interchain_distance(target_res: list[dict], binder_res: list[dict], shift: float) -> float:
    """Closest approach between the two chains, so a rung records its own geometry."""
    t = np.array([[a[2], a[3], a[4]] for r in target_res for a in r["atoms"]])
    b = np.array([[a[2], a[3], a[4]] for r in binder_res for a in r["atoms"]])
    b = b + np.array([shift, 0.0, 0.0])
    return float(np.linalg.norm(t[:, None, :] - b[None, :, :], axis=-1).min())


def scramble_population(work: Path, levels: list[float], seed: int):
    """Pass 7's population: one native interface, binder sequence randomised in nine steps."""
    residues = read_crop_cif(str(TARGET_CIF))
    target = residues[TARGET_SLICE[0]:TARGET_SLICE[1]]
    binder = residues[BINDER_SLICE[0]:BINDER_SLICE[1]]
    native = seq_of(binder)
    pdb = str(work / "native_complex.pdb")
    write_complex_pdb(pdb, target, binder, shift=0.0)
    for level in levels:
        seq = mutate(native, level, seed + int(level * 1000))
        yield {"level": level,
               "identity": sum(a == b for a, b in zip(seq, native)) / len(native)}, pdb, seq


def pose_population(work: Path, shifts: list[float]):
    """The pose ladder: the parity fixture's binder walked in from 60 A, sequence held native."""
    residues = read_crop_cif(str(FIXTURE_CIF))
    binder = residues[:FIXTURE_BINDER]
    native = seq_of(binder)
    for shift in shifts:
        pdb = str(work / ("pose_%05.1f.pdb" % shift))
        write_complex_pdb(pdb, residues, binder, shift=shift)
        yield {"level": shift,
               "shift_a": shift,
               "min_interchain_a": round(min_interchain_distance(residues, binder, shift), 2)}, \
              pdb, native


def passes(scalars: dict) -> dict:
    return {k: (scalars[k] > bar if op == ">" else scalars[k] < bar)
            for k, (op, bar) in AF2_EASY.items()}


def load_arm(state, arm: str, template: bool = True):
    from tt_bio.af2_reference import load_af2_model
    if arm == "torch":
        return load_af2_model(state, template=template, trunk_dtype=torch.bfloat16)
    from tt_bio.af2 import load_af2_device_model
    return load_af2_device_model(state, template=template, trunk_dtype=torch.bfloat16)


def population_rows(path: Path):
    """The design population built by design_population.py: one row per (backbone, sequence)."""
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["pdb"], row["seq"])
        if key in seen:            # the MPNN fasta repeats its input sequence as entry 0
            continue
        seen.add(key)
        yield {k: row[k] for k in ("id", "design", "temp", "sample", "identity", "mpnn_score",
                                   "source", "tokens") if k in row}, row["pdb"], row["seq"]


def score(model, pdb: str, binder_seq: str, recycles: int) -> dict:
    from tt_bio.af2_confidence import confidence_scalars
    from tt_bio.af2_data import complex_features, initial_recycle_state
    from tt_bio.af2_reference import run_recycles

    feats_np = complex_features(pdb, binder_seq)
    prev_np = initial_recycle_state(feats_np)

    def to_torch(a):
        if a.dtype == np.bool_:
            return torch.from_numpy(a)
        if a.dtype.kind in "iu":
            return torch.from_numpy(a.astype(np.int64))
        return torch.from_numpy(a.astype(np.float32))

    feats = {k: to_torch(v) for k, v in feats_np.items()}
    prev = {k: to_torch(v) for k, v in prev_np.items()}
    last = None
    for out in run_recycles(model, feats, prev, num_recycles=recycles):
        last = out
    return confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                              feats["seq_mask"], feats["asym_id"],
                              binder_len=len(binder_seq))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--mode", default="scramble", choices=["scramble", "pose", "designs"])
    ap.add_argument("--arm", default="torch", choices=["torch", "device"])
    ap.add_argument("--population", default=None, help="designs mode: population.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--levels", default=None,
                    help="scramble: mutation fractions. pose: binder shifts in angstrom.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default="/tmp/af2ig_tolerance")
    args = ap.parse_args()

    default_levels = {"scramble": "0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0",
                      "pose": "60,50,46,44,42,40,38,36,34,32",
                      "designs": ""}
    levels = [float(x) for x in (args.levels or default_levels[args.mode]).split(",")
              if x.strip()]

    from tt_bio.af2_weights import load_af2_state_dict

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.mode == "designs":
        assert args.population, "--mode designs needs --population"
        population = list(population_rows(Path(args.population)))
        if args.limit:
            population = population[:args.limit]
    elif args.mode == "scramble":
        population = scramble_population(work, levels, args.seed)
    else:
        population = pose_population(work, levels)

    delta = measured_delta()
    model = load_arm(load_af2_state_dict(args.params), args.arm)

    out = Path(args.out)
    key = (lambda label: label["id"]) if args.mode == "designs" else (lambda label: label["level"])
    done = set()
    if out.exists():
        done = {key(json.loads(line)) for line in out.read_text().splitlines() if line.strip()}

    for label, pdb, seq in population:
        if key(label) in done:
            continue
        t0 = time.time()
        ref = score(model, pdb, seq, args.recycles)
        dev = {k: ref[k] + delta.get(k, 0.0) for k in ref}
        row = {
            "mode": args.mode,
            "arm": args.arm,
            **label,
            "seconds": round(time.time() - t0, 1),
            "ref": {k: round(v, 6) for k, v in ref.items()},
            "dev": {k: round(v, 6) for k, v in dev.items()},
            "ref_pass": passes(ref),
            "dev_pass": passes(dev),
        }
        row["flipped"] = sorted(k for k in AF2_EASY if row["ref_pass"][k] != row["dev_pass"][k])
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
